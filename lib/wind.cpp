#include "wind.h"


// 2D WINDS

Wind2d* get_wind2d(const param& params) {

    std::string wind_type = params.s.at("wind_type");

    if (wind_type == "const") { // Constant wind
        double v[] = {params.d.at("v_wind_x"), params.d.at("v_wind_y")};
        return new Wind2d_const(v);
    }
    if (wind_type == "stream") { // With simple stream function
        return new Wind2d_stream(params.d.at("k_wind"), params.d.at("eps_wind"));
    }
    else throw std::invalid_argument( "Invalid wind type" );
}


double* Wind2d_stream::velocity(double x, double y){
    double aux1 = sin(PI*x/period[0]);
    double aux2 = sin(PI*y/period[1]);
    m_vel[0] = 0.5*k_wind*y*(2*eps_wind*aux1*aux2 + eps_wind*PI*y/period[1]*aux1*cos(PI*y/period[1]) + 2);
    m_vel[1] = -k_wind*eps_wind*PI*y*y/(2*period[0])*cos(PI*x/period[0])*aux2;
    return m_vel;
}


// 3D WINDS

double* Wind3d_log::velocity(double x, double y, double z, double t) {
    double arg = z/z0;
    if (z/z0 <= 0)
        m_vel[0] = 0;
    else
        m_vel[0] = vr * log(arg);

    return m_vel;
}

Wind3d_lognoise::Wind3d_lognoise(double vr, double z0, const vecd std, std::mt19937& generator) :
vr{vr}, z0{z0}, generator{generator} {
    normalx = std::normal_distribution<double>(0.0f, std[0]);
    normaly = std::normal_distribution<double>(0.0f, std[1]);
    normalz = std::normal_distribution<double>(0.0f, std[2]);
};

double* Wind3d_lognoise::velocity(double x, double y, double z, double t) {
    double arg = z/z0;
    if (z/z0 <= 0)
        m_vel[0] = normalx(generator);
    else
        m_vel[0] = vr * log(arg) + normalx(generator);
    m_vel[1] = normaly(generator);
    m_vel[2] = normalz(generator);
    return m_vel;
}

Wind3d_turboframe::Wind3d_turboframe(const param& params) {
    std::string v_path = params.s.at("windv_file_path");
    std::string q_path = params.s.at("windq_file_path");
    wind_amplif = params.d.at("wind_amplification");

    read_grid_file(q_path, q_grid);
    read_grid_file(v_path, v_grid);
}

void Wind3d_turboframe::read_grid_file(std::string path, double grid_data[][3]){
    std::ifstream file (path);
    if (!file.is_open())
        throw std::runtime_error("Error in opening the wind file at "+path);

    std::string line;
    vecd l = vecd(3);
    int count = 0;
    while ( getline (file, line) ){
        l = str2vecd(line, " ", false);
        grid_data[count][0] = l[0];
        grid_data[count][1] = l[2];
        grid_data[count][2] = l[1];
        count++;
    }
    file.close();
}


double* Wind3d_turboframe::init(double x, double y, double z) {
    //std::cout << x << " " << y << " "<< z << "\n";
    // Imposing the periodic boundary condition on the x
    int mx = x/x_size;
    x -= mx*x_size;

    // Translating the y such that 0 is in the middle of the canal and imposing boundary conditions
    y += y_size / 2.0;
    int my = y / y_size;
    y -= my*y_size;

    // Translating the z such that 0 is on the ground (we assume that z doesn't goes out of bounds)
    // Below the ground the velocity is the one on the ground
    z -= z_half_size;

    //std::cout << x << " " << y << " "<< z << "\n";
    for (size_t i = 0; i < n_y_axis_points-1; i++) {
        if (y>=q_grid[i][1] && y<q_grid[i+1][1]){
          n_y=i;
          break;
        }
    }
    //std::cout << n_y << "\n";
    for (size_t i = 0; i < n_x_axis_points-1; i++) {
        //std::cout << i << " " << q_grid[i*n_xy_axis_points][0] << " " << q_grid[(i+1)*n_xy_axis_points][0] << "\n";
        if (x>=q_grid[i*n_y_axis_points][0] && x<q_grid[(i+1)*n_y_axis_points][0]){
          n_x=i;
          break;
        }
    }
    //std::cout << n_x << "\n";
    if (z < -z_half_size) n_z = 0;
    else {
        for (size_t i = 0; i < n_z_axis_points-1; i++) {
            //std::cout << i << " " << q_grid[i*n_xy_axis_points*n_xy_axis_points][1] << " " << q_grid[i+(n_xy_axis_points*n_xy_axis_points)][1] << "\n";
            if (z>=q_grid[i*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(i+1)*n_x_axis_points*n_y_axis_points][2]){
              n_z=i;
              /*if ((i+1)*n_xy_axis_points*n_xy_axis_points > n_grid_points-1) {
                  std::cout << "index out of bounds" << '\n';}*/
              break;
            }
        }
    }
    //std::cout << n_x << " " << n_y << " " << n_z << "\n";
    return compute_velocity(x, y, z, 0);
}

double* Wind3d_turboframe::velocity(double x, double y, double z, double t){

    int frame = t / delta_time;
    frame = frame % n_frames;

    // Imposing the periodic boundary condition on the x
    int mx = x/x_size;
    x -= mx*x_size;

    // Translating the y such that 0 is in the middle of the canal and imposing boundary conditions
    y += y_size / 2.0;
    int my = y / y_size;
    y -= my*y_size;

    // Translating the z such that 0 is on the ground (we assume that z doesn't goes out of bounds)
    // Below the ground the velocity is the one on the ground
    z -= z_half_size;

    if (!(y>=q_grid[n_y][1] && y<q_grid[n_y+1][1]))
    {
        if (n_y != 0 && y>=q_grid[n_y-1][1] && y<q_grid[n_y][1]) n_y -= 1;
        else if (n_y == 0 && y>=q_grid[n_y_axis_points-2][1] && y<q_grid[n_y_axis_points-1][1]) n_y = n_y_axis_points-2;
        else if (n_y != n_y_axis_points-2 && y>=q_grid[n_y+1][1] && y<q_grid[n_y+2][1]) n_y += 1;
        else if (n_y == n_y_axis_points-2 && y>=q_grid[0][1] && y<q_grid[1][1]) n_y = 0;
        else {
            for (size_t i = 0; i < n_y_axis_points-1; i++) {
                if (y>=q_grid[i][1] && y<q_grid[i+1][1]){
                  n_y=i;
                  break;
                }
            }
        }
    }

    if (!(x>=q_grid[n_x*n_y_axis_points][0] && x<q_grid[(n_x+1)*n_y_axis_points][0])){
        if (n_x != 0 && x>=q_grid[(n_x-1)*n_y_axis_points][0] && x<q_grid[n_x*n_y_axis_points][0]) n_x -= 1;
        else if (n_x == 0 && x>=q_grid[(n_x_axis_points-2)*n_y_axis_points][0] && x<q_grid[(n_x_axis_points-1)*n_y_axis_points][0]) n_x = n_x_axis_points-2;
        else if (n_x != n_x_axis_points-2 && x>=q_grid[(n_x+1)*n_y_axis_points][0] && x<q_grid[(n_x+2)*n_y_axis_points][0]) n_x += 1;
        else if (n_x == n_x_axis_points-2 && x>=q_grid[0][0] && x<q_grid[n_y_axis_points][0]) n_x = 0;
        else {
            for (size_t i = 0; i < n_x_axis_points-1; i++) {
                if (x>=q_grid[i*n_y_axis_points][0] && x<q_grid[(i+1)*n_y_axis_points][0]){
                  n_x=i;
                  break;
                }
            }
        }
    }

    if (!(z>=q_grid[n_z*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(n_z+1)*n_x_axis_points*n_y_axis_points][2])) {
        if (z < -z_half_size) n_z = 0;
        else {
            if (n_z != 0 && z>=q_grid[(n_z-1)*n_x_axis_points*n_y_axis_points][2] && z<q_grid[n_z*n_x_axis_points*n_y_axis_points][2]) n_z -= 1;
            else if (n_z != n_z_axis_points-2 && z>=q_grid[(n_z+1)*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(n_z+2)*n_x_axis_points*n_y_axis_points][2]) n_z += 1;
            else {
                for (size_t i = 0; i < n_z_axis_points-1; i++) {
                    if (z>=q_grid[i*n_x_axis_points*n_y_axis_points][2] && z<q_grid[(i+1)*n_x_axis_points*n_y_axis_points][2]){
                      n_z=i;
                      break;
                    }
                }
            }
        }
    }
    //std::cout << n_x << " " << n_y << " " << n_z << "\n";

    return compute_velocity(x, y, z, frame);
}


double* Wind3d_turboframe::compute_velocity(double x, double y, double z, int frame) {
    int ind=n_z*n_x_axis_points*n_y_axis_points+n_x*n_y_axis_points+n_y;
    //std::cout << x << " " << y << " "<< z << " " << n_x << " " << n_y << " "<< n_z << "\n";

    double q_d[3];
    q_d[0]=(x-q_grid[ind][0])/(q_grid[ind+n_y_axis_points][0]-q_grid[ind][0]);
    q_d[1]=(y-q_grid[ind][1])/(q_grid[ind+1][1]-q_grid[ind][1]);
    q_d[2]=(z-q_grid[ind][2])/(q_grid[ind+n_x_axis_points*n_y_axis_points][2]-q_grid[ind][2]);
    //std::cout << q_d[0] << " " << q_d[1] << " "<< q_d[2] << "\n";

    double vel_corner[8];
    for (size_t i=0; i<3; i++){
        vel_corner[0] = v_grid[ind][i];
        vel_corner[1] = v_grid[ind+1][i];
        vel_corner[2] = v_grid[ind+n_x_axis_points*n_y_axis_points+1][i];
        vel_corner[3] = v_grid[ind+n_x_axis_points*n_y_axis_points][i];
        vel_corner[4] = v_grid[ind+n_y_axis_points][i];
        vel_corner[5] = v_grid[ind+n_y_axis_points+1][i];
        vel_corner[6] = v_grid[ind+n_x_axis_points*n_y_axis_points+n_y_axis_points+1][i];
        vel_corner[7] = v_grid[ind+n_x_axis_points*n_y_axis_points+n_y_axis_points][i];
        //std::cout << vel_corner[0] << " " << vel_corner[1] << " "<< vel_corner[2] << " "<< vel_corner[3] << " ";
        //std::cout << vel_corner[4] << " " << vel_corner[5] << " "<< vel_corner[6] << " "<< vel_corner[7] << "\n";
        m_vel[i] = interpolation(q_d, vel_corner)*wind_amplif;
        //std::cout << m_vel[i] << " ";
    }
    //std::cout << '\n';

    return m_vel;
}


double Wind3d_turboframe::interpolation(double q_d[], double vel[]){
   double vel_x[4];
   double vel_xy[2];
   vel_x[0]=vel[0]*(1-q_d[0])+vel[4]*q_d[0];
   vel_x[1]=vel[1]*(1-q_d[0])+vel[5]*q_d[0];
   vel_x[2]=vel[2]*(1-q_d[0])+vel[6]*q_d[0];
   vel_x[3]=vel[3]*(1-q_d[0])+vel[7]*q_d[0];

   vel_xy[0]=vel_x[0]*(1-q_d[1])+vel_x[1]*q_d[1];
   vel_xy[1]=vel_x[3]*(1-q_d[1])+vel_x[2]*q_d[1];

   return vel_xy[0]*(1-q_d[2])+vel_xy[1]*q_d[2];
}


Wind3d_turbo::Wind3d_turbo(const param& params) {
    std::string v_dir = params.s.at("windv_file_dir");
    std::string v_name = params.s.at("windv_file_name");
    int start_frame = params.d.at("start_frame");
    std::string q_path = params.s.at("windq_file_path");
    wind_amplif = params.d.at("wind_amplification");

    read_grid_file(q_path, q_grid);
    read_grid_files(v_dir, v_name, start_frame, vt_grid);
}


void Wind3d_turbo::read_grid_files(std::string dir, std::string name, int start_frame, float grid_data[][n_grid_points][3]){

    for (int t=0; t<n_frames; t++) {
        std::cout << t <<"\n";
        std::string path = dir + name + std::to_string(t+start_frame) + ".txt";
        std::ifstream file (path);
        if (!file.is_open())
            throw std::runtime_error("Error in opening the wind file at "+path);

        std::string line;
        vecd l = vecd(3);
        int count = 0;
        while ( getline (file, line) ){
            l = str2vecd(line, " ", false);
            grid_data[t][count][0] = l[0];
            grid_data[t][count][1] = l[2];
            grid_data[t][count][2] = l[1];
            count++;
        }
        file.close();
    }
}


double* Wind3d_turbo::compute_velocity(double x, double y, double z, int frame) {
    int ind=n_z*n_x_axis_points*n_y_axis_points+n_x*n_y_axis_points+n_y;
    //std::cout << x << " " << y << " "<< z << " " << n_x << " " << n_y << " "<< n_z << "\n";

    double q_d[3];
    q_d[0]=(x-q_grid[ind][0])/(q_grid[ind+n_y_axis_points][0]-q_grid[ind][0]);
    q_d[1]=(y-q_grid[ind][1])/(q_grid[ind+1][1]-q_grid[ind][1]);
    q_d[2]=(z-q_grid[ind][2])/(q_grid[ind+n_x_axis_points*n_y_axis_points][2]-q_grid[ind][2]);
    //std::cout << q_d[0] << " " << q_d[1] << " "<< q_d[2] << "\n";

    double vel_corner[8];
    for (size_t i=0; i<3; i++){
        vel_corner[0] = vt_grid[frame][ind][i];
        vel_corner[1] = vt_grid[frame][ind+1][i];
        vel_corner[2] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points+1][i];
        vel_corner[3] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points][i];
        vel_corner[4] = vt_grid[frame][ind+n_y_axis_points][i];
        vel_corner[5] = vt_grid[frame][ind+n_y_axis_points+1][i];
        vel_corner[6] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points+n_y_axis_points+1][i];
        vel_corner[7] = vt_grid[frame][ind+n_x_axis_points*n_y_axis_points+n_y_axis_points][i];
        //std::cout << vel_corner[0] << " " << vel_corner[1] << " "<< vel_corner[2] << " "<< vel_corner[3] << " ";
        //std::cout << vel_corner[4] << " " << vel_corner[5] << " "<< vel_corner[6] << " "<< vel_corner[7] << "\n";
        m_vel[i] = interpolation(q_d, vel_corner)*wind_amplif;
        //std::cout << m_vel[i] << " ";
    }
    //std::cout << '\n';

    return m_vel;
}



Wind3d* get_wind3d(const param& params, std::mt19937& generator) {

    std::string wind_type = params.s.at("wind_type");
    if (wind_type == "const") { // Constant wind
        double v[] = {params.d.at("v_wind_x"), params.d.at("v_wind_y"), params.d.at("v_wind_z")};
        return new Wind3d_const(v);
    }
    if (wind_type == "lin") { // Linear wind
        return new Wind3d_lin(params.d.at("v_ground"), params.d.at("v_ang_coef"));
    }
    if (wind_type == "log") { // Logarithmic wind
        return new Wind3d_log(params.d.at("vr"), params.d.at("z0"));
    }
    if (wind_type == "lognoise") { // Logarithmic wind with noise
        return new Wind3d_lognoise(params.d.at("vr"), params.d.at("z0"), params.vecd.at("std"), generator);
    }
    if (wind_type == "turboframe") { // Logarithmic wind
        return new Wind3d_turboframe(params);
    }
    if (wind_type == "turbo") { // Logarithmic wind
        return new Wind3d_turbo(params);
    }

    else throw std::invalid_argument( "Invalid wind type" );
}