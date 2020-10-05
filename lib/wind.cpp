#include "wind.h"


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


Wind3d_turboframe::Wind3d_turboframe(const param& params) {
    std::string v_path = params.s.at("windv_file_path");
    std::string q_path = params.s.at("windq_file_path");

    std::ifstream v_file (v_path);
    if (!v_file.is_open())
        throw std::runtime_error("Error in opening the wind file at "+v_path);

    std::string line;
    vecd l = vecd(3);
    int count = 0;
    while ( getline (v_file, line) ){
        l = str2vecd(line, " ", false);
        v_grid[count][0] = l[0];
        v_grid[count][1] = l[1];
        v_grid[count][2] = l[2];
        count++;
    }
    v_file.close();

    std::ifstream q_file (q_path);
    if (!q_file.is_open())
        throw std::runtime_error("Error in opening the wind file at "+q_path);

    count = 0;
    while ( getline (q_file, line) ){
        l = str2vecd(line, " ", false);
        q_grid[count][0] = l[0];
        q_grid[count][1] = l[1];
        q_grid[count][2] = l[2];
        count++;
    }
    q_file.close();
}


double* Wind3d_turboframe::velocity(double x, double y, double z){

    int mx = x/x_size;
    x -= mx*x_size;

    y -= y_size;
    if (y < -y_size || y > y_size)
        throw std::runtime_error("y out of bounds");

    int mz = z/z_size;
    z -= mz*z_size;

    int n_x, n_y, n_z;

    for (size_t i = 0; i < n_axis_points-1; i++) {
        if (z>=q_grid[i][2] && z<q_grid[i+1][2]){
          n_z=i;
          break;
        }
    }

    for (size_t i = 0; i < n_axis_points-1; i++) {
        if (x>=q_grid[i*n_axis_points][0] && x<q_grid[(i+1)*n_axis_points][0]){
          n_x=i;
          break;
        }
    }

    for (size_t i = 0; i < n_axis_points-1; i++) {
        if (y>=q_grid[i*n_axis_points*n_axis_points][1] && y<q_grid[(i+1)*n_axis_points*n_axis_points][1]){
          n_y=i;
          break;
        }
    }

    int ind=n_y*n_axis_points*n_axis_points+n_x*n_axis_points+n_z;

    double q_d[3];
    q_d[2]=(z-q_grid[ind][2])/(q_grid[ind+1][2]-q_grid[ind][2]);
    q_d[0]=(x-q_grid[ind][0])/(q_grid[ind+n_axis_points][0]-q_grid[ind][0]);
    q_d[1]=(y-q_grid[ind][1])/(q_grid[ind+n_axis_points*n_axis_points][1]-q_grid[ind][1]);

    double vel_corner[8];
    for (size_t i=0; i<3; i++){
        vel_corner[0] = v_grid[ind][i];
        vel_corner[1] = v_grid[ind+1][i];
        vel_corner[2] = v_grid[ind+n_axis_points*n_axis_points+1][i];
        vel_corner[3] = v_grid[ind+n_axis_points*n_axis_points][i];
        vel_corner[4] = v_grid[ind+n_axis_points][i];
        vel_corner[5] = v_grid[ind+n_axis_points+1][i];
        vel_corner[6] = v_grid[ind+n_axis_points*n_axis_points+n_axis_points+1][i];
        vel_corner[7] = v_grid[ind+n_axis_points*n_axis_points+n_axis_points][i];
        m_vel[i] = interpolation(q_d, vel_corner);

    }
    return m_vel;
}


Wind3d* get_wind3d(const param& params) {
    
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
    if (wind_type == "turboframe") { // Logarithmic wind
        return new Wind3d_turboframe(params);
    }
    
    else throw std::invalid_argument( "Invalid wind type" );
}



double interpolation(double q_d[], double vel[]){
   double vel_x[4];
   double vel_xy[2];
   vel_x[0]=vel[0]*(1-q_d[0])+vel[1]*q_d[0];
   vel_x[1]=vel[3]*(1-q_d[0])+vel[2]*q_d[0];
   vel_x[2]=vel[4]*(1-q_d[0])+vel[5]*q_d[0];
   vel_x[3]=vel[7]*(1-q_d[0])+vel[6]*q_d[0];

   vel_xy[0]=vel_x[0]*(1-q_d[1])+vel_x[2]*q_d[1];
   vel_xy[1]=vel_x[1]*(1-q_d[1])+vel_x[3]*q_d[1];

   return vel_xy[0]*(1-q_d[2])+vel_xy[1]*q_d[2];
}
