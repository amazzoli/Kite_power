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


// void streamfunction3d_hard(double *rk, double *W){ // kp kiteposition in (x,y,z)

//     W[0] = 0.5*k_wind*rk[2]*(2*epsilon_wind_hard*sin(PI*rk[0]/Lx)*sin(PI*rk[2]/Ly) + \
//       epsilon_wind_hard*PI*rk[2]/Ly*sin(PI*rk[0]/Lx)*cos(PI*rk[2]/Ly) + 2);
//     W[1] = 0;
//     W[2] = -k_wind*epsilon_wind_hard*PI*rk[2]*rk[2]/(2*Lx)*cos(PI*rk[0]/Lx)*sin(PI*rk[2]/Ly);
// }


Wind3d* get_wind3d(const param& params) {
    
    std::string wind_type = params.s.at("wind_type");

    if (wind_type == "const") { // Constant wind
        double v[] = {params.d.at("v_wind_x"), params.d.at("v_wind_y"), params.d.at("v_wind_z")};
        return new Wind3d_const(v);
    }
    
    else throw std::invalid_argument( "Invalid wind type" );
}