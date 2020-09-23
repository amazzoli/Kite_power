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
    
    else throw std::invalid_argument( "Invalid wind type" );
}