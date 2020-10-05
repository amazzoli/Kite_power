#ifndef WIND_H
#define WIND_H

#include "utils.h"


class Wind2d {
    protected:
        double m_vel[2];
    public:
        virtual double* velocity(double x, double y) = 0;
        virtual const std::string descr() const = 0;
};


class Wind3d {
    protected:
        double m_vel[3];

    public:
        virtual double* velocity(double x, double y, double z) = 0;
        virtual const std::string descr() const = 0;
};


class Wind2d_const : public Wind2d {
    public:
        Wind2d_const(double vel[2]) { m_vel[0] = vel[0]; m_vel[1] = vel[1];  };
        virtual double* velocity(double x, double y) { return m_vel; }
        const std::string descr() const { return "2d constant wind."; }
};   


class Wind2d_stream : public Wind2d {
    private:
        double k_wind;
        double eps_wind;
        double period[2] {50, 50};
    public:
        Wind2d_stream(double k_wind, double eps_wind) : k_wind{k_wind}, eps_wind{eps_wind} {};
        virtual double* velocity(double x, double y);
        const std::string descr() const { return "2d wind with stram function."; }
};   


class Wind3d_const : public Wind3d {
    public:
        Wind3d_const(double vel[3]) { m_vel[0] = vel[0]; m_vel[1] = vel[1]; ; m_vel[2] = vel[2];  };
        virtual double* velocity(double x, double y, double z) { return m_vel; }
        const std::string descr() const { return "3d constant wind."; }
};   


class Wind3d_lin : public Wind3d {

    private:
        /* Wind x speed on the gorund */
        double vel_ground;
        /* Angular coefficient of the wind profile */
        double ang_coef;

    public:
        Wind3d_lin(double vel_ground, double ang_coef) : vel_ground{vel_ground}, ang_coef{ang_coef} 
        { m_vel[1] = 0; m_vel[2]=0; };

        virtual double* velocity(double x, double y, double z) { 
            m_vel[0] = ang_coef * z + vel_ground;
            return m_vel; 
        }

        const std::string descr() const 
        { return "3d wind parallel to x-axis and linearly increasing with the height."; }
};  


class Wind3d_log : public Wind3d {

    private:
        double vr;
        double z0;

    public:
        Wind3d_log(double vr, double z0) : vr{vr}, z0{z0} 
        { m_vel[1] = 0; m_vel[2]=0; };

        virtual double* velocity(double x, double y, double z) { 
            double arg = z/z0;
            if (z/z0 <= 0) 
                m_vel[0] = 0;
            else 
                m_vel[0] = vr * log(arg);
                
            return m_vel; 
        }

        const std::string descr() const 
        { return "3d wind parallel to x-axis and logarithmically increasing with the height."; }
};  


class Wind3d_turboframe : public Wind3d {

    private:
        const static int n_grid_points = 185193;
        const static int n_axis_points = 57;
        constexpr static double x_size = 100.531;
        constexpr static double y_size = 50;
        constexpr static double z_size = 100.531;
        double q_grid[n_grid_points][3];
        double v_grid[n_grid_points][3];

    public:
        Wind3d_turboframe(const param& params);

        virtual double* velocity(double x, double y, double z);

        const std::string descr() const 
        { return "3d wind. Static frame of a turbolent flow."; }
};  


Wind2d* get_wind2d(const param& params);

Wind3d* get_wind3d(const param& params);

double interpolation(double q_d[], double vel[]);


#endif