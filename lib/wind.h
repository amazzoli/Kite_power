#ifndef WIND_H
#define WIND_H

#include "utils.h"


// 2D WINDS

class Wind2d {
    protected:
        double m_vel[2];
    public:
        virtual double* velocity(double x, double y) = 0;
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


// 3D WINDS

/* Abstract class */
class Wind3d {
    protected:
        /* Velocity variable that is referenced to by velocity method*/
        double m_vel[3];

    public:
        virtual double* init(double x0, double y0, double z0) { return velocity(x0,y0,z0,0.0); };
        virtual double* velocity(double x, double y, double z, double t) = 0;
        virtual const std::string descr() const = 0;
};

/* Constant wind */
class Wind3d_const : public Wind3d {
    public:
        Wind3d_const(double vel[3]) { m_vel[0] = vel[0]; m_vel[1] = vel[1]; m_vel[2] = vel[2];  };
        virtual double* velocity(double x, double y, double z, double t) { return m_vel; }
        const std::string descr() const { return "3d constant wind."; }
};

/* Linear wind */
class Wind3d_lin : public Wind3d {

    private:
        /* Wind x speed on the gorund */
        double vel_ground;
        /* Angular coefficient of the wind profile */
        double ang_coef;

    public:
        Wind3d_lin(double vel_ground, double ang_coef) : vel_ground{vel_ground}, ang_coef{ang_coef}
        { m_vel[1] = 0; m_vel[2]=0; };

        virtual double* velocity(double x, double y, double z, double t) {
            m_vel[0] = ang_coef * z + vel_ground;
            return m_vel;
        }

        const std::string descr() const
        { return "3d wind parallel to x-axis and linearly increasing with the height."; }
};

/* Logarithmic wind */
class Wind3d_log : public Wind3d {

    private:
        double vr;
        double z0;

    public:
        Wind3d_log(double vr, double z0) : vr{vr}, z0{z0}
        { m_vel[1] = 0; m_vel[2] = 0; };
        virtual double* velocity(double x, double y, double z, double t);
        const std::string descr() const
        { return "3d wind parallel to x-axis and logarithmically increasing with the height."; }
};

/* Logarithmic wind plus noise */
class Wind3d_lognoise : public Wind3d {

    private:
        double vr;
        double z0;
        std::mt19937 generator;
        std::normal_distribution<double> normalx;
        std::normal_distribution<double> normaly;
        std::normal_distribution<double> normalz;

    public:
        Wind3d_lognoise(double vr, double z0, const vecd std, std::mt19937& generator);
        virtual double* velocity(double x, double y, double z, double t);
        const std::string descr() const
        { return "3d wind parallel to x-axis and logarithmically increasing with the height. Additive gaussian noise."; }
};

/* Wind of a static frame of a turbolent flow */
class Wind3d_turboframe : public Wind3d {

    protected:
        const static int n_grid_points = 185193;
        //const static int n_grid_points = 499059;
        const static int n_x_axis_points = 57;
        const static int n_y_axis_points = 57;
        const static int n_z_axis_points = 57;
        //const static int n_x_axis_points = 71;
        //const static int n_y_axis_points = 71;
        //const static int n_z_axis_points = 99;
        constexpr static double x_size = 100.531;
        constexpr static double y_size = 100.531;
        constexpr static double z_half_size = 50;

        const static int n_frames = 3000;
        const double delta_time = 0.2;

        double q_grid[n_grid_points][3];
        double v_grid[n_grid_points][3];
        double wind_amplif;
        int n_x, n_y, n_z;

        void read_grid_file(std::string path, double grid_data[][3]);
        double interpolation(double q_d[], double vel[]);
        virtual double* compute_velocity(double x, double y, double z, int t);

    public:
        Wind3d_turboframe() {};
        Wind3d_turboframe(const param& params);
        virtual double* init(double x, double y, double z);
        virtual double* velocity(double x, double y, double z, double t);

        virtual const std::string descr() const
        { return "3d wind. Static frame of a turbolent flow."; }
};


/* Wind of a sequence of frames of a turbolent flow */
class Wind3d_turbo : public Wind3d_turboframe {
    private:

        float vt_grid[n_frames][n_grid_points][3];

        void read_grid_files(std::string dir, std::string name, int start_frame);
        double* compute_velocity(double x, double y, double z, int frame);

    public:
        Wind3d_turbo(const param& params);
        //virtual double* init(double x, double y, double z);
        //virtual double* velocity(double x, double y, double z, double t);

        const std::string descr() const
        { return "3d wind. Sequence of frames of a turbolent flow. Max " + std::to_string(n_frames*delta_time) + " seconds."; }
};



Wind2d* get_wind2d(const param& params);

Wind3d* get_wind3d(const param& params, std::mt19937& generator);

double interpolation(double q_d[], double vel[]);


#endif
