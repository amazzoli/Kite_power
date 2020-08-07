#ifndef WIND_H
#define WIND_H

#include "utils.h"


class Wind2d {
    protected:
        double m_vel[2];
    public:
        virtual double* velocity(double x, double y) = 0;
};


class Wind3d {
    protected:
        double m_vel[3];
    public:
        virtual double* velocity(double x, double y, double z) = 0;
};


class Wind2d_const : public Wind2d {
    public:
        Wind2d_const(double vel[2]) { m_vel[0] = vel[0]; m_vel[1] = vel[1];  };
        virtual double* velocity(double x, double y) { return m_vel; }
};   


class Wind2d_stream : public Wind2d {
    private:
        double k_wind;
        double eps_wind;
        double period[2] {50, 50};
    public:
        Wind2d_stream(double k_wind, double eps_wind) : k_wind{k_wind}, eps_wind{eps_wind} {};
        virtual double* velocity(double x, double y);
};   


Wind2d* get_wind2d(const dictd& params);
//Wind3d* get_wind3d(const dictd& params);


#endif