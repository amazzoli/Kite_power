#ifndef QALG_H
#define QALG_H


#include "alg.h"


/* Generic algorithm for the quality update and epsilon-greed expolration. The exploration 
   and the learning rate parameters have a power-law shape. */ 
class QAlg_eps : public RLAlgorithm {

    protected:

        /* Quality function */
        vec2d quality;
        /* Quality function trajectory */
        vec3d quality_traj;
        /* Which aggregate states are stored in the trajectory (to save space) */
        veci traj_states;
        /* Learning rate scheduling */
        d_i_fnc lr;
        /* Expolation scheduling */
        d_i_fnc eps;
        /* Uniform random distribution */ 
        std::uniform_real_distribution<double> unif_dist;
        /* Uniform distribution over the actions */
        std::uniform_int_distribution<int> unif_act_dist;

        // OVERRIDED FUNCTIONS
        virtual void init(const param& params);
        virtual int get_action();
        virtual void build_traj();
        virtual void print_traj(std::string out_dir) const;

        // ABSTRACT FUNCTIONS
        virtual void learning_update() = 0;

        // AUX METOHDS
        vec2d const_quals(double val);

    public:

        /* Construct the algorithm given the parameters dictionary */
        QAlg_eps(Environment* env, const param& params, std::mt19937& generator);

        /* Algorithm description */
        virtual const std::string descr() const = 0;
};


class SARSA_eps : public QAlg_eps {

    private:

        int old_state;
        int old_action;
        double old_reward;

    protected:

        virtual void learning_update();

    public:

        SARSA_eps(Environment* env, const param& params, std::mt19937& generator) :
        QAlg_eps(env, params, generator) {};

        virtual const std::string descr() const { return "SARSA algorithm with epsilon greedy exploration."; }
};


class QL_eps : public QAlg_eps {

    protected:

        virtual void learning_update();

    public:

        QL_eps(Environment* env, const param& params, std::mt19937& generator) :
        QAlg_eps(env, params, generator) {};

        virtual const std::string descr() const { return "Q-learning algorithm with epsilon greedy exploration."; }
};


#endif