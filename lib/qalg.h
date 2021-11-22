#ifndef QALG_H
#define QALG_H


#include "alg.h"


/* Generic algorithm for the quality update and epsilon-greed expolration. The exploration
   and the learning rate parameters have a power-law shape. */
class QAlg_eps : public RLAlgorithm {

    protected:

        /* Quality function */
        vec2d quality;
        /* Which aggregate states are stored in the trajectory (to save space) */
        veci traj_states;
        /* Learning rate scheduling */
        d_i_fnc lr;
        /* Learning rate scheduling according to occupation*/
        d_i_fnc lr_sa;
        /* Exploration rate scheduling */
        d_i_fnc eps;
        /* Exploration rate scheduling according to occupation*/
        d_i_fnc eps_sa;
        /* Uniform random distribution */
        std::uniform_real_distribution<double> unif_dist;
        /* Uniform distribution over the actions */
        std::uniform_int_distribution<int> unif_act_dist;

        std::normal_distribution<double> gauss_dist;

        // TRAJECTROIES TO BUILD
        int t_time;
        /* Quality function trajectory */
        vec3d quality_traj;
        /* Episode at which traj is computed */
        veci ep_traj;
        /* Learning rate traj */
        vecd lr_traj;
        /* Exploration traj */
        vecd eps_traj;
        /* Environment information traj */
        vec2d env_info_traj;

        // OVERRIDED FUNCTIONS
        void init(const param& params);
        int get_action(bool sw);
        void build_traj();
        void build_eval_traj();
        void print_traj(std::string out_dir) const;

        // ABSTRACT FUNCTIONS
        virtual void learning_update() = 0;

        // AUX METOHDS
        vec2d const_quals(double val);
        vec2d noisy_quals(double val);

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


using vecpair = std::vector<std::array<int,2>>;

// Eligibility traces
class ET_eps : public QAlg_eps {

    private:

        double lambda;
        vecpair sa_pairs_to_update;
        vecd traces;
        const double trace_th = 0.001;

        int old_state;
        int old_action;
        double old_reward;

    protected:

        virtual void learning_update();

    public:

        ET_eps(Environment* env, const param& params, std::mt19937& generator);

        virtual const std::string descr() const { return "Eligibility traces algorithm with epsilon greedy exploration."; }
};


#endif
