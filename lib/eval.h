#ifndef EVAL_H
#define EVAL_H

#include "env.h"
#include "alg.h"



/* Evaluation algorithm */
class Eval : Algorithm {

    private:

        /* Discount factor */
        double m_gamma;
        /* Policy to evaluate */
        vec2d m_policy;
        /* Trajectory of all the states */
        vec2d m_state_traj;
        /* Trajectory of the returns */
        vecd m_return_traj;
        /* Length of the episodes */
        veci m_ep_len_traj;
        /* Aggregate states trajectory */
        veci m_aggr_st_traj;

    public:

        /* Construct the algorithm given the parameters dictionary */
        Eval(Environment* env, vec2d policy, dictd& params, std::mt19937& generator): 
        Algorithm(env, generator), m_policy{policy}, m_gamma{params.at("gamma")} {};

        /* Run the evaluation */
        virtual void run(const param& params);

        /* Print the the trajectories of the states in the file "ev_states.txt" contained in the out_dir.
           The file has a header containing the names of the states */
        virtual void print_output(std::string out_dir) const;
};


#endif