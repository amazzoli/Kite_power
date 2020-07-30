#ifndef EVAL_H
#define EVAL_H

#include "env.h"



// class Observable {

//     public:
//         std::string name;
//         std::function<double(Environment*)> compute;
// };


// using vec_obs = std::vector<Observable>;


/* Evaluation class */
class Eval {

    private:

        /* Discount factor */
        double m_gamma;
        /* Random number generator */ 
        std::mt19937 m_generator;        
        /* MDP to evaluate */
        Environment* m_env;
        /* Policy to evaluate */
        vec2d m_policy;
        /* Trajectory of all the states */
        vec2d m_state_traj;
        /* Trajectory of the returns */
        vecd m_return_traj;
        // /* Observables to compute */
        // vec_obs m_obs;
        // /* Trajectory of all the observables */
        // vec2d m_obs_traj;

    public:

        /* Construct the algorithm given the parameters dictionary */
        Eval(Environment* env, vec2d policy, dictd& params, std::mt19937& generator): 
        m_env{env}, m_policy{policy}, m_generator{generator}, m_gamma{params["gamma"]}
        /*, m_obs{observables}*/ {};

        /* Run the evaluation */
        void run(int n_steps, int step_traj);

        /* Print the the trajectories of the states in the file "ev_states.txt" contained in the out_dir.
           The file has a header containing the names of the states */
        void print_traj(std::string out_dir) const;
};


#endif