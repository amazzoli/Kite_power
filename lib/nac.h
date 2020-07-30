#ifndef NAC_H
#define NAC_H

#include "env.h"


// Here we define the Standard Actor Critic algorithm "AC" that learns the best policy in a 
// given environment.
// The Natural Actor Critic with advantage parameters "NAC_AP" (see Bathnagar et al. 2009, algorithm 3)
// is defined as a derived class of AC where the two virtual methods for the parameter initialization
// and the actor update are overrided.
// All the methods are written in "nac.cpp"


// Type for the learning rates scheduling
using d_i_fnc = std::function<double(int)>;


/* Standard Actor Critic algorithm */
class AC {

    private:

        /* Discount factor */
        double m_gamma;   
        /* Critic learning rate dependent on time */
        d_i_fnc m_lr_crit;
        /* Actor learning rate dependent on time */
        d_i_fnc m_lr_act;  
        /* Trajectory of the policy parameters */ 
        vec3d m_policy_par_traj;
        /* Trajectory of the values */ 
        vec2d m_value_traj;
        /* Trajectory of the returns */ 
        vecd m_return_traj;
        /* Random number generator */ 
        std::mt19937 m_generator;

        const double p_par_bounds[2] = {-100, 100};

    protected:

        /* MDP to solve */
        Environment* m_env;   
        /* Aggregate state at the current time step of the learning */
        int m_curr_aggr_state;
        /* Chosen action at the current time step of the learning */
        int m_curr_action;
        /* Policy at the current time step of the learning */
        vecd m_curr_policy;
        /* Critic learning rate at the current time step of the learning */
        double m_curr_crit_lr;
        /* Actor learning rate at the current time step of the learning */
        double m_curr_act_lr;
        /* Algorithm specific initialization */
        virtual void init_pars() {};
        /* Actor delta update */
        virtual void  delta_act_updt(vec2d& policy_pars, double td_error);

    public:

        /* Construct the algorithm given the parameters dictionary */
        AC(Environment* env, dictd& params, std::mt19937& generator);

        /* Run the algorithm for n_steps, given the critic and actor learning rates, which
           are functions of the time step. */
        void run(int n_steps, int n_point_traj=2000, double init_values=0);

        /* Print the policy and the value trajectories */
        void print_traj(std::string out_dir) const;

        /* Print the best policy found */
        void print_best_pol(std::string out_dir) const;

        /* Get the trajectory of the policy */ 
        const vec3d& policy_par_traj() const { return m_policy_par_traj; }

        /* Get the trajectory of the values */ 
        const vec2d& value_traj() const { return m_value_traj; }
};


/* Natural Actor Critic with advantage parameters */
class NAC_AP : public AC {

    private:

        /* Advantage parameters */
        vec2d m_ap_par;

    protected:

        virtual void init_pars();
        virtual void delta_act_updt(vec2d& policy_pars, double td_error);

    public:

        NAC_AP(Environment* env, dictd& params, std::mt19937& generator) : 
        AC{env, params, generator} {};
};


#endif