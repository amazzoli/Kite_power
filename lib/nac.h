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

        // AUX FUNCTIONS
        /* Build constant value parameters */
        vecd const_values(double val);
        /* Build a flat policy */
        vec2d flat_policy();
        /* Initialization of the variables before running */ 
        void init(vecd init_values, vec2d init_policies);
        /* Reset the variables at episode end */
        void reset_env();

    protected:

        /* MDP to solve */
        Environment* env;   

        // "CURRENT VARIABLES" CHANGED AT EACH LEARNING STEP
        /* Aggregate state at the current time step of the learning */
        int curr_aggr_state;
        /* Chosen action at the current time step of the learning */
        int curr_action;
        /* Policy at the current time step of the learning */
        vecd curr_policy;
        /* Critic learning rate at the current time step of the learning */
        double curr_crit_lr;
        /* Actor learning rate at the current time step of the learning */
        double curr_act_lr;
        /* Value/critic parameters */
        vecd curr_v_pars;
        /* Policy/actor parameters */
        vec2d curr_p_pars;
        /* Return of the episode */
        double curr_return;
        /* Gamma power number of steps from episode onset */
        double curr_gamma_fact;

        // VIRTUAL FUNCTIONS
        /* Algorithm specific initialization */
        virtual void init_pars() {};
        /* Actor delta update */
        virtual void  delta_act_updt(double td_error);

    public:

        /* Construct the algorithm given the parameters dictionary */
        AC(Environment* env, const dictd& params, std::mt19937& generator);

        /* Run the algorithm for n_steps, given the initial condition of the values and the policies */
        void run(int n_steps, vecd init_values, vec2d init_policies, int n_point_traj=2000);
        /* Run the algorithm for n_steps, given the initial condition of the values and flat policies */
        void run(int n_steps, vecd init_values, int n_point_traj=2000);
        /* Run the algorithm for n_steps, given the initial condition of the policies and constant values */
        void run(int n_steps, double init_values, vec2d init_policies, int n_point_traj=2000);
        /* Run the algorithm for n_steps, given constant values and flat policies */
        void run(int n_steps, double init_values, int n_point_traj=2000);

        /* Print the policy and the value trajectories */
        void print_traj(std::string out_dir) const;

        /* Print the best policy and the best value found */
        void print_best_pol_val(std::string out_dir) const;
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