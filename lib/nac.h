#ifndef NAC_H
#define NAC_H

#include "alg.h"


// Here we define the Standard Actor Critic algorithm "AC" that learns the best policy in a 
// given environment.
// The Natural Actor Critic with advantage parameters "NAC_AP" (see Bathnagar et al. 2009, algorithm 3)
// is defined as a derived class of AC where the two virtual methods for the parameter initialization
// and the actor update are overrided.
// All the methods are written in "nac.cpp"


/* Standard Actor Critic algorithm */
class AC : public RLAlgorithm {

    private:
 
        /* Critic learning rate dependent on time */
        d_i_fnc lr_crit;
        /* Actor learning rate dependent on time */
        d_i_fnc lr_act;  
        /* Trajectory of the policy parameters */ 
        vec3d policy_par_traj;
        /* Trajectory of the values */ 
        vec2d value_traj;
        /* Which aggregate states are stored in the trajectory (to save space) */
        veci traj_states;
        /* Trajectory of the learning rates */ 
        vec2d lr_traj;


        // AUX FUNCTIONS
        /* Build constant value parameters */
        vecd const_values(double val);
        /* Build a flat policy */
        vec2d flat_policy();

    protected:  

        // "CURRENT VARIABLES" CHANGED AT EACH LEARNING STEP
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

        // METHODS TO OVERRIDE
        virtual void init(const param& params);
        virtual int get_action(bool eval);
        virtual void learning_update();
        virtual void build_traj();
        virtual void print_traj(std::string out_dir) const;

        // CHILD ALGORITHM METHODS
        virtual void child_init() {};
        virtual void child_update(double td_error);

    public:

        /* Construct the algorithm given the parameters dictionary */
        AC(Environment* env, const param& params, std::mt19937& generator);

        /* Algorithm description */
        const std::string descr() const { return "Actor critic algorithm."; }
};


/* Natural Actor Critic with advantage parameters */
class NAC_AP : public AC {

    private:

        /* Advantage parameters */
        vec2d m_ap_par;

    protected:

        virtual void child_init();
        virtual void child_update(double td_error);

    public:

        NAC_AP(Environment* env, const param& params, std::mt19937& generator) : 
        AC{env, params, generator} {};

        const std::string descr() const { return "Natural actor critic with advantage parameters algorithm."; }
};


#endif