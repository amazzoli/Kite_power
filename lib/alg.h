#ifndef ALG_H
#define ALG_H

#include "env.h"


/* Abstract reinforcement learning algorithm. It loops over the environment 
   and calls the algorithm-specific action for getting the action and updating 
   the learning quantities. */
class RLAlgorithm {

    private:

        // TRAJECTORY FOR EVALUATION   
        int ev_traj_step;
        /* Trajectory of all the states */
        vec2d state_traj;
        /* Aggregate states trajectory */
        veci aggr_st_traj;
        /* Action trajectory */
        veci action_traj;
        /* Reward trajectory */
        vecd rew_traj;
        /* Done trajectory */
        veci done_traj;

        void train(int n_steps, int traj_step, const param& params);
        void evaluate(int eval_steps);

    protected:

        /* MDP to solve */
        Environment* env; 
        /* Random number generator */ 
        std::mt19937 generator;
        /* Discount factor */
        double m_gamma;   
        /* Trajectory of the returns */ 
        vecd return_traj;
        /* Length of the episodes */
        veci ep_len_traj;

        // "CURRENT VARIABLES" CHANGED AT EACH LEARNING STEP
        /* Aggregate state at the current time step of the learning */
        int curr_aggr_state;
        /* Chosen action at the current time step of the learning */
        int curr_action;
        /* Current reward and terminal state info */
        env_info curr_info;
        /* Aggregate state at the current time step of the learning */
        int curr_new_aggr_state;
        /* Current episode */
        int curr_episode;
        /* Temporal step of the episode */
        int curr_ep_step;
        /* Temporal step of the algorithm */
        int curr_step;
        /* Gamma factor */
        int curr_gamma_fact;

        // ALGORITHM SPECIFIC METHODS
        virtual void init(const param& params) = 0;
        virtual int get_action(bool eval) = 0;
        virtual void learning_update() = 0;
        virtual void build_traj() = 0;
        virtual void print_traj(std::string out_dir) const = 0;

    public:

        /* Construct the algorithm given the parameters dictionary */
        RLAlgorithm(Environment* env, const param& params, std::mt19937& generator);

        /* Algorithm description */
        virtual const std::string descr() const = 0;

        /* Run the algorithm */
        void run(const param& params);

        /* Print the policy, the value trajectories and their final result */
        virtual void print_output(std::string out_dir) const;
};


#endif