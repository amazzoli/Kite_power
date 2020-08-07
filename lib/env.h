#ifndef ENV_H
#define ENV_H

#include "utils.h"


/* Outcome of an environment transition*/
struct env_info {
    double reward;
    bool done;
};


/* Abstract. It contains all the info of the single player MDP to be solved
   by the Natural Actor Critic algorithm with state aggregation */
class Environment {

    protected:

        /* Size of the aggregate state space */
        unsigned int m_n_aggr_state;
        /* Number of actions */
        unsigned int m_n_actions;
        /* Current state */
        vecd m_state;
        // Descriptors of each position of m_state
        vecs m_state_descr;
        // Descriptors of each aggregate state
        vecs m_aggr_state_descr;
        // Descriptors of each action
        vecs m_act_descr;
        /* Random number generator */ 
        std::mt19937 m_generator;

    public:

        // CONSTRUCTOR
        Environment(const dictd& params, std::mt19937& generator) : m_generator{generator} {};

        // GET
        /* Get the aggregate-state-space shape */
        unsigned int n_aggr_state() const { return m_n_aggr_state; }
        /* Get the number of actions */
        unsigned int n_actions() const { return m_n_actions; }
        /* Get the current state */
        vecd state() const { return m_state; }
        /* Get the description of each state index */
        const vecs& state_descr() const { return m_state_descr; }
        /* Get the description of each aggregare state */
        const vecs& aggr_state_descr() const { return m_aggr_state_descr; }
        /* Get the description of each action */
        const vecs& action_descr() const { return m_act_descr; }

        // ABSTRACT/VIRTUAL
        /* Get the current aggregate state */
        virtual int aggr_state() const = 0;
        /* Set the environment in the initial state */
        virtual void reset_state() = 0;
        /* Environmental transition given the action which modifies the 
           internal state and return the reward and the termination flag. */
        virtual env_info step(int action) = 0;
        /* Reward ot penalty in the terminal state, zero by default */
        virtual double terminal_reward(double gamma) { return 0; };
};


//Environment* get_env(std::string env_name, dictd params, std::mt19937& generator);



#endif