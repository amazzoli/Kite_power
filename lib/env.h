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

        /* Random number generator */ 
        std::mt19937 m_generator;

    public:

        // CONSTRUCTOR
        Environment(const param& par, std::mt19937& generator) : m_generator{generator} {};
        virtual ~Environment() {};

        /* Abstract. Get the description of the environment */
        virtual const std::string descr() const = 0;
        /* Get the aggregate-state-space shape */
        virtual unsigned int n_aggr_state() const = 0;
        /* Get the number of actions */
        virtual unsigned int n_actions() = 0;
        /* Get the current state */
        virtual const vecd& state() = 0;
        /* Get the description of each state index */
        virtual const vecs state_descr() const = 0;
        /* Get the description of each aggregare state */
        virtual const vecs aggr_state_descr() const = 0;
        /* Get the description of each action */
        virtual const vecs action_descr() const = 0;
        /* Get the current aggregate state */
        virtual int aggr_state() const = 0;
        /* Set the environment in the initial state and returns the state */
        virtual int reset_state() = 0;
        /* Environmental transition given the action which modifies the 
           internal state and return the reward and the termination flag. */
        virtual env_info step(int action, bool eval) = 0;
        /* Reward ot penalty in the terminal state, zero by default */
        virtual double terminal_reward(double gamma) { return 0; };
        /* Information about the environment */
        virtual vecd env_data() { return vecd(0); }
        virtual vecs env_data_headers() { return vecs(0); }
};


//Environment* get_env(std::string env_name, dictd params, std::mt19937& generator);



#endif