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
        Environment(dictd params, std::mt19937& generator) : m_generator{generator} {};

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


// /* Abstract. Generic class for all the kites containing the common constants */
// class Kite : public Environment {

// 	protected:

// 		// FIXED PHYSICAL CONSTANTS
// 		/* Kite mass, kg */
// 		const double m_kite = 1.0;
// 		/* Kite area, m^2 */
// 		const double a_kite = 5.0;
// 		/* Cable length, m */
// 		const double R = 50.0;
// 		/* Block mass, kg */
// 		const double m_block = 50.0;
// 		/* Block friction coefficient */
// 		const double coef_friction = 0.4;

// 		// FIXED ATTACK ANGLE DISCRETIZATION
// 		/* Discretization of the attack angle */
// 		const double alphas[15] = {-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
// 		/* Lift coefficient for each attack angle */
// 		const double CL_alpha[15] = {-0.15, -0.05, 0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.9, 1.0, 1.08, 1.1, 1.05};
// 		/* Drag coefficient for each attack angle */
// 		const double CD_alpha[15] = {0.005, 0.005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.13, 0.18, 0.18, 0.21};

// 		/* Penalty of falling */
// 		const double fall_penalty = 300;

// 		// FIXED COMPUTATIONAL CONSTANTS
// 		/* Integration constant */
// 		const double h = 10E-4;
// 		/* Zero threshold for the velocity */
// 		const double v_threshold = 10E-8;

// 		/* Episode length */
// 		int ep_length;
// 		/* Number of consecutive steps between training */
// 		int steps_btw_train;

// 		/* Current step of the episode */
// 		int curr_ep_step;


// 	public:

// 		Kite(dictd params, std::mt19937& generator) : Environment{params, generator} {
// 			ep_length = params["ep_length"];
// 			steps_btw_train = params["steps_btw_train"];
// 		}

// 		virtual int aggr_state() const = 0;

//         virtual void reset_state() = 0;

//         /* Step is called before every learning update. It integrates the trajectory
//            for "steps_btw_train" calling the system-specific abstract method for the
//            one-step integration "integrate_trajectory()". Also the env_info are
//            system-specific and obtained with "get_rew_and_done()" */
//         virtual env_info step(int action) {
//         	for (int t=0; t<steps_btw_train; t++)
//         		integrate_trajectory();
//         	return get_rew_and_done();
//         }

//         virtual void integrate_trajectory() = 0;

//         virtual env_info get_rew_and_done() = 0;

//         virtual double terminal_reward(double gamma) { return 0; };

//         int n_alphas() const { return sizeof(alphas)/sizeof(alphas[0]); }
// };


// class Kite2d : public Kite {
	
// 	private:

// 		/* Initial angle of the block-kite with the ground */
// 		double init_theta;
// 		/* Initial angular velocity of the block-kite */
// 		double init_dtheta;
// 		/* Current value of the attack angle index */
// 		int curr_alpha_ind;

// 		//TODO: generic wind function
// 		//std::function<int[2](int[2])> v_wind;
// 		/* Constant wind speed */
// 		int v_wind[2];

// 		double t2;

// 	public:

// 		// All functions defined in kite2d.cpp
// 		Kite2d(dictd params, std::mt19937& generator);
// 		virtual int aggr_state() const;
//         virtual void reset_state();
//         virtual void integrate_trajectory();
//         virtual env_info get_rew_and_done();
//         virtual double terminal_reward(double gamma);
// };



//#include "envs/kite.h"
//class Kite2d;

Environment* get_env(std::string env_name, dictd params, std::mt19937& generator);



#endif