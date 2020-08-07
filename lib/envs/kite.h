#ifndef KITE_H
#define KITE_H

#include "../env.h"
#include "../wind.h"



// Header file containing all the environments that can be trained


/* Abstract. Generic class for all the kites containing the common constants */
class Kite : public Environment {

	private:

		// INTERNAL VARIABLE
		/* Current step of the episode */
		int curr_ep_step;

		// SET BY CONSTRUCTOR
		/* Episode length */
		int ep_length;
		/* Number of consecutive steps between training */
		int steps_btw_train;

	protected:

		// FIXED CONSTANTS
		/* Kite mass, kg */
		const double m_kite = 1.0;
		/* Kite area, m^2 */
		const double a_kite = 5.0;
		/* Cable length, m */
		const double R = 50.0;
		/* Block mass, kg */
		const double m_block = 50.0;
		/* Block friction coefficient */
		const double coef_friction = 0.4;
		/* Discretization of the attack angle */
		const double alphas[15] = {-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
		/* Lift coefficient for each attack angle */
		const double CL_alpha[15] = {-0.15, -0.05, 0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.75, 0.82, 0.9, 1.0, 1.08, 1.1, 1.05};
		/* Drag coefficient for each attack angle */
		const double CD_alpha[15] = {0.005, 0.005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1, 0.13, 0.18, 0.18, 0.21};
		/* Penalty of falling */
		const double fall_penalty = 300;
		/* Zero threshold for the velocity */
		const double v_threshold = 10E-8;
		
		// SET BY CONSTRUCTOR
		/* Integration time step */
		double h;

	public:

		Kite(const dictd& params, std::mt19937& generator);

		/* Abstract. Get the current aggregate state */
		virtual int aggr_state() const = 0;

		/* Set the environment in the initial state */
        virtual void reset_state();

		/* Abstract. Set the environment in the initial state */
        virtual void reset_kite() = 0;

        /* Step is called before every learning update. It first imposes the action though the env-specific abstract 
		   method for applying the action "impose_action(int action)". Then it integrates the trajectory for 
		   "steps_btw_train" calling "integrate_trajectory()". At the end it retruns the env_info at the end with 
		   "get_rew_and_done()" */
        virtual env_info step(int action);

        /* Abstract. Change the internal state as a consequence of the action */ 
        virtual void impose_action(int action) = 0;

        /* Abstract. One temporal step of the internal dynamics */
        virtual bool integrate_trajectory() = 0;

        /* Abstract. Get the reward given the internal state */
        virtual double get_rew(int steps_from_training) = 0;

		/* Number of attack angles */ 
        int n_alphas() const { return sizeof(alphas)/sizeof(alphas[0]); }
};


class Kite2d : public Kite {
	
	protected:

		/* Generic wind type */
		Wind2d* wind;

		// INITIAL CONDITIONS
		/* Initial angle of the block-kite with the ground */
		double init_theta;
		/* Initial angular velocity of the block-kite */
		double init_dtheta;
		/* Initial attack angle index. If > n_alphas they are generated at random */
		int init_alpha_ind;

		// DYNAMICAL VARIABLES
		/* Current value of the attack angle index */
		int alpha_ind;
		/* Vetor for computing the lift */
		int t2;
		double r_diff[2];
		double theta;
		double beta;
		double f_aer[2];
		double tension[2];
		double friction;

		// AUXILIARY FUNCTION FOR THE DYNAMICS
		int update_aggr_state(int action);
		void compute_F_aer();
		void compute_tension_still();
		void compute_tension_move();
		void update_state();

	public:

		Kite2d(const dictd& params, Wind2d* wind, std::mt19937& generator);
		~Kite2d() { delete wind; }
		virtual int aggr_state() const;
        virtual void reset_kite();
        virtual void impose_action(int action);
        virtual bool integrate_trajectory();
        virtual double get_rew(int steps_from_training);
        virtual double terminal_reward(double gamma);
};


/* Two-dim kite having in addition the kite-wind relative velocity angle as state */
class Kite2d_vrel : public Kite2d {
	private:
		/* Discretized relative velocity angles. Note, those are the boundaries of the bins.
		   The actual states refer to the middle point between each bin, and are one less that the array size */
		double beta_bins[25] = {-PI, -3.0, -2.9, -2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3, \
                                0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 2.9, 3.0, PI};

	public:

		Kite2d_vrel(const dictd& params, Wind2d* wind, std::mt19937& generator);
		virtual int aggr_state() const;
		/* Number of relative velocity angles */ 
        int n_betas() const { return sizeof(beta_bins)/sizeof(beta_bins[0])-1; }
};



Environment* get_env(std::string env_name, const dictd& params, std::mt19937& generator);


#endif