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
		const double m_block = 40.0;
		/* Block friction coefficient */
		const double coef_friction = 0.4;
		/* Penalty of falling */
		const double fall_penalty = 1000;
		/* Zero threshold for the velocity */
		const double v_threshold = 10E-8;
		
		// SET BY CONSTRUCTOR
		/* Initial angle of the block-kite with the ground */
		double init_theta;
		/* Initial angular velocity of the block-kite */
		double init_dtheta;
		/* Initial attack angle index. If > n_alphas they are generated at random */
		int init_alpha_ind;
		/* Integration time step */
		double h;
		/* Discretization of the attack angle */
		vecd alphas;
		/* Lift coefficient for each attack angle */
		vecd CL_alpha;
		/* Drag coefficient for each attack angle */
		vecd CD_alpha;

		// DYNAMICAL VARIABLES
		/* Current value of the attack angle index */
		int alpha_ind;

	public:

		/* Constructor */
		Kite(const param& params, std::mt19937& generator);

		/* Kite description */
		virtual const std::string descr() const = 0;

		/* Abstract. Get the current aggregate state */
		virtual int aggr_state() const = 0;

		/* Set the environment in the initial state */
        virtual int reset_state();

		/* Abstract. Set the specific kite in the initial state */
        virtual int reset_kite() = 0;

        /* Step is called before every learning update. It first imposes the action though the env-specific abstract 
		   method "impose_action(int action)". Then it integrates the trajectory for "steps_btw_train" times calling 
		   "integrate_trajectory()". At the end it retruns the env_info with "get_rew_and_done()" */
        virtual env_info step(int action);

        /* Abstract. Change the internal state as a consequence of the action */ 
        virtual void impose_action(int action) = 0;

        /* Abstract. One temporal step of the internal dynamics */
        virtual bool integrate_trajectory() = 0;

        /* Abstract. Get the reward given the internal state */
        virtual double get_rew(int steps_from_training) = 0;

		/* Number of attack angles */ 
        int n_alphas() const { return alphas.size(); }
};


/* 2d kite which observes and controls the attack angles */
class Kite2d : public Kite {
	
	protected:

		/* Generic wind type */
		Wind2d* wind;

		// DYNAMICAL VARIABLES
		int t2;
		double r_diff[2];
		double theta;
		double beta;
		double f_aer[2];
		double tension[2];
		double friction;

		// AUXILIARY FUNCTION FOR THE DYNAMICS
		void compute_F_aer();
		void compute_tension_still();
		void compute_tension_move();
		void update_state();

	public:

		Kite2d(const param& params, Wind2d* wind, std::mt19937& generator);
		~Kite2d() { delete wind; }
		virtual const std::string descr() const;
		virtual int aggr_state() const;
        virtual int reset_kite();
        virtual void impose_action(int action);
        virtual bool integrate_trajectory();
        virtual double get_rew(int steps_from_training);
        virtual double terminal_reward(double gamma);
};


/* 3d kite which observes and controls the attack and the bank angles */
class Kite3d : public Kite {
	
	protected:

		/* Generic wind type */
		Wind3d* wind;

		// SET BY CONSTRUCTOR
		/* Initial angle from x axis*/
		double init_phi;
		/* Initial agular velocity of phi */
		double init_dphi;
		/* Initial index of the bank angle */
		int init_bank_ind;
		/* Bank angles discretizations */
		vecd bank; 

		// DYNAMICAL VARIABLES
		int bank_ind;
		double x_diff;
		double theta;
		double beta;
		double phi;
		double f_aer[3];
		double tension[3];
		double friction;

		// AUXILIARY FUNCTION FOR THE DYNAMICS
		void compute_F_aer();
		void compute_tension_still();
		void compute_tension_move();
		void update_state();

	public:

		Kite3d(const param& params, Wind3d* wind, std::mt19937& generator);
		~Kite3d() { delete wind; }
		virtual const std::string descr() const;
		virtual int aggr_state() const;
        virtual int reset_kite();
        virtual void impose_action(int action);
        virtual bool integrate_trajectory();
        virtual double get_rew(int steps_from_training);
        virtual double terminal_reward(double gamma);
		int n_banks() const { return bank.size(); }
};


/* 2d kite which controls the attack angles and observes the attack and the relative-velocity angles */
class Kite2d_vrel : public Kite2d {
	private:

		/* Discretized relative velocity angles. Note, those are the boundaries of the bins.
		   The actual states refer to the middle point between each bin, and are one less that the array size */
		vecd beta_bins;

	public:

		Kite2d_vrel(const param& params, Wind2d* wind, std::mt19937& generator);
		virtual const std::string descr() const;
		virtual int aggr_state() const;
        int n_betas() const { return beta_bins.size()-1; }
};


/* 3d kite which controls the attack and the bank angles and observes the attack, the bank and the relative-velocity angles */
class Kite3d_vrel : public Kite3d {
	private:

		/* Discretized relative velocity angles. Note, those are the boundaries of the bins.
		   The actual states refer to the middle point between each bin, and are one less that the array size */
		vecd beta_bins;

	public:

		Kite3d_vrel(const param& params, Wind3d* wind, std::mt19937& generator);
		virtual const std::string descr() const;
		virtual int aggr_state() const;
        int n_betas() const { return beta_bins.size()-1; }
};


#endif