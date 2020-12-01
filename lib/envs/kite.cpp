#include "kite.h"


Kite::Kite(const param& params, std::mt19937& generator) : Environment{params, generator} {

	// Check temporal inconsistencies
	if (params.d.at("ep_length") < params.d.at("decision_time") || params.d.at("ep_length") < params.d.at("int_steps")
		|| params.d.at("decision_time") < params.d.at("int_steps"))
		throw std::runtime_error ( "Temporal scales of the kite are not consistent\n" );

	try {
		ep_length = int(params.d.at("ep_length")/params.d.at("int_steps"));
		if (params.d.find("ep_length_eval") != params.d.end())
			ep_length_ev = int(params.d.at("ep_length_eval")/params.d.at("int_steps"));
		else
			ep_length_ev = ep_length;
		steps_btw_train = int(params.d.at("decision_time")/params.d.at("int_steps"));
		h = params.d.at("int_steps");
		alphas = params.vecd.at("alphas");
		CL_alpha = params.vecd.at("CL_alphas");
		CD_alpha = params.vecd.at("CD_alphas");
	} catch (std::exception) {
		throw std::invalid_argument("Invalide kite prameter");
	}

	fallen_times = 0;
	errors = 0;
}


int Kite::reset_state(){
    // Initial attack angle
    if (init_alpha_ind >= n_alphas())
        alpha_ind = std::uniform_int_distribution<int>(0, n_alphas()-1)(m_generator);
    else 
        alpha_ind = init_alpha_ind;
	
	curr_ep_step = 0;
	return reset_kite();
}


env_info Kite::step(int action, bool eval) {
	impose_action(action);
	int t;
	bool done = false;
	for (t=0; t<steps_btw_train; t++){
		// Check the episode end or a terminal state reached
		if (integrate_trajectory()) {
			fallen_times++;
			done = true;
			break;
		}
		if (!eval && curr_ep_step >= ep_length){
			done = true;
			break;
		}
		if (eval && curr_ep_step >= ep_length_ev){
			done = true;
			break;
		}
		curr_ep_step++;
	}

	double r = get_rew(t);
	if (r > 1000) {
		errors++;
		return env_info {0, true};
	}
	else
		return env_info {r, done};
}

vecd Kite::env_data() {
	vecd data = vecd { (double)fallen_times, (double)errors };
	fallen_times = 0;
	return data;
}

vecs Kite::env_data_headers() {
	return vecs { "Fallen_times", "Reward_error" };;
}