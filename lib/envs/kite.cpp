#include "kite.h"


Kite::Kite(const param& params, std::mt19937& generator) : Environment{params, generator} {

	// Check temporal inconsistencies
	if (params.d.at("ep_length") < params.d.at("decision_time") || params.d.at("ep_length") < params.d.at("int_steps")
		|| params.d.at("decision_time") < params.d.at("int_steps"))
		throw std::runtime_error ( "Temporal scales are not consistent\n" );

	ep_length = int(params.d.at("ep_length")/params.d.at("int_steps"));
	steps_btw_train = int(params.d.at("decision_time")/params.d.at("int_steps"));
	h = params.d.at("int_steps");
	alphas = params.vecd.at("alphas");
	CL_alpha = params.vecd.at("CL_alphas");
	CD_alpha = params.vecd.at("CD_alphas");
	fallen_times = 0;
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


env_info Kite::step(int action) {
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
		if (curr_ep_step >= ep_length){
			done = true;
			break;
		}
		curr_ep_step++;
	}
	return env_info {get_rew(t), done};
}

vecd Kite::env_data() {
	vecd data = vecd { (double)fallen_times };
	fallen_times = 0;
	return data;
}

vecs Kite::env_data_headers() {
	return vecs { "Fallen_times" };;
}