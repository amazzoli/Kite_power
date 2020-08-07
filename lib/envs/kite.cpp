#include "kite.h"


Kite::Kite(const dictd& params, std::mt19937& generator) : Environment{params, generator} {

	// Check temporal inconsistencies
	if (params.at("ep_length") < params.at("decision_time") || params.at("ep_length") < params.at("int_steps")
		|| params.at("decision_time") < params.at("int_steps"))
		throw std::runtime_error ( "Temporal scales are not consistent\n" );

	ep_length = int(params.at("ep_length")/params.at("int_steps"));
	steps_btw_train = int(params.at("decision_time")/params.at("int_steps"));
	h = params.at("int_steps");
}


void Kite::reset_state(){
	curr_ep_step = 0;
	reset_kite();
}


env_info Kite::step(int action) {
	impose_action(action);
	int t;
	bool done = false;
	for (t=0; t<steps_btw_train; t++){
		// Check the episode end or a terminal state reached
		if (curr_ep_step >= ep_length || integrate_trajectory()){
			done = true;
			break;
		}
		curr_ep_step++;
	}
	return env_info {get_rew(t), done};
}


Environment* get_env(std::string env_name, const dictd& params, std::mt19937& generator) {
    if (env_name == "kite2d"){
		Wind2d* wind = get_wind2d(params);
        Environment* env = new Kite2d(params, wind, generator);
        return env;
    }
    if (env_name == "kite2d_vrel"){
		Wind2d* wind = get_wind2d(params);
        Environment* env = new Kite2d_vrel(params, wind, generator);
        return env;
    }
    else throw std::invalid_argument( "Invalid environment name" );
}