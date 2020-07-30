#include "kite.h"


Kite::Kite(dictd params, std::mt19937& generator) : Environment{params, generator} {

	// If parameters are not present, they are init to zero
	if (params["ep_length"] == 0 || params["decision_time"] == 0 || params["int_steps"] == 0)
		throw std::runtime_error ( "Environment parameters error\n" );

	// Check temporal inconsistencies
	if (params["ep_length"] < params["decision_time"] || params["ep_length"] < params["int_steps"] \
		|| params["decision_time"] < params["int_steps"])
		throw std::runtime_error ( "Temporal scales are not consistent\n" );

	ep_length = int(params["ep_length"]/params["int_steps"]);
	steps_btw_train = int(params["decision_time"]/params["int_steps"]);
	h = params["int_steps"];
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


Environment* get_env(std::string env_name, dictd params, std::mt19937& generator) {
    if (env_name == "kite2d"){
        Environment* env = new Kite2d(params, generator);
        return env;
    }
        
    else{
        throw std::invalid_argument( "Invalid environment name" );
        Environment* env = new Kite2d(params, generator);
        return env;
    }
}