#include "alg.h"


RLAlgorithm::RLAlgorithm(Environment* env, const param& params, std::mt19937& generator) :
env{env}, generator{generator}, m_gamma{params.d.at("gamma")} {}


void RLAlgorithm::run(const param& params) {

    int n_steps = params.d.at("n_steps");
    int traj_points = params.d.at("traj_points");
    int traj_step = round(n_steps/float(traj_points));
    curr_episode = 1;
    curr_ep_step = 1;
    int perc_step = 5;
    Perc perc(perc_step, n_steps-1);
    double av_ret = 0;
    int ep_for_av_ret = 1;

    // Env initialization   
    curr_aggr_state = (*env).reset_state();
    double ret = 0;
    curr_gamma_fact = 1;

    // Algorithm-specific initialization
    init(params);

    // Main loop
    for (curr_step=0; curr_step<n_steps; ++curr_step){

        // Algorithm-specific action at the current step
        curr_action = get_action();

        // Envitonmental step
        curr_info = (*env).step(curr_action);
        ret += curr_info.reward * curr_gamma_fact;
        curr_new_aggr_state = (*env).aggr_state();

        // Algorithm-specific update
        learning_update();

        // Building the trajectory
        if (traj_step > 0 && curr_step%traj_step == 0) build_traj();

        // At terminal state
        if (curr_info.done){ 
            ret += (*env).terminal_reward(m_gamma) * curr_gamma_fact;
            return_traj.push_back(ret);
            av_ret += ret;
            ep_for_av_ret++;
            ret = 0;
            ep_len_traj.push_back(curr_ep_step);
            curr_ep_step = 1;
            curr_aggr_state = (*env).reset_state();
            curr_gamma_fact = 1;
            curr_episode++;
        } 
        // At non-terminal state
        else { 
            curr_aggr_state = curr_new_aggr_state;
            curr_gamma_fact *= m_gamma;
            curr_ep_step++;
        }

        if (perc.step(curr_step)) {
            if (ep_for_av_ret != 0)
                std:: cout << " average return over " << ep_for_av_ret << " ep: " << av_ret / (float)ep_for_av_ret;
            ep_for_av_ret = 0;
            av_ret = 0;
        }
    }
}



void RLAlgorithm::print_output(std::string dir) const {

    // Printing the returns and the episode lengths
    std::ofstream file_r;
    file_r.open(dir + "/return_traj.txt");
    file_r << "Return\tEpisode_length\n";
    for (int t=0; t<return_traj.size(); t++){
        file_r << return_traj[t] << "\t" << ep_len_traj[t] << "\n";
    }

    // Printing the algorithm-specific trajectories
    print_traj(dir);
}


