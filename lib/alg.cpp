#include "alg.h"


RLAlgorithm::RLAlgorithm(Environment* env, const param& params, std::mt19937& generator) :
env{env}, generator{generator}, m_gamma{params.d.at("gamma")} {}


void RLAlgorithm::run(const param& params) {

    int n_steps, traj_step, eval_steps;
    try {
        n_steps = params.d.at("n_steps");
        int traj_points = params.d.at("traj_points");
        traj_step = round(n_steps/float(traj_points));
        if (params.d.find("eval_steps") != params.d.end())
            eval_steps = params.d.at("eval_steps");
    } catch (std::exception) {
        throw std::invalid_argument("Invalid temporal parameters of the algorithm.");
    }

    std::cout << "Training\n";
    // Training loop
    train(n_steps, traj_step, params);

    if (eval_steps > 0) {
        std::cout << "\nEvaluating...";
        // Evaluation loop
        evaluate(eval_steps);
    }
}


void RLAlgorithm::train(int n_steps, int traj_step, const param& params) {

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

    for (curr_step=0; curr_step<n_steps; ++curr_step){

        // Algorithm-specific action at the current step
        curr_action = get_action(false);

        // Envitonmental step
        curr_info = (*env).step(curr_action, false);
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

    build_traj();
}


void RLAlgorithm::evaluate(int eval_steps) {

    state_traj = vec2d(eval_steps);
    aggr_st_traj = veci(eval_steps);
    action_traj = veci(eval_steps);
    rew_traj = vecd(eval_steps);
    done_traj = veci(eval_steps);

    curr_aggr_state = (*env).reset_state();

    for (int e_step=0; e_step<eval_steps; ++e_step){

        // Algorithm-specific action at the current step
        curr_action = get_action(true);

        // Envitonmental step
        curr_info = (*env).step(curr_action, true);
        curr_new_aggr_state = (*env).aggr_state();

        state_traj[e_step] = (*env).state();
        aggr_st_traj[e_step] = curr_aggr_state;
        action_traj[e_step] = curr_action;
        rew_traj[e_step] = curr_info.reward;
        done_traj[e_step] = curr_info.done;

        // At terminal state
        if (curr_info.done){ 
            curr_aggr_state = (*env).reset_state();
        } 
        // At non-terminal state
        else { 
            curr_aggr_state = curr_new_aggr_state;
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
    file_r.close();

    // Printing the algorithm-specific trajectories
    print_traj(dir);

    // Printing the evaluation trajectory of the states
    std::ofstream file_s;
    file_s.open(dir + "/ev_states.txt");

    for (int k=0; k<state_traj[0].size(); k++)
        file_s << (*env).state_descr()[k] << "\t";
    file_s << "\n";
    for (int t=0; t<state_traj.size(); t++){
        for (int k=0; k<state_traj[0].size(); k++)
            file_s << state_traj[t][k] << "\t";
        file_s << "\n";
    }
    file_s.close();

    // Printing the evaluation trajectory of the info
    std::ofstream file_info;
    file_info.open(dir + "/ev_info.txt");
    file_info << "state_index\tstate_descr\tacion_index\taction_decr\treward\n";
    for (int t=0; t<aggr_st_traj.size(); t++){
        file_info << aggr_st_traj[t] << "\t";
        file_info << (*env).aggr_state_descr()[aggr_st_traj[t]] << "\t";
        file_info << action_traj[t] << "\t";
        file_info << (*env).action_descr()[action_traj[t]] << "\t";
        file_info << rew_traj[t] << "\n";
    }
    file_info.close();
}


