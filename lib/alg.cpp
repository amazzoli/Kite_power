#include "alg.h"


RLAlgorithm::RLAlgorithm(Environment* env, const param& params, std::mt19937& generator) :
env{env}, generator{generator}, m_gamma{params.d.at("gamma")} {}


void RLAlgorithm::run(const param& params) {

    //ev_step = 0;
    int n_steps, traj_step;
    try {
        n_steps = params.d.at("n_steps");
        int traj_points = params.d.at("traj_points");
        traj_step = round(n_steps/float(traj_points));
        std::cout << traj_step << '\n';
        /*if (params.d.find("eval_steps") != params.d.end())
            ev_step = params.d.at("eval_steps");*/
        if (params.d.find("switch_time") != params.d.end()){
            switch_quality = true;
            steps_before_sw = params.d.at("steps_before_switch");
            //std::cout << "dioboia" << '\n';
        } else {
            switch_quality = false;
            steps_before_sw = n_steps;
        }
    } catch (std::exception) {
        throw std::invalid_argument("Invalid temporal parameters of the algorithm.");
    }

    std::cout << "Training\n";
    // Training loop
    train(n_steps, traj_step, params);

    /*if (ev_step > 0) {
        std::cout << "\nEvaluating...";
        // Evaluation loop
        evaluate();
    }*/
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
    m_sa = vec2i((*env).n_aggr_state(), veci((*env).n_actions(),0));


    // Algorithm-specific initialization
    init(params);

    for (curr_step=0; curr_step<n_steps; ++curr_step){

        // Algorithm-specific action at the current step
        if ((switch_quality) && (curr_ep_step >= steps_before_sw)) {
            curr_action = get_action(true);
        } else {
            curr_action = get_action(false);
        }

        curr_occ_n = m_sa[curr_aggr_state][curr_action];
        m_sa[curr_aggr_state][curr_action]++;

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
            dist_traj.push_back((*env).terminal_distance());
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
            //if (av_ret / (float)ep_for_av_ret > 4000)
                //break;
            ep_for_av_ret = 0;
            av_ret = 0;
        }
    }

    build_traj();
}


/*void RLAlgorithm::evaluate() {

    state_traj = vec2d(ev_step);
    aggr_st_traj = veci(ev_step);
    action_traj = veci(ev_step);
    rew_traj = vecd(ev_step);
    done_traj = veci(ev_step);

    curr_aggr_state = (*env).reset_state();

    for (int e_step=0; e_step<ev_step; ++e_step){

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
}*/


void RLAlgorithm::print_output(std::string dir) const {

    // Printing the returns and the episode lengths
    std::ofstream file_r;
    file_r.open(dir + "/return_traj.txt");
    file_r << "Return\tEpisode_length\tReached distance\n";
    for (int t=0; t<return_traj.size(); t++){
        file_r << return_traj[t] << "\t" << ep_len_traj[t] << "\t" << dist_traj[t] << "\n";
    }
    file_r.close();

    std::ofstream file_m;
    file_m.open(dir + "/occ_matrix2.txt");
    for (size_t i = 0; i < (*env).n_aggr_state(); i++) {
      for (size_t j = 0; j < (*env).n_actions()-1; j++) {
        file_m << m_sa[i][j] << ",";
      }
      file_m << m_sa[i][(*env).n_actions()-1] << "\n";
    }
    file_m.close();

    // Printing the algorithm-specific trajectories
    print_traj(dir);

    // Printing the evaluation trajectory of the states
    /*if (ev_step > 0) {
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
    }*/
}
