#include "qalg.h"


// QALG METHODS

QAlg_eps::QAlg_eps(Environment* env, const param& params, std::mt19937& generator) :
RLAlgorithm(env, params, generator) {

    // Learning rate scheduling
    lr = d_i_fnc{  
        [&params](int step) { 
            return plaw_dacay(step, params.d.at("lr_burn"), params.d.at("lr_expn"), params.d.at("lr0"), params.d.at("lrc"));
        }
    };

    // Exploration scheduling
    eps = d_i_fnc{  
        [&params](int step) { 
            return plaw_dacay(step, params.d.at("eps_burn"), params.d.at("eps_expn"), params.d.at("eps0"), params.d.at("epsc"));
        }
    };

    if (params.vecd.find("traj_states") != params.vecd.end()) {
        vecd traj_states_d = params.vecd.at("traj_states");
        traj_states = veci(0);
        for (double k : traj_states_d)
            traj_states.push_back(int(k));
    }
    else {
        traj_states = veci((*env).n_aggr_state());
        for (int k=0; k<(*env).n_aggr_state(); k++)
            traj_states[k] = k;
    }

    unif_dist = std::uniform_real_distribution<double>(0.0,1.0);
    unif_act_dist = std::uniform_int_distribution<int>(0, (*env).n_actions()-1);
}


void QAlg_eps::init(const param& params) {

    if (params.s.find("init_q_path") == params.s.end()) {
        std::cout << "Starting from flat initial conditions equal to " << params.d.at("init_quals") << "\n";
        quality = const_quals(params.d.at("init_quals"));
    }
    else {
        std::cout << "Starting from initial conditions at " << params.s.at("init_q_path") << "\n";
        quality = read_quality(params.s.at("init_q_path"));
    }

    // Trajectory init
    t_time = 0;
    quality_traj = vec3d(params.d.at("traj_points")+1, vec2d(0));
    ep_traj = veci(params.d.at("traj_points")+1, 0);
    lr_traj = vecd(params.d.at("traj_points")+1, 0);
    eps_traj = vecd(params.d.at("traj_points")+1, 0);
    env_info_traj = vec2d(params.d.at("traj_points")+1, vecd(0));
}


int QAlg_eps::get_action(bool eval) {

    int action;
    if (eval || unif_dist(generator) > eps(curr_step)){
        auto max_elem = std::max_element(quality[curr_aggr_state].begin(), quality[curr_aggr_state].end());
        action = std::distance(quality[curr_aggr_state].begin(), max_elem);        
    }
    else {
        action = unif_act_dist(generator);
    }

    return action;
}


void QAlg_eps::build_traj() {
    // Quality trajectory
    vec2d q = vec2d(0);
    for (int k : traj_states)
        q.push_back(quality[k]);
    quality_traj[t_time] = q;

    // Info trajectory
    ep_traj[t_time] = curr_episode;
    lr_traj[t_time] = lr(curr_step);
    eps_traj[t_time] = eps(curr_step);
    env_info_traj[t_time] = (*env).env_data();

    t_time++;
}


void QAlg_eps::print_traj(std::string out_dir) const {

    // PRINTING THE TRAJECTOIES
    std::ofstream out_q, out_p, out_i;
    out_q.open(out_dir + "quality_traj.txt");
    out_i.open(out_dir + "info_traj.txt");
    
    // Headers Q
    for (int k : traj_states)
        out_q << (*env).aggr_state_descr()[k] << "\t";
    out_q << "\n";
    for (int a=0; a<quality_traj[0][0].size(); a++){
        out_q << (*env).action_descr()[a];
        if (a < quality_traj[0][0].size()-1) out_q << ",";
    }
    out_q << "\n";
    // Headers info
    out_i << "Episode\tLearn_rate\tEpsilon\t";
    for (std::string h : (*env).env_data_headers())
        out_i << h << "\t";
    out_i << "\n";

    // Body
    for (int t=0; t<quality_traj.size(); t++){
        out_i << ep_traj[t] << "\t" << lr_traj[t] << "\t" << eps_traj[t] << "\t";
        for (int k=0; k<(*env).env_data_headers().size(); k++) 
            out_i << env_info_traj[t][k] << "\t";
        for (int k=0; k<quality_traj[t].size(); k++){
            for (int a=0; a<quality_traj[t][k].size(); a++){
                out_q << quality_traj[t][k][a];
                if (a < quality_traj[t][k].size()-1) out_q << ",";
            }
            out_q << "\t";
        }
        out_q << "\n";
        out_i << "\n";
    }
    out_q.close();
    out_i.close();
    
    // PRINTING THE BEST VALUES AND THE BEST POLICIES
    out_p.open(out_dir + "best_policy.txt");
    out_q.open(out_dir + "best_quality.txt");
    for (int k=0; k<quality.size(); k++){
        vecd policy = vecd(quality[k].size());
        auto max_a_iter = std::max_element(quality[k].begin(), quality[k].end());
        int max_a = std::distance(quality[k].begin(), max_a_iter);
        policy[max_a] = 1.0;
        for (int a=0; a<policy.size()-1; a++){
            out_p << policy[a];
            out_q << quality[k][a];
            if (a < policy.size()-2)
                out_p << " ";
            out_q << ",";
        }
        out_q << quality[k][policy.size()-1] << "\n";
        out_p << "\n";
    }
    out_q.close();
    out_p.close();
}


vec2d QAlg_eps::const_quals(double val){
    vecd qual_at_s = vecd((*env).n_actions(), val);
    return vec2d((*env).n_aggr_state(), qual_at_s);
}


// SARSA

void SARSA_eps::learning_update() {

    if (curr_ep_step > 1){
        // Updating the previous state-action pair
        double td_error = old_reward + m_gamma * quality[curr_aggr_state][curr_action] - quality[old_state][old_action];
        quality[old_state][old_action] += lr(curr_step) * td_error;
    }

    if (curr_info.done){
        // Update of the current state-action pair generating a new action
        double new_q;
        if (unif_dist(generator) < eps(curr_step)) // q from exploration
            double new_q = quality[curr_new_aggr_state][unif_act_dist(generator)];
        else // q from exploitation
            double new_q = *std::max(quality[curr_new_aggr_state].begin(), quality[curr_new_aggr_state].end());
        double td_error = curr_info.reward + m_gamma * new_q - quality[curr_aggr_state][curr_action];
        quality[curr_aggr_state][curr_action] += lr(curr_step) * td_error;

        // Terminal update for all the qualities at the terminal state
        for (int a=0; a<quality[curr_new_aggr_state].size(); a++)
            quality[curr_new_aggr_state][a] += lr(curr_step) * ((*env).terminal_reward(m_gamma) - quality[curr_new_aggr_state][a]);
    }

    old_state = curr_aggr_state;
    old_action = curr_action;
    old_reward = curr_info.reward;
}


// Q-LEARNING

void QL_eps::learning_update() {
    double max_new_q = *std::max(quality[curr_new_aggr_state].begin(), quality[curr_new_aggr_state].end());
    double td_error = curr_info.reward + m_gamma * max_new_q - quality[curr_aggr_state][curr_action];
    quality[curr_aggr_state][curr_action] += lr(curr_step) * td_error;

    if (curr_info.done){
        for (int a=0; a<quality[curr_new_aggr_state].size(); a++)
            quality[curr_new_aggr_state][a] += lr(curr_step) * ((*env).terminal_reward(m_gamma) - quality[curr_new_aggr_state][a]);

    }
}


// ELIGIBILITY TRACES

ET_eps::ET_eps(Environment* env, const param& params, std::mt19937& generator) : 
QAlg_eps(env, params, generator) {

    try {
        lambda = params.d.at("lambda");
    }
    catch (std::exception)
        { throw std::runtime_error("Eligibility traces parameters error"); }

    sa_pairs_to_update = vecpair(0);
    traces = vecd(0);
}


void ET_eps::learning_update() {

    // Init at the beginning of the episode
    if (curr_ep_step == 1) {
        sa_pairs_to_update = vecpair(0);
        traces = vecd(0);
    }
    else if (curr_ep_step > 1){

        // Computing TD error
        double td_error = old_reward + m_gamma * quality[curr_aggr_state][curr_action] - quality[old_state][old_action];

        // Check if the new pair is in memory
        int trace_ind = -1;
        for (int i=0; i<sa_pairs_to_update.size(); i++){
            if (sa_pairs_to_update[i][0] == old_state && sa_pairs_to_update[i][1] == old_action){
                trace_ind = i;
                break;
            }
        }

        // Update the traces in memory
        if (trace_ind < 0){
            std::array<int,2> pair = {old_state, old_action};
            sa_pairs_to_update.push_back(pair);
            traces.push_back(1);
        } else { traces[trace_ind] += 1; }

        // Update the qualities
        double alpha = lr(curr_step);
        int vec_size = sa_pairs_to_update.size();
        for (int i=0; i<vec_size; i++) {
            int s = sa_pairs_to_update[i][0];
            int a = sa_pairs_to_update[i][1];
            quality[s][a] += alpha * td_error * traces[i];
            traces[i] *= lambda * m_gamma;

            // Remove traces under the threshold
            if ( traces[i] < trace_th ) {
                traces.erase(traces.begin()+i);
                sa_pairs_to_update.erase(sa_pairs_to_update.begin()+i);
                i--;
                vec_size--;
            }
        }
    }

    if (curr_info.done){
        // Update of the current state-action pair generating a new action
        double new_q;
        if (unif_dist(generator) < eps(curr_step)) // q from exploration
            double new_q = quality[curr_new_aggr_state][unif_act_dist(generator)];
        else // q from exploitation
            double new_q = *std::max(quality[curr_new_aggr_state].begin(), quality[curr_new_aggr_state].end());
        double td_error = curr_info.reward + m_gamma * new_q - quality[curr_aggr_state][curr_action];
        quality[curr_aggr_state][curr_action] += lr(curr_step) * td_error;

        // Terminal update for all the qualities at the terminal state
        for (int a=0; a<quality[curr_new_aggr_state].size(); a++)
            quality[curr_new_aggr_state][a] += lr(curr_step) * ((*env).terminal_reward(m_gamma) - quality[curr_new_aggr_state][a]);
    }

    old_state = curr_aggr_state;
    old_action = curr_action;
    old_reward = curr_info.reward;
}

