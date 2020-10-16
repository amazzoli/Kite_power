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

    if (params.s.find("init_q_path") == params.s.end())
        quality = const_quals(params.d.at("init_quals"));
    else
        quality = read_quality(params.s.at("init_q_path"));

    quality_traj = vec3d(0);
}


int QAlg_eps::get_action() {

    double u = unif_dist(generator);
    int action;
    if (u < eps(curr_step)){
        action = unif_act_dist(generator);
    }
    else {
        auto max_elem = std::max_element(quality[curr_aggr_state].begin(), quality[curr_aggr_state].end());
        action = std::distance(quality[curr_aggr_state].begin(), max_elem);
    }

    return action;
}


void QAlg_eps::build_traj() {
    vec2d q = vec2d(0);
    for (int k : traj_states)
        q.push_back(quality[k]);
    quality_traj.push_back(q);
}


void QAlg_eps::print_traj(std::string out_dir) const {

    // PRINTING THE TRAJECTOIES
    std::ofstream out_q, out_p;
    out_q.open(out_dir + "quality_traj.txt");
    // Headers
    for (int k : traj_states)
        out_q << (*env).aggr_state_descr()[k] << "\t";
    out_q << "\n";
    for (int a=0; a<quality_traj[0][0].size(); a++){
        out_q << (*env).action_descr()[a];
        if (a < quality_traj[0][0].size()-1) out_q << ",";
    }
    out_q << "\n";
    // Body
    for (int t=0; t<quality_traj.size(); t++){
        for (int k=0; k<quality_traj[t].size(); k++){
            for (int a=0; a<quality_traj[t][k].size(); a++){
                out_q << quality_traj[t][k][a];
                if (a < quality_traj[t][k].size()-1) out_q << ",";
            }
            out_q << "\t";
        }
        out_q << "\n";
    }
    out_q.close();

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