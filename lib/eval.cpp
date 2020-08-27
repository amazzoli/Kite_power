#include "eval.h"


void Eval::run(const param& params){

    int n_steps = int(params.d.at("ev_time")/params.d.at("decision_time"));
    int n_point_traj = params.d.at("traj_points");
    int traj_step = std::max(1, (int)round(n_steps/float(n_point_traj)));

    m_state_traj = vec2d(0);
    m_return_traj = vecd(0);
    m_ep_len_traj = veci(0);
    m_aggr_st_traj = veci(0);
    (*env).reset_state();

    int aggr_state = (*env).aggr_state();
    double ret = 0;
    double gamma_factor = 1;
    Perc perc(10, n_steps-1);
    int ep_step = 0;

    // Main loop
    for (int t=0; t<n_steps; ++t){
        perc.step(t);
        ep_step++;

        // Extracting the action from the policy
        std::discrete_distribution<int> dist (m_policy[aggr_state].begin(), m_policy[aggr_state].end());
        int action = dist(generator);

        // Envitonmental step
        env_info info = (*env).step(action);
        int new_aggr_state = (*env).aggr_state();
        ret += info.reward * gamma_factor;

        // Building the state trajectory
        if (t%traj_step == 0){
            m_state_traj.push_back((*env).state());
            m_aggr_st_traj.push_back(aggr_state);
        }

        if (info.done){ // Terminal state
            aggr_state = (*env).aggr_state();
            gamma_factor = 1;
            ret += (*env).terminal_reward(m_gamma) * gamma_factor;
            m_return_traj.push_back(ret);
            ret = 0;
            m_ep_len_traj.push_back(ep_step);
            ep_step = 0;
            (*env).reset_state();
        } else {// Non-terminal state
            aggr_state = new_aggr_state;
            gamma_factor *= m_gamma;
        }
    }
    m_return_traj.push_back(ret);
    m_ep_len_traj.push_back(ep_step);
}


void Eval::print_output(std::string dir) const {
    std::ofstream file_s;
    file_s.open(dir + "/ev_states.txt");
    for (int k=0; k<m_state_traj[0].size(); k++)
        file_s << (*env).state_descr()[k] << "\t";
    file_s << "\n";
    for (int t=0; t<m_state_traj.size(); t++){
        for (int k=0; k<m_state_traj[0].size(); k++)
            file_s << m_state_traj[t][k] << "\t";
        file_s << "\n";
    }

    std::ofstream file_r;
    file_r.open(dir + "/ev_return.txt");
    file_r << "Return\tEpisode_length\n";
    for (int t=0; t<m_return_traj.size(); t++){
        file_r << m_return_traj[t] << "\t" << m_ep_len_traj[t] << "\n";
    }

    std::ofstream file_as;
    file_as.open(dir + "/ev_aggr_st.txt");
    file_as << "Index\tDescription\n";
    for (int t=0; t<m_aggr_st_traj.size(); t++){
        file_as << m_aggr_st_traj[t] << "\t" << (*env).aggr_state_descr()[m_aggr_st_traj[t]] << "\n";
    }
}