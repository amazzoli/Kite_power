#include "eval.h"


void Eval::run(int n_steps, int n_point_traj){

    int traj_step = round(n_steps/float(n_point_traj));
    std::cout << n_steps << " " << traj_step << " " << n_point_traj << "\n";

    m_state_traj = vec2d(0);
    m_return_traj = vecd(0);
    //m_obs_traj = vec_obs(0);
    (*m_env).reset_state();

    int aggr_state = (*m_env).aggr_state();
    double ret = 0;
    double gamma_factor = 1;
    Perc perc(10, n_steps-1);

    // Main loop
    for (int t=0; t<n_steps; ++t){
        perc.step(t);

        // Extracting the action from the policy
        std::discrete_distribution<int> dist (m_policy[aggr_state].begin(), m_policy[aggr_state].end());
        int action = dist(m_generator);

        // Envitonmental step
        env_info info = (*m_env).step(action);
        int new_aggr_state = (*m_env).aggr_state();
        ret += info.reward * gamma_factor;

        // Building the state trajectory
        if (t%traj_step == 0){
            m_state_traj.push_back((*m_env).state());
            // aux_obs = vecd(m_obs.size());
            // for (int o=0; o<m_obs.size(); o++)
            //     aux_obs.push_back(m_obs[o].compute(*env));
        }

        if (info.done){ // Terminal state
            (*m_env).reset_state();
            aggr_state = (*m_env).aggr_state();
            gamma_factor = 1;
            m_return_traj.push_back(ret);
            ret = 0;
        } else {// Non-terminal state
            aggr_state = new_aggr_state;
            gamma_factor *= m_gamma;
        }

    }
}


void Eval::print_traj(std::string dir) const {
    std::ofstream file_s;
    file_s.open(dir + "/ev_states.txt");
    for (int k=0; k<m_state_traj[0].size(); k++)
        file_s << (*m_env).state_descr()[k] << "\t";
    file_s << "\n";
    for (int t=0; t<m_state_traj.size(); t++){
        for (int k=0; k<m_state_traj[0].size(); k++)
            file_s << m_state_traj[t][k] << "\t";
        file_s << "\n";
    }

    std::ofstream file_r;
    file_r.open(dir + "/ev_return.txt");
    for (int t=0; t<m_return_traj.size(); t++){
        file_r << m_return_traj[t] << "\n";
    }
}