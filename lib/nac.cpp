#include "nac.h"


AC::AC(Environment* env, dictd& params, std::mt19937& generator) : 
m_env{env}, m_generator{generator} {
    m_gamma = params["gamma"];
    // Learning rate scheduling
    m_lr_crit = d_i_fnc{  // Critic
        [&params](int step) { 
            return plaw_dacay(step, params["a_burn"], params["a_expn"], params["a0"], params["ac"]);
        }
    };
    m_lr_act = d_i_fnc{  // Actor
        [&params](int step) { 
            return plaw_dacay(step, params["b_burn"], params["b_expn"], params["b0"], params["bc"]);
        }
    };
}


void AC::run(int n_steps, int n_point_traj, double init_values){

    int traj_step = round(n_steps/float(n_point_traj));

    // Value parameters for each aggregate state
    vecd v_pars = vecd((*m_env).n_aggr_state());
    for (int s=0; s<(*m_env).n_aggr_state(); ++s) 
        v_pars[s] = init_values;

    // Policy parameters for each aggr state (first dim) and each action (second dim)
    vec2d p_pars = vec2d(0);
    for (int s=0; s<(*m_env).n_aggr_state(); ++s) 
        p_pars.push_back(vecd((*m_env).n_actions()));

    // Trejectory init
    m_policy_par_traj = vec3d(0);
    m_value_traj = vec2d(0);
    m_return_traj = vecd(0);

    // Init
    init_pars();
    (*m_env).reset_state();
    m_curr_aggr_state = (*m_env).aggr_state();
    double gamma_factor = 1;
    double ret = 0;
    Perc perc(10, n_steps-1);

    // Main loop
    for (int t=0; t<n_steps; ++t){
        perc.step(t);

        // Policy from parameters
        m_curr_policy = boltzman_weights(p_pars[m_curr_aggr_state]);

        // Extracting the action from the policy
        std::discrete_distribution<int> dist (m_curr_policy.begin(), m_curr_policy.end());
        m_curr_action = dist(m_generator);

        // Envitonmental step
        env_info info = (*m_env).step(m_curr_action);
        int new_aggr_state = (*m_env).aggr_state();

        // Temporal difference error
        double td_error = info.reward + m_gamma * v_pars[new_aggr_state] - v_pars[m_curr_aggr_state];
        ret += info.reward * gamma_factor;

        // Critic update
        m_curr_crit_lr = m_lr_crit(t) * gamma_factor;
        v_pars[m_curr_aggr_state] += m_curr_crit_lr  * td_error;

        // Actor update
        m_curr_act_lr = m_lr_act(t) * gamma_factor;
        delta_act_updt(p_pars, td_error);
        // for (int a=0; a<p_pars[m_curr_aggr_state].size(); a++) {
        //     p_pars[m_curr_aggr_state][a] = std::max(p_pars[m_curr_aggr_state][a], p_par_bounds[0]);
        //     p_pars[m_curr_aggr_state][a] = std::min(p_pars[m_curr_aggr_state][a], p_par_bounds[1]);
        // }

        if (info.done){ // Terminal state
            v_pars[new_aggr_state] += m_curr_crit_lr * ((*m_env).terminal_reward(m_gamma) - v_pars[new_aggr_state]);
            (*m_env).reset_state();
            m_curr_aggr_state = (*m_env).aggr_state();
            gamma_factor = 1;
            m_return_traj.push_back(ret);
            ret = 0;
        } else { // Non-terminal state
            m_curr_aggr_state = new_aggr_state;
            gamma_factor *= m_gamma;
        }

        // Building the trajectory
        if (t%traj_step == 0){
            m_policy_par_traj.push_back(p_pars);
            m_value_traj.push_back(v_pars);
        }
    }
}


void AC::delta_act_updt(vec2d& p_pars, double td_error){ 
    for (int a=0; a<p_pars[m_curr_aggr_state].size(); a++){
        if (a == m_curr_action) 
            p_pars[m_curr_aggr_state][a] += m_curr_act_lr * td_error * (1 - m_curr_policy[a]); 
        else 
            p_pars[m_curr_aggr_state][a] -= m_curr_act_lr * td_error * m_curr_policy[a];
    }
}


void AC::print_traj(std::string out_dir) const {
    std::ofstream out_p, out_p1, out_v, out_r;
    out_p.open(out_dir + "policy_traj.txt");
    out_p1.open(out_dir + "policyp_traj.txt");
    out_v.open(out_dir + "value_traj.txt");
    out_r.open(out_dir + "return_traj.txt");
    for (int t=0; t<m_policy_par_traj.size(); t++){
        for (int k=0; k<m_policy_par_traj[0].size(); k++){
            out_v << m_value_traj[t][k] << "\t";
            vecd policy = boltzman_weights(m_policy_par_traj[t][k]);
            for (int a=0; a<policy.size(); a++){
                out_p << policy[a];
                out_p1 << m_policy_par_traj[t][k][a];
                if (a < policy.size()-1){
                    out_p << ",";
                    out_p1 << ",";
                }
            }
            out_p << "\t";
            out_p1 << "\t";
        }
        out_v << "\n";
        out_p << "\n";
        out_p1 << "\n";
    }
    for (int t=0; t<m_return_traj.size(); t++) out_r << m_return_traj[t] << "\n";
}


void AC::print_best_pol(std::string out_dir) const {
    std::ofstream out;
    out.open(out_dir + "best_policy.txt");
    int fin_time = m_policy_par_traj.size()-1;
    for (int k=0; k<m_policy_par_traj[fin_time].size(); k++){
        vecd policy = boltzman_weights(m_policy_par_traj[fin_time][k]);
        for (int a=0; a<policy.size()-1; a++){
            out << policy[a];
            if (a < policy.size()-2)
                out << " ";
        }
        out << "\n";
    }
}


// NATURAL ACTOR CRITIC WITH ADVANTAGE PARAMETERS

void NAC_AP::init_pars(){
    m_ap_par = vec2d(0);
    for (int s=0; s<(*m_env).n_aggr_state(); ++s) 
        m_ap_par.push_back(vecd((*m_env).n_actions()));
}

void NAC_AP::delta_act_updt(vec2d& p_pars, double td_error){ 

    double aux_t = td_error - m_ap_par[m_curr_aggr_state][m_curr_action];
    for (int a=0; a<p_pars[m_curr_aggr_state].size(); a++)
        aux_t += m_curr_policy[a] * m_ap_par[m_curr_aggr_state][a];
    for (int a=0; a<p_pars[m_curr_aggr_state].size(); a++){
        if (a == m_curr_action) 
            m_ap_par[m_curr_aggr_state][a] += m_curr_crit_lr * (1 - m_curr_policy[a]) * aux_t; 
        else 
            m_ap_par[m_curr_aggr_state][a] -= m_curr_crit_lr * m_curr_policy[a] * aux_t;
    }
    for (int s=0; s<m_ap_par.size(); s++)
        for (int a=0; a<p_pars[m_curr_aggr_state].size(); a++)
            p_pars[s][a] += m_curr_act_lr * m_ap_par[s][a];
}