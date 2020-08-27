#include "nac.h"


AC::AC(Environment* env, const dictd& params, std::mt19937& generator) : 
Algorithm(env, generator) {
    m_gamma = params.at("gamma");
    // Learning rate scheduling
    m_lr_crit = d_i_fnc{  // Critic
        [&params](int step) { 
            return plaw_dacay(step, params.at("a_burn"), params.at("a_expn"), params.at("a0"), params.at("ac"));
        }
    };
    m_lr_act = d_i_fnc{  // Actor
        [&params](int step) { 
            return plaw_dacay(step, params.at("b_burn"), params.at("b_expn"), params.at("b0"), params.at("bc"));
        }
    };
}


void AC::run(const param& params){

    // Without priors -> constant init values and flat policy
    if (params.s.find("init_val_path") == params.s.end() && params.s.find("init_pol_path") == params.s.end()){
        run(params.d.at("n_steps"), params.d.at("init_values"), params.d.at("traj_points"));

    // With value prior
    } else if (params.s.find("init_val_path") != params.s.end() && params.s.find("init_pol_path") == params.s.end()){
        vecd val = read_best_val( params.s.at("init_val_path") );
        run(params.d.at("n_steps"), val, params.d.at("traj_points"));

    // With policy prior
    } else if (params.s.find("init_val_path") == params.s.end() && params.s.find("init_pol_path") != params.s.end()){
        vec2d pol = read_best_pol( params.s.at("init_pol_path") );
        run(params.d.at("n_steps"), params.d.at("init_values"), pol, params.d.at("traj_points"));

    // With policy and value priors
    } else {
        vecd val = read_best_val( params.s.at("init_val_path") );
        vec2d pol = read_best_pol( params.s.at("init_pol_path") );
        run(params.d.at("n_steps"), val, pol, params.d.at("traj_points"));
    }
}


void AC::run(int n_steps, double init_values, int n_point_traj){
    run(n_steps, const_values(init_values), flat_policy(), n_point_traj);
}


void AC::run(int n_steps, vecd init_values, int n_point_traj){
    run(n_steps, init_values, flat_policy(), n_point_traj);
}


void AC::run(int n_steps, double init_values, vec2d init_policies, int n_point_traj){
    run(n_steps, const_values(init_values), init_policies, n_point_traj);
}


void AC::run(int n_steps, vecd init_values, vec2d init_policies, int n_point_traj){

    init(init_values, init_policies);
    reset_env();
    int traj_step = round(n_steps/float(n_point_traj));
    Perc perc(5, n_steps-1);

    // Main loop
    for (int t=0; t<n_steps; ++t){
        perc.step(t);

        // Policy from parameters
        par2pol_boltzmann(curr_p_pars[curr_aggr_state], curr_policy);

        // Extracting the action from the policy
        std::discrete_distribution<int> dist (curr_policy.begin(), curr_policy.end());
        curr_action = dist(generator);

        // Envitonmental step
        env_info info = (*env).step(curr_action);
        int new_aggr_state = (*env).aggr_state();

        // Temporal difference error
        double td_error = info.reward + m_gamma * curr_v_pars[new_aggr_state] - curr_v_pars[curr_aggr_state];
        curr_return += info.reward * curr_gamma_fact;

        // Critic update
        curr_crit_lr = m_lr_crit(t) * curr_gamma_fact;
        curr_v_pars[curr_aggr_state] += curr_crit_lr  * td_error;

        // Actor update
        curr_act_lr = m_lr_act(t) * curr_gamma_fact;
        delta_act_updt(td_error);

        if (info.done){ // Terminal state
            curr_return += (*env).terminal_reward(m_gamma) * curr_gamma_fact;
            m_return_traj.push_back(curr_return);
            curr_v_pars[new_aggr_state] += curr_crit_lr * ((*env).terminal_reward(m_gamma) - curr_v_pars[new_aggr_state]);
            reset_env();
        } else { // Non-terminal state
            curr_aggr_state = new_aggr_state;
            curr_gamma_fact *= m_gamma;
        }

        // Building the trajectory
        if (t%traj_step == 0){
            m_policy_par_traj.push_back(curr_p_pars);
            m_value_traj.push_back(curr_v_pars);
        }
    }
}


void AC::delta_act_updt(double td_error){ 
    for (int a=0; a<curr_p_pars[curr_aggr_state].size(); a++){
        if (a == curr_action) 
            curr_p_pars[curr_aggr_state][a] += curr_act_lr * td_error * (1 - curr_policy[a]); 
        else 
            curr_p_pars[curr_aggr_state][a] -= curr_act_lr * td_error * curr_policy[a];
    }
}


void AC::init(vecd init_values, vec2d init_policies){
    // Trejectory init
    m_policy_par_traj = vec3d(0);
    m_value_traj = vec2d(0);
    m_return_traj = vecd(0);
    // Parameter init
    curr_v_pars = init_values;
    curr_p_pars = vec2d(0);
    for (int k=0; k<(*env).n_aggr_state(); k++) {
        vecd par_at_state = vecd((*env).n_actions());
        pol2par_boltzmann(init_policies[k], par_at_state);
        curr_p_pars.push_back(par_at_state);
    }
    init_pars();
    curr_policy = vecd((*env).n_actions());
}


void AC::reset_env(){
    (*env).reset_state();
    curr_aggr_state = (*env).aggr_state();
    // Algorithm init
    curr_gamma_fact = 1;
    curr_return = 0;
}


vecd AC::const_values(double val){
    return vecd((*env).n_aggr_state(), val);
}


vec2d AC::flat_policy(){
    vec2d policy = vec2d(0);
    for (int s=0; s<(*env).n_aggr_state(); ++s) 
        policy.push_back( vecd( (*env).n_actions(), 1.0/(double)(*env).n_actions() ) );
    return policy;
}


void AC::print_output(std::string out_dir) const {

    // PRINTING THE TRAJECTORIES
    std::ofstream out_p, out_p1, out_v, out_r;
    out_p.open(out_dir + "policy_traj.txt");
    out_v.open(out_dir + "value_traj.txt");
    out_r.open(out_dir + "return_traj.txt");
    // Headers
    for (int k=0; k<m_value_traj[0].size(); k++){
        out_v << (*env).aggr_state_descr()[k] << "\t";
        out_p << (*env).aggr_state_descr()[k] << "\t";
    }
    out_v << "\n";
    out_p << "\n";
    for (int a=0; a<m_policy_par_traj[0][0].size(); a++){
        out_p << (*env).action_descr()[a];
        if (a < m_policy_par_traj[0][0].size()-1) out_p << ",";
    }
    out_p << "\n";
    // Body
    for (int t=0; t<m_policy_par_traj.size(); t++){
        for (int k=0; k<m_policy_par_traj[0].size(); k++){
            out_v << m_value_traj[t][k] << "\t";
            vecd policy = vecd(m_policy_par_traj[t][k].size());
            par2pol_boltzmann(m_policy_par_traj[t][k], policy);
            for (int a=0; a<policy.size(); a++){
                out_p << policy[a];
                if (a < policy.size()-1) out_p << ",";
            }
            out_p << "\t";
        }
        out_v << "\n";
        out_p << "\n";
    }
    for (int t=0; t<m_return_traj.size(); t++) out_r << m_return_traj[t] << "\n";
    out_p.close();
    out_v.close();
    out_r.close();

    // PRINTING THE BEST VALUES AND THE BEST POLICIES
    out_p.open(out_dir + "best_policy.txt");
    out_v.open(out_dir + "best_value.txt");
    int fin_time = m_policy_par_traj.size()-1;
    for (int k=0; k<m_policy_par_traj[fin_time].size(); k++){
        out_v << m_value_traj[fin_time][k] << "\n";
        vecd policy = vecd(m_policy_par_traj[fin_time][k].size());
        par2pol_boltzmann(m_policy_par_traj[fin_time][k], policy);
        for (int a=0; a<policy.size()-1; a++){
            if (policy[a]>0 && policy[a]<std::numeric_limits<double>::min())
                policy[a] = 0;
            out_p << policy[a];
            if (a < policy.size()-2)
                out_p << " ";
        }
        out_p << "\n";
    }
}


// NATURAL ACTOR CRITIC WITH ADVANTAGE PARAMETERS

void NAC_AP::init_pars(){
    m_ap_par = vec2d(0);
    for (int s=0; s<(*env).n_aggr_state(); ++s) 
        m_ap_par.push_back(vecd((*env).n_actions()));
}

void NAC_AP::delta_act_updt(vec2d& p_pars, double td_error){ 

    double aux_t = td_error - m_ap_par[curr_aggr_state][curr_action];
    for (int a=0; a<p_pars[curr_aggr_state].size(); a++)
        aux_t += curr_policy[a] * m_ap_par[curr_aggr_state][a];
    for (int a=0; a<p_pars[curr_aggr_state].size(); a++){
        if (a == curr_action) 
            m_ap_par[curr_aggr_state][a] += curr_crit_lr * (1 - curr_policy[a]) * aux_t; 
        else 
            m_ap_par[curr_aggr_state][a] -= curr_crit_lr * curr_policy[a] * aux_t;
    }
    for (int s=0; s<m_ap_par.size(); s++)
        for (int a=0; a<p_pars[curr_aggr_state].size(); a++)
            p_pars[s][a] += curr_act_lr * m_ap_par[s][a];
}