#include "nac.h"


AC::AC(Environment* env, const param& params, std::mt19937& generator) : 
RLAlgorithm(env, params, generator) {

    // Learning rate scheduling
    lr_crit = d_i_fnc{  // Critic
        [&params](int step) { 
            return plaw_dacay(step, params.d.at("a_burn"), params.d.at("a_expn"), params.d.at("a0"), params.d.at("ac"));
        }
    };
    lr_act = d_i_fnc{  // Actor
        [&params](int step) { 
            return plaw_dacay(step, params.d.at("b_burn"), params.d.at("b_expn"), params.d.at("b0"), params.d.at("bc"));
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
}


void AC::init(const param& params){

    // Trejectory init
    policy_par_traj = vec3d(0);
    value_traj = vec2d(0);
    return_traj = vecd(0);

    // Value parameter init
    if (params.s.find("init_val_path") != params.s.end())
        curr_v_pars = read_value( params.s.at("init_val_path") );
    else
        curr_v_pars = const_values( params.d.at("init_values") );
    
    // Policy parameters init
    curr_policy = vecd((*env).n_actions());
    vec2d aux_pol = vec2d(0);
    if (params.s.find("init_pol_path") != params.s.end())
        aux_pol = read_policy( params.s.at("init_pol_path") );
    else
        aux_pol = flat_policy();
    curr_p_pars = vec2d(0);
    for (int k=0; k<(*env).n_aggr_state(); k++) {
        vecd par_at_state = vecd((*env).n_actions());
        pol2par_boltzmann(aux_pol[k], par_at_state);
        curr_p_pars.push_back(par_at_state);
    }

    // Child algorithm init
    child_init();
}


int AC::get_action() {

        // Policy from parameters
        par2pol_boltzmann(curr_p_pars[curr_aggr_state], curr_policy);

        // Extracting the action from the policy
        std::discrete_distribution<int> dist (curr_policy.begin(), curr_policy.end());
        return dist(generator);
}


void AC::learning_update() {

    double td_error = curr_info.reward + m_gamma * curr_v_pars[curr_new_aggr_state] - curr_v_pars[curr_aggr_state];

    // Critic update
    curr_crit_lr = lr_crit(curr_step) * curr_gamma_fact;
    curr_v_pars[curr_aggr_state] += curr_crit_lr  * td_error;
    if (curr_info.done)
        curr_v_pars[curr_new_aggr_state] += curr_crit_lr * ((*env).terminal_reward(m_gamma) - curr_v_pars[curr_new_aggr_state]);

    // Actor update
    curr_act_lr = lr_act(curr_step) * curr_gamma_fact;
    child_update(td_error);
}


void AC::build_traj() {
    vec2d p_pars = vec2d(0);
    vecd vals = vecd(0);
    for (int k : traj_states){
        p_pars.push_back(curr_p_pars[k]);
        vals.push_back(curr_v_pars[k]);
    }
    policy_par_traj.push_back(p_pars);
    value_traj.push_back(vals);

    vecd lrs = vecd{curr_crit_lr, curr_act_lr};
    lr_traj.push_back(lrs);
}


void AC::print_traj(std::string out_dir) const {

    // PRINTING THE TRAJECTORIES
    std::ofstream out_p, out_lr, out_v;
    out_p.open(out_dir + "policy_traj.txt");
    out_v.open(out_dir + "value_traj.txt");
    out_lr.open(out_dir + "lr_traj.txt");
    // Headers
    out_lr << "Critic_lr\tActor_lr\n";
    for (int k : traj_states){
        out_v << (*env).aggr_state_descr()[k] << "\t";
        out_p << (*env).aggr_state_descr()[k] << "\t";
    }
    out_v << "\n";
    out_p << "\n";
    for (int a=0; a<policy_par_traj[0][0].size(); a++){
        out_p << (*env).action_descr()[a];
        if (a < policy_par_traj[0][0].size()-1) out_p << ",";
    }
    out_p << "\n";
    // Body
    for (int t=0; t<policy_par_traj.size(); t++){
        out_lr << lr_traj[t][0] << "\t" << lr_traj[t][1] << "\n";
        for (int k=0; k<policy_par_traj[0].size(); k++){
            out_v << value_traj[t][k] << "\t";
            vecd policy = vecd(policy_par_traj[t][k].size());
            par2pol_boltzmann(policy_par_traj[t][k], policy);
            for (int a=0; a<policy.size(); a++){
                out_p << policy[a];
                if (a < policy.size()-1) out_p << ",";
            }
            out_p << "\t";
        }
        out_v << "\n";
        out_p << "\n";
    }
    out_p.close();
    out_v.close();
    out_lr.close();

    // PRINTING THE BEST VALUES AND THE BEST POLICIES
    out_p.open(out_dir + "best_policy.txt");
    out_v.open(out_dir + "best_value.txt");
    for (int k=0; k<curr_p_pars.size(); k++){
        out_v << curr_v_pars[k] << "\n";
        vecd policy = vecd(curr_p_pars[k].size());
        par2pol_boltzmann(curr_p_pars[k], policy);
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


void AC::child_update(double td_error){ 
    for (int a=0; a<curr_p_pars[curr_aggr_state].size(); a++){
        if (a == curr_action) 
            curr_p_pars[curr_aggr_state][a] += curr_act_lr * td_error * (1 - curr_policy[a]); 
        else 
            curr_p_pars[curr_aggr_state][a] -= curr_act_lr * td_error * curr_policy[a];
    }
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



// NATURAL ACTOR CRITIC WITH ADVANTAGE PARAMETERS

void NAC_AP::child_init(){
    m_ap_par = vec2d(0);
    for (int s=0; s<(*env).n_aggr_state(); ++s) 
        m_ap_par.push_back(vecd((*env).n_actions()));
}

void NAC_AP::child_update(double td_error){ 

    double aux_t = td_error - m_ap_par[curr_aggr_state][curr_action];
    for (int a=0; a<curr_p_pars[curr_aggr_state].size(); a++)
        aux_t += curr_policy[a] * m_ap_par[curr_aggr_state][a];
    for (int a=0; a<curr_p_pars[curr_aggr_state].size(); a++){
        if (a == curr_action) 
            m_ap_par[curr_aggr_state][a] += curr_crit_lr * (1 - curr_policy[a]) * aux_t; 
        else 
            m_ap_par[curr_aggr_state][a] -= curr_crit_lr * curr_policy[a] * aux_t;
    }
    for (int s=0; s<m_ap_par.size(); s++)
        for (int a=0; a<curr_p_pars[curr_aggr_state].size(); a++)
            curr_p_pars[s][a] += curr_act_lr * m_ap_par[s][a];
}