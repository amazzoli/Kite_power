#include "eval.h"


Eval::Eval(Environment* env, const param& params, std::mt19937& generator):
RLAlgorithm(env, params, generator) {
    if (params.d.find("epsilon") != params.d.end())
        eps = params.d.at("epsilon");
    else
        eps = 0;

    m_quality = read_quality(params.s.at("quality_path"));

    if (params.s.find("quality2_path") != params.s.end()){
        m_quality2 = read_quality(params.s.at("quality2_path"));
    }
    else {
        m_quality2 = m_quality;
    }

    unif_dist = std::uniform_real_distribution<double>(0.0,1.0);
    unif_act_dist = std::uniform_int_distribution<int>(0, (*env).n_actions()-1);
}


void Eval::init(const param& params){
    m_state_traj = vec2d(0);
    m_aggr_st_traj = veci(0);
    action_traj = veci(0);
}


int Eval::get_action(bool sw) {
    double u = unif_dist(generator);
    int action;
    if (u < eps){
        action = unif_act_dist(generator);
    }
    else {
        if (!sw) {
          auto max_elem = std::max_element(m_quality[curr_aggr_state].begin(), m_quality[curr_aggr_state].end());
          action = std::distance(m_quality[curr_aggr_state].begin(), max_elem);
        } else {
          auto max_elem = std::max_element(m_quality2[curr_aggr_state].begin(), m_quality2[curr_aggr_state].end());
          action = std::distance(m_quality2[curr_aggr_state].begin(), max_elem);
        }
    }
    return action;
}

void Eval::learning_update() {}


void Eval::build_traj() {
    m_state_traj.push_back((*env).state());
    m_aggr_st_traj.push_back(curr_aggr_state);
    action_traj.push_back(curr_action);
}


void Eval::print_traj(std::string dir) const {

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
    file_s.close();

    std::ofstream file_as;
    file_as.open(dir + "/ev_aggr_st.txt");
    file_as << "Index\tDescription\n";
    for (int t=0; t<m_aggr_st_traj.size(); t++){
        file_as << m_aggr_st_traj[t] << "\t" << (*env).aggr_state_descr()[m_aggr_st_traj[t]] << "\t" << action_traj[t] << "\n";
    }
    file_as.close();
}
