#include "utils.h"


void par2pol_boltzmann(const vecd& params, vecd& policy){
    double max = *std::max_element(params.begin(), params.end());
    double norm = 0;
    for (int i=0; i<params.size(); i++){
        double val = exp(params[i]-max);
        policy[i] = val;
        norm += val;
    }
    for (int i=0; i<policy.size(); i++) policy[i] /= norm;
}


void pol2par_boltzmann(const vecd& policy, vecd& params){
    for (int i=0; i<params.size(); i++){
        double val = 0;
        if (policy[i] > 10E-20)
            params[i] = log(policy[i]);
        else
            params[i] = -20;
    }
}


double plaw_dacay(double t, double t_burn, double expn, double a0, double ac){
    if (t < t_burn) 
        return a0;
    else {
        return a0 * ac / (ac + pow(t-t_burn, expn));
    }
}

dictd parse_param_file(std::string file_path){

    dictd dict;
    std::ifstream param_file (file_path);
    if (!param_file.is_open())
        throw std::runtime_error("Error in opening the parameter file at "+file_path);

    std::string line;
    while ( getline (param_file, line) )
    {
        std::size_t tab_pos = line.find("\t");
        std::string key = line.substr(0,tab_pos);
        double value = std::stod(line.substr(tab_pos+1, std::string::npos));
        dict[key] = value;
    }
    param_file.close();

    if (dict.size() == 0)
        throw std::runtime_error("Empty parameter file");

    return dict;
}


vec2d read_best_pol(std::string file_path) {

    vec2d policy(0);
    std::ifstream pol_file (file_path);
    if (!pol_file.is_open())
        throw std::runtime_error("Error in opening the best_policy file at "+file_path);

    std::string line;
    while ( getline (pol_file, line) )
    {
        std::size_t tab_pos = line.find(" ");
        std::string p = line.substr(0, tab_pos);
        vecd pol_at_state = {std::stod(p)};
        double cum_p = std::stod(p);
        while (tab_pos != std::string::npos){
            std::size_t next_tab_pos = line.find(" ", tab_pos+1);
            std::string p = line.substr(tab_pos+1, next_tab_pos-tab_pos);
            pol_at_state.push_back(std::stod(p));
            cum_p += std::stod(p);
            tab_pos = next_tab_pos;
        }
        if (cum_p > 1+10E-6) 
            throw std::runtime_error("Not normalized policy in "+file_path+" "+std::to_string(cum_p));
        pol_at_state.push_back(1-cum_p);
        policy.push_back(pol_at_state);
    }
    pol_file.close();

    return policy;
}


vecd read_best_val(std::string file_path) {

    vecd val(0);
    std::ifstream val_file (file_path);
    if (!val_file.is_open())
        throw std::runtime_error("Error in opening the best_policy file at "+file_path);

    std::string line;
    while ( getline (val_file, line) )
        val.push_back(std::stod(line));
    val_file.close();

    return val;
}