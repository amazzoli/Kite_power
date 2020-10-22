#include "kite.h"


// KITE 2D HAVING THE ATTACK ANGLE AS AGGREGATE STATE


/* Constructor */
Kite3d::Kite3d(const param& params, Wind3d* wind, std::mt19937& generator) : 
Kite{params, generator}, wind{wind} {

    m_state = vecd(state_descr().size());
    bank = params.vecd.at("banks");

    // SETTING SPECIFIC PARAMETERS
    init_theta = params.d.at("init_theta");
    init_dtheta = params.d.at("init_dtheta");
    init_alpha_ind = params.d.at("init_alpha");
    init_phi = params.d.at("init_phi");
    init_dphi = params.d.at("init_dphi");
    init_bank_ind = params.d.at("init_bank");

    //debug_file.open("debug.txt");
}


const std::string Kite3d::descr() const {
    return "3d kite. Attack and bank angles observed. Attack and bank angles controlled. " + wind->descr();
}


const vecs Kite3d::state_descr() const {
    vecs m_state_descr = {
        "kite_pos_x",   // 0
        "kite_pos_y",   // 1
        "kite_pos_z",   // 2
        "kite_vel_x",   // 3
        "kite_vel_y",   // 4
        "kite_vel_z",   // 5
        "kite_acc_x",   // 6 
        "kite_acc_y",   // 7
        "kite_acc_z",   // 8
        "block_pos_x",  // 9
        "block_vel_x",  // 10
        "block_acc_x",  // 11 
    };
    return m_state_descr;
}


const vecd& Kite3d::state() { 
    m_state[0] = pos_kite[0];
    m_state[1] = pos_kite[1];
    m_state[2] = pos_kite[2];
    m_state[3] = vel_kite[0];
    m_state[4] = vel_kite[1];
    m_state[5] = vel_kite[2];
    m_state[6] = acc_kite[0];
    m_state[7] = acc_kite[1];
    m_state[8] = acc_kite[2];
    m_state[9] = x_block;
    m_state[10] = vx_block;
    m_state[11] = ax_block;
    return m_state; 
}


const vecs Kite3d::aggr_state_descr() const {
    vecs m_aggr_state_descr = vecs(0);
    for (int a=0; a<n_alphas(); a++) 
        for (int b=0; b<n_banks(); b++) 
            m_aggr_state_descr.push_back("attack_ang_"+std::to_string(alphas[a])+",bank_angle_"+std::to_string(bank[b]));
    return m_aggr_state_descr;
}


const vecs Kite3d::action_descr() const {
    vecs m_act_descr = { "a_decr_b_decr", "a_decr_b_stay", "a_decr_b_incr", \
                         "a_stay_b_decr", "a_stay_b_stay", "a_stay_b_incr", \
                         "a_incr_b_decr", "a_incr_b_stay", "a_incr_b_incr" };
    return m_act_descr;
}

/* Aggregate state: it is defined by the current attack angle and bank angle */
int Kite3d::aggr_state() const {
    return bank_ind + n_banks() * alpha_ind;
}


/* Initial configuration given the initial theta and dtheta*/
int Kite3d::reset_kite(){

    time_sec = 0;

    // Initial bank angle (attack angle set in base class)
    bank_ind = init_bank_ind;

    // Block position, velocity, acceleration
    x_block = 0;
    vx_block = 0;
    ax_block = 0;
    // Kite position
    pos_kite[0] = x_block + R*sin(init_theta)*cos(init_phi);
    pos_kite[1] = R*sin(init_theta)*sin(init_phi);
    pos_kite[2] = R*cos(init_theta);
    // Kite velocity
    vel_kite[0] = vx_block + R*cos(init_theta)*cos(init_phi)*init_dtheta - R*sin(init_theta)*sin(init_phi)*init_dphi;
    vel_kite[1] = R*cos(init_theta)*sin(init_phi)*init_dtheta + R*sin(init_theta)*cos(init_phi)*init_dphi;
    vel_kite[2] = -R*sin(init_theta)*init_dtheta;
    // Kite acceleration
    acc_kite[0] = R*(ax_block/R -sin(init_theta)*cos(init_phi)*(init_dtheta*init_dtheta + init_dphi*init_dphi) - \
                 2*cos(init_theta)*sin(init_phi)*init_dtheta*init_dphi);
    acc_kite[1] = R*(sin(init_theta)*sin(init_phi)*(init_dtheta*init_dtheta + init_dphi*init_dphi) + \
                 2*cos(init_theta)*cos(init_phi)*init_dtheta*init_dphi);
    acc_kite[2] = -R*cos(init_theta)*init_dtheta*init_dtheta;

    x_diff = pos_kite[0] - x_block;
    double* v_wind = (*wind).init(pos_kite[0], pos_kite[1], pos_kite[2]);
    va[0] = vel_kite[0] - v_wind[0];
    va[1] = vel_kite[1] - v_wind[1];
    va[2] = vel_kite[2] - v_wind[2];
    beta = atan2(va[2], va[0]);
    theta = atan2(sqrt(x_diff*x_diff + pos_kite[1]*pos_kite[1]), pos_kite[2]);
    phi = atan2(pos_kite[1], x_diff);

    return aggr_state();
}


void Kite3d::impose_action(int a){
    switch (a){
        case 0:
            alpha_ind = std::max(alpha_ind - 1, 0);
            bank_ind = std::max(bank_ind - 1, 0);
        break;
        case 1:
            alpha_ind = std::max(alpha_ind - 1, 0);
        break;
        case 2:
            alpha_ind = std::max(alpha_ind - 1, 0);
            bank_ind = std::min(bank_ind + 1, n_banks()-1);
        break;
        case 3:
            bank_ind = std::max(bank_ind - 1, 0);
        break;
        case 4:
        break;
        case 5:
            bank_ind = std::min(bank_ind + 1, n_banks()-1);
        break;
        case 6:
            alpha_ind = std::min(alpha_ind + 1, n_alphas()-1);
            bank_ind = std::max(bank_ind - 1, 0);
        break;
        case 7:
            alpha_ind = std::min(alpha_ind + 1, n_alphas()-1);
        break;
        case 8:
            alpha_ind = std::min(alpha_ind + 1, n_alphas()-1);
            bank_ind = std::min(bank_ind + 1, n_banks()-1);
        break;
        default:
            throw std::runtime_error ( "Invalid action " + std::to_string(a) + "\n" );
    }
}


bool Kite3d::integrate_trajectory() {

    x_diff = pos_kite[0] - x_block;
    theta = atan2(sqrt(x_diff*x_diff + pos_kite[1]*pos_kite[1]), pos_kite[2]);
    phi = atan2(pos_kite[1], x_diff);

    // Aerodynamical forces
    compute_F_aer();

    // Tension and friction
    if ( fabs(vx_block) < v_threshold ) // BLOCK NOT MOVING
        compute_tension_still();
    else // BLOCK MOVING
        compute_tension_move();

    // Update positions, velocities and accelerations
    update_state();

    time_sec += h;

    // Check if a terminal state is reached. Kite fallen
    if (pos_kite[2] <= fall_limit) return true;
    return false;
}


/* Compute the aerodynamical forces */
void Kite3d::compute_F_aer(){
    // Apparent velocity
    double* v_wind = (*wind).velocity(pos_kite[0], pos_kite[1], pos_kite[2], time_sec);
    //debug_file << pos_kite[0] << " " << v_wind[0] << " " << pos_kite[1] << " " << v_wind[1] << " " << pos_kite[2] << " " << v_wind[2] << "\n";
    va[0] = vel_kite[0] - v_wind[0];
    va[1] = vel_kite[1] - v_wind[1];
    va[2] = vel_kite[2] - v_wind[2];
    double va_mod = sqrt(va[0]*va[0] + va[1]*va[1] + va[2]*va[2]);
    beta = atan2(va[2], va[0]);

    double t1[3] = {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)}; 
    if (pos_kite[1] == 0) t1[1] = 0; // Errore in approx quando phi=pi-greco
    double t2[3] = {t1[1]*va[2] - t1[2]*va[1], t1[2]*va[0] - t1[0]*va[2], t1[0]*va[1] - t1[1]*va[0]};
    double t2_mod = sqrt(t2[0]*t2[0] + t2[1]*t2[1] + t2[2]*t2[2]);
    t2[0] /= t2_mod; t2[1] /= t2_mod; t2[2] /= t2_mod;
    double t3[3] = {va[1]*t2[2] - va[2]*t2[1], va[2]*t2[0] - va[0]*t2[2], va[0]*t2[1] - va[1]*t2[0]};
    double t3_mod = sqrt(t3[0]*t3[0] + t3[1]*t3[1] + t3[2]*t3[2]);
    t3[0] /= t3_mod; t3[1] /= t3_mod; t3[2] /= t3_mod;

    // Drag
    double aux_d = 0.5 * rho * CD_alpha[alpha_ind] * a_kite * va_mod;
    double drag[3] = { -aux_d*va[0], -aux_d*va[1], -aux_d*va[2] };

    // Lift
    double mu = bank[bank_ind];
    double aux_l = 0.5 * rho * CL_alpha[alpha_ind] * a_kite * va_mod * va_mod;
    double lift[3] = {aux_l*(t2[0]*sin(mu) + t3[0]*cos(mu)), aux_l*(t2[1]*sin(mu) + t3[1]*cos(mu)), aux_l*(t2[2]*sin(mu) + t3[2]*cos(mu))};

    f_aer[0] = drag[0] + lift[0];
    f_aer[1] = drag[1] + lift[1];
    f_aer[2] = drag[2] + lift[2];
}


/* Compute the tension and the friction of a still block */
void Kite3d::compute_tension_still(){

    double vx_diff = vel_kite[0] - vx_block;
    double aux_A = (f_aer[0]*x_diff + f_aer[1]*pos_kite[1] + f_aer[2]*pos_kite[2])/m_kite + (vx_diff*vx_diff + vel_kite[1]*vel_kite[1] + vel_kite[2]*vel_kite[2]);
    double aux_B = coef_friction*(cos(phi)*x_diff + sin(phi)*pos_kite[1]);
    double aux_C = R*(m_kite + m_block)/(m_kite*m_block);
    // |Mg| > |Tz|
    double T1 = (aux_A - g*(pos_kite[2] - aux_B)) / (aux_C - cos(theta)/m_block*(pos_kite[2] - aux_B));
    // |Mg| < |Tz|
    double T2 = (aux_A - g*(pos_kite[2] + aux_B)) / (aux_C - cos(theta)/m_block*(pos_kite[2] + aux_B));

    double T;
    if ( m_block*g > T1*cos(theta) ) T = T1;
    else if ( m_block*g <= T2*cos(theta) ) T = T2;
    else  throw std::runtime_error ( "Tension error for still block\n" );
    tension[0] = T * sin(theta)*cos(phi);
    tension[1] = T * sin(theta)*sin(phi);
    tension[2] = T * cos(theta);

    double N = m_block*g - tension[2];
    friction = -coef_friction*fabs(N)*cos(phi);

    // If the computed tension is less than friction force: F_friction = -Tension[0]
    if ( fabs(tension[0]) < fabs(friction) ){
        double T = (aux_A - g*pos_kite[2]) / (aux_C - sin(theta)/m_block*(cos(phi)*x_diff + sin(phi)*pos_kite[1]) - cos(theta)/m_block*pos_kite[2]);
        tension[0] = T * sin(theta)*cos(phi);
        tension[1] = T * sin(theta)*sin(phi);
        tension[2] = T * cos(theta);
        friction = -tension[0];
    }   
}


/* Compute the tension and the friction of a moving block */
void Kite3d::compute_tension_move(){

    double vx_diff = vel_kite[0] - vx_block;
    double aux_A = (f_aer[0]*x_diff + f_aer[1]*pos_kite[1] + f_aer[2]*pos_kite[2])/m_kite + (vx_diff*vx_diff + vel_kite[1]*vel_kite[1] + vel_kite[2]*vel_kite[2]);
    double aux_B = coef_friction/fabs(vx_block)*x_diff*vx_block;
    double aux_C = R*(m_kite + m_block)/(m_kite*m_block);
    // |Mg| > |Tz|
    double T1 = (aux_A - g*(pos_kite[2] - aux_B)) / (aux_C - cos(theta)/m_block*(pos_kite[2] - aux_B));
    // |Mg| < |Tz|
    double T2 = (aux_A - g*(pos_kite[2] + aux_B)) / (aux_C - cos(theta)/m_block*(pos_kite[2] + aux_B));

    double T;
    if ( m_block*g > T1*cos(theta) ) T = T1;
    else if ( m_block*g <= T2*cos(theta) ) T = T2;
    else  throw std::runtime_error ( "Tension error for moving block\n" );
    tension[0] = T * sin(theta)*cos(phi);
    tension[1] = T * sin(theta)*sin(phi);
    tension[2] = T * cos(theta);

    double N = m_block*g - tension[2];
    friction = -coef_friction*fabs(N)*vx_block/fabs(vx_block);
}


void Kite3d::update_state(){

    // Block acc
    ax_block = ( tension[0] + friction ) / m_block;
    // Block vel
    vx_block = vx_block + h*ax_block;
    // Block pos
    x_block = x_block + h*vx_block;

    // Kite acc
    acc_kite[0] = (f_aer[0] - tension[0])/m_kite;
    acc_kite[1] = (f_aer[1] - tension[1])/m_kite;
    acc_kite[2] = (f_aer[2] - tension[2] - m_kite*g)/m_kite;
    // Kite vel
    vel_kite[0] = vel_kite[0] + h*acc_kite[0];
    vel_kite[1] = vel_kite[1] + h*acc_kite[1];
    vel_kite[2] = vel_kite[2] + h*acc_kite[2];
    // Kite pos
    pos_kite[0] = pos_kite[0] + h*vel_kite[0];
    pos_kite[1] = pos_kite[1] + h*vel_kite[1];
    pos_kite[2] = pos_kite[2] + h*vel_kite[2];

    // imposing the rigid thread constraint
    double r_diff_modulo = sqrt(x_diff*x_diff + pos_kite[1]*pos_kite[1] + pos_kite[2]*pos_kite[2]);
    pos_kite[0] = x_block + (pos_kite[0] - x_block)/r_diff_modulo*R;
    pos_kite[1] = pos_kite[1]/r_diff_modulo*R;
    pos_kite[2] = pos_kite[2]/r_diff_modulo*R;
    //std::cout << pos_kite[0] << " " << pos_kite[1] << " " << acc_kite[0] << " " << acc_kite[1] << " " << alpha_ind << "\n";
}


double Kite3d::get_rew(int steps_from_training) {
    return vx_block*h*(steps_from_training+1);
}


double Kite3d::terminal_reward(double gamma){
    if (pos_kite[2] <= fall_limit)
        return -fall_penalty;
    else
        return 0;
}


Kite3d_vrel_old::Kite3d_vrel_old(const param& params, Wind3d* wind, std::mt19937& generator) :
Kite3d{params, wind, generator} {
    beta_bins = params.vecd.at("beta_bins");
}


const std::string Kite3d_vrel_old::descr() const {
    return "3d kite. Attack, bank and relative-velocity angles observed. Attack and bank angles controlled. " + wind->descr();
}


const vecs Kite3d_vrel_old::aggr_state_descr() const {
    vecs m_aggr_state_descr = vecs(0);
    for (int a=0; a<n_alphas(); a++) 
        for (int p=0; p<n_banks(); p++) 
            for (int b=0; b<n_betas(); b++){
                double beta = (beta_bins[b+1] + beta_bins[b]) / 2.0;
                m_aggr_state_descr.push_back("attack_ang_"+std::to_string(alphas[a])+",bank_angle_"+std::to_string(bank[p])+",vrel_angle_"+std::to_string(beta));
            }
    return m_aggr_state_descr;
}


int Kite3d_vrel_old::aggr_state() const {
    int b;
    for (b=0; b<n_betas(); b++){
        if (beta >= beta_bins[b] && beta < beta_bins[b+1]){
            break;
        }
    }
    return b + n_betas()*(bank_ind + n_banks()*alpha_ind);
}


int Kite3d_vrel::reset_kite() {
    Kite3d_vrel_old::reset_kite();
    beta = atan2(va[2], sqrt(va[0]*va[0] + va[1]*va[1]));
    return aggr_state();
}


void Kite3d_vrel::compute_F_aer() {
    Kite3d_vrel_old::compute_F_aer();
    beta = atan2(va[2], sqrt(va[0]*va[0] + va[1]*va[1]));
}