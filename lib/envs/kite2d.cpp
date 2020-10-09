#include "kite.h"


// KITE 2D HAVING THE ATTACK ANGLE AS AGGREGATE STATE


/* Constructor */
Kite2d::Kite2d(const param& params, Wind2d* wind, std::mt19937& generator) : 
Kite{params, generator}, wind{wind} {

    m_state = vecd(state_descr().size());
    init_theta = params.d.at("init_theta");
    init_dtheta = params.d.at("init_dtheta");
    init_alpha_ind = params.d.at("init_alpha");
}


const std::string Kite2d::descr() const {
    return "2d kite. Attack angle observed. Attack angle controlled. " + wind->descr();
}


const vecs Kite2d::state_descr() const {
    vecs m_state_descr = {
        "kite_pos_x",   // 0
        "kite_pos_y",   // 1
        "kite_vel_x",   // 2
        "kite_vel_y",   // 3
        "kite_acc_x",   // 4 
        "kite_acc_y",   // 5
        "block_pos_x",  // 6
        "block_pos_y",  // 7
        "block_vel_x",  // 8
        "block_vel_y",  // 9
        "block_acc_x",  // 10 
        "block_acc_y",  // 11
    };
    return m_state_descr;
}


const vecs Kite2d::aggr_state_descr() const {
    vecs m_aggr_state_descr = vecs(0);
    for (int a=0; a<n_alphas(); a++) 
        m_aggr_state_descr.push_back("attack_ang_"+std::to_string(alphas[a]));
    return m_aggr_state_descr;
}


const vecs Kite2d::action_descr() const {
    vecs m_act_descr = { "attack_ang_decr", "attack_ang_stay", "attack_ang_incr" };
    return m_act_descr;
}


/* Aggregate state: it is defined by the current attack angle */
int Kite2d::aggr_state() const {
    return alpha_ind;
}


/* Initial configuration given the initial theta and dtheta*/
int Kite2d::reset_kite(){

    // Block position, velocity, acceleration
    m_state[6] = 0;
    m_state[7] = 0;
    m_state[8] = 0;
    m_state[9] = 0;
    m_state[10] = 0;
    m_state[11] = 0;
    // Kite position
    m_state[0] = m_state[6] + R*cos(init_theta);
    m_state[1] = m_state[7] + R*sin(init_theta);
    // Kite velocity
    m_state[2] = -R*init_dtheta*sin(init_theta);
    m_state[3] = R*init_dtheta*cos(init_theta);
    // Kite acceleration
    m_state[4] = -R*init_dtheta*init_dtheta*cos(init_theta);
    m_state[5] = -R*init_dtheta*init_dtheta*sin(init_theta);

    r_diff[0] = m_state[0] - m_state[6];
    r_diff[1] = m_state[1] - m_state[7];
    double* v_wind = (*wind).velocity(m_state[0], m_state[1]);
    double va_x = m_state[2] - v_wind[0];
    double va_y = m_state[3] - v_wind[1];
    beta = atan2(va_y, va_x);
    theta = atan2(r_diff[1], r_diff[0]);
    t2 = 1;
    if (beta > theta) t2 = 1;

    return aggr_state();
}


void Kite2d::impose_action(int a){
    if (a == 0) alpha_ind = std::max(alpha_ind - 1, 0);
    else if (a == 1) alpha_ind = alpha_ind;
    else if (a == 2) alpha_ind = std::min(alpha_ind + 1, n_alphas()-1);
    else throw std::runtime_error ( "Invalid action " + std::to_string(a) + "\n" );
}


bool Kite2d::integrate_trajectory() {

    // Angle with the ground
    r_diff[0] = m_state[0] - m_state[6];
    r_diff[1] = m_state[1] - m_state[7];
    theta = atan2(r_diff[1], r_diff[0]);

    // Aeroynamical forces
    compute_F_aer();

    // Tension and friction
    if ( fabs(m_state[8]) < v_threshold ) // BLOCK NOT MOVING
        compute_tension_still();
    else // BLOCK MOVING
        compute_tension_move();

    // Update positions, velocities and accelerations
    update_state();
    
    // Check if a terminal state is reached. Kite fallen
    if (m_state[1] <= 0) {
        return true;
    }
    return false;
}


/* Compute the aerodynamical forces */
void Kite2d::compute_F_aer(){
    // Apparent velocity
    double* v_wind = (*wind).velocity(m_state[0], m_state[1]);
    double va_x = m_state[2] - v_wind[0];
    double va_y = m_state[3] - v_wind[1];
    double va_mod = sqrt(va_x*va_x + va_y*va_y);
    beta = atan2(va_y, va_x);

    // Drag
    double coef = 0.5 * rho * a_kite * va_mod;
    double aux_d = coef * CD_alpha[alpha_ind];
    double D_x = -aux_d * va_x;
    double D_y = -aux_d * va_y;   

    // Lift
    double aux_l = coef * CL_alpha[alpha_ind];
    double L_x = aux_l * va_y * t2;
    double L_y = -aux_l * va_x * t2;

    f_aer[0] = L_x + D_x;
    f_aer[1] = L_y + D_y;
}


/* Compute the tension and the friction of a still block */
void Kite2d::compute_tension_still(){

    double v_diff[2] = {m_state[2] - m_state[8], m_state[3] - m_state[9]};
    double aux_A = (f_aer[0]*r_diff[0] + f_aer[1]*r_diff[1])/m_kite + (v_diff[0]*v_diff[0] + v_diff[1]*v_diff[1]);
    double aux_B1 = r_diff[1] - coef_friction*cos(theta)/fabs(cos(theta))*r_diff[0];
    double aux_B2 = r_diff[1] + coef_friction*cos(theta)/fabs(cos(theta))*r_diff[0];
    double aux_C = R*(m_kite + m_block)/(m_kite*m_block);
    // |Mg| > |Tz|
    double T1 =  (aux_A - g*aux_B1) / (aux_C - sin(theta)/m_block*aux_B1);
    // |Mg| < |Tz|
    double T2 = (aux_A - g*aux_B2) / (aux_C - sin(theta)/m_block*aux_B2);

    double T;
    if ( m_block*g > T1*sin(theta) ) T = T1;
    else if ( m_block*g <= T2*sin(theta) ) T = T2;
    else  throw std::runtime_error ( "Tension error for still block\n" );
    tension[0] = T * cos(theta);
    tension[1] = T * sin(theta);
    double N = m_block*g - tension[1];
    friction = -coef_friction*fabs(N)*cos(theta)/fabs(cos(theta));

    // If the computed tension is less than friction force: F_friction = -Tension[0]
    if ( fabs(tension[0]) < fabs(friction) ){
        double T = (aux_A - g*r_diff[1]) / (aux_C - sin(theta)*r_diff[1]/m_block - cos(theta)*r_diff[0]/m_block);
        tension[0] = T*cos(theta);
        tension[1] = T*sin(theta);
        friction = -tension[0];
    }   
}


/* Compute the tension and the friction of a moving block */
void Kite2d::compute_tension_move(){

    double v_diff[2] = {m_state[2] - m_state[8], m_state[3] - m_state[9]};
    double aux_A = (f_aer[0]*r_diff[0] + f_aer[1]*r_diff[1])/m_kite + (v_diff[0]*v_diff[0] + v_diff[1]*v_diff[1]);
    double aux_B1 = r_diff[1] - coef_friction*r_diff[0]*m_state[8]/fabs(m_state[8]);
    double aux_B2 = r_diff[1] + coef_friction*r_diff[0]*m_state[8]/fabs(m_state[8]);
    double aux_C = R*(m_kite + m_block)/(m_kite*m_block);
    // |Mg| > |Tz|
    double T1 = (aux_A - g*aux_B1) / (aux_C - sin(theta)/m_block*aux_B1);
    // |Mg| < |Tz|
    double T2 = (aux_A - g*aux_B2) / (aux_C - sin(theta)/m_block*aux_B2);

    double T;
    if ( m_block*g > T1*sin(theta) ) T = T1;
    else if ( m_block*g <= T2*sin(theta) ) T = T2;
    else  throw std::runtime_error ( "Tension error for moving block\n" );

    tension[0] = T*cos(theta);
    tension[1] = T*sin(theta);
    double N = m_block*g - T*sin(theta);
    friction = -coef_friction*fabs(N)*m_state[8]/fabs(m_state[8]);
}


void Kite2d::update_state(){
    // Block acc
    m_state[10] = ( tension[0] + friction ) / m_block;
    m_state[11] = 0;
    // Block vel
    m_state[8] = m_state[8] + h*m_state[10]; 
    m_state[9] = m_state[9] + h*m_state[11];
    // Block pos
    m_state[6] = m_state[6] + h*m_state[8]; 
    m_state[7] = m_state[7] + h*m_state[9];
    // Kite acc
    m_state[4] = (f_aer[0] - tension[0])/m_kite;
    m_state[5] = (f_aer[1] - tension[1] - m_kite*g)/m_kite;
    // Kite vel
    m_state[2] = m_state[2] + h*m_state[4];
    m_state[3] = m_state[3] + h*m_state[5];
    // Kite pos
    m_state[0] = m_state[0] + h*m_state[2];
    m_state[1] = m_state[1] + h*m_state[3];

    // imposing the rigid thread constraint
    double r_diff_modulo = sqrt(r_diff[0]*r_diff[0] + r_diff[1]*r_diff[1]);
    m_state[0] = m_state[6] + (m_state[0] - m_state[6])/fabs(r_diff_modulo)*R;
    m_state[1] = m_state[7] + (m_state[1] - m_state[7])/fabs(r_diff_modulo)*R;
    //std::cout << m_state[0] << " " << m_state[1] << " " << m_state[6] << " " << m_state[7] << " " << curr_alpha_ind << "\n";
}


double Kite2d::get_rew(int steps_from_training) {
    return m_state[8]*h*(steps_from_training+1);
}


double Kite2d::terminal_reward(double gamma){
    if (m_state[1] <= 0)
        return -fall_penalty;
    else
        return 0;
}


// KITE 2D HAVING ALSO THE RELATIVE VELOCITY ANGLE AS STATE


Kite2d_vrel::Kite2d_vrel(const param& params, Wind2d* wind, std::mt19937& generator) :
Kite2d{params, wind, generator} {

    beta_bins = params.vecd.at("beta_bins");
}


const std::string Kite2d_vrel::descr() const {
    return "2d kite. Attack angle and relative-velocity angle observed. Attack angle controlled. " + wind->descr();
}

const vecs Kite2d_vrel::aggr_state_descr() const {
    vecs m_aggr_state_descr = vecs(0);
    for (int a=0; a<n_alphas(); a++) 
        for (int b=0; b<n_betas(); b++) {
            double ref_beta = (beta_bins[b+1] + beta_bins[b]) / 2.0;
            m_aggr_state_descr.push_back("attack_ang_"+std::to_string(alphas[a])+",vrel_angle_"+std::to_string(ref_beta));
        }
    return m_aggr_state_descr;
}

int Kite2d_vrel::aggr_state() const {
    int b;
    for (b=0; b<n_betas(); b++){
        if (beta >= beta_bins[b] && beta < beta_bins[b+1]){
            break;
        }
    }

    return b + n_betas()*alpha_ind;
}