#include "kite.h"


// KITE 2D HAVING THE ATTACK ANGLE AS AGGREGATE STATE


/* Constructor */
Kite3d::Kite3d(const param& params, Wind3d* wind, std::mt19937& generator) : 
Kite{params, generator}, wind{wind} {

	// BUILDING THE STATE AND ACTION INFORMATION
	// Description of each position of the full state vector
	m_state_descr = {
		"kite_pos_x",	// 0
		"kite_pos_y",	// 1
        "kite_pos_z",	// 2
		"kite_vel_x",	// 3
		"kite_vel_y",	// 4
        "kite_vel_z",	// 5
		"kite_acc_x",	// 6 
		"kite_acc_y",	// 7
        "kite_acc_z",	// 8
		"block_pos_x",	// 9
		"block_vel_x",	// 10
		"block_acc_x",	// 11 
	};
    m_state = vecd(m_state_descr.size());
    bank = params.vecd.at("banks");

	// Description of each position of the aggregate state vector
	m_aggr_state_descr = vecs(0);
	for (int a=0; a<n_alphas(); a++) 
        for (int b=0; b<n_banks(); b++) 
		    m_aggr_state_descr.push_back("attack_ang_"+std::to_string(alphas[a])+",bank_angle_"+std::to_string(bank[b]));
	m_n_aggr_state = m_aggr_state_descr.size();

	// Description of each position of the actions
	m_act_descr = { "a_decr_b_decr", "a_decr_b_stay", "a_decr_b_incr", \
                    "a_stay_b_decr", "a_stay_b_stay", "a_stay_b_incr", \
                    "a_incr_b_decr", "a_incr_b_stay", "a_incr_b_incr" };
	m_n_actions = m_act_descr.size();

	// SETTING SPECIFIC PARAMETERS
	init_theta = params.d.at("init_theta");
	init_dtheta = params.d.at("init_dtheta");
    init_alpha_ind = params.d.at("init_alpha");
	init_phi = params.d.at("init_phi");
	init_dphi = params.d.at("init_dphi");
    init_bank_ind = params.d.at("init_bank");
}


/* Aggregate state: it is defined by the current attack angle and bank angle */
int Kite3d::aggr_state() const {
	return bank_ind + n_banks() * alpha_ind;
}


/* Initial configuration given the initial theta and dtheta*/
void Kite3d::reset_kite(){

    // Initial bank angle (attack angle set in base class)
    bank_ind = init_bank_ind;

	// Block position, velocity, acceleration
    m_state[9] = 0;
    m_state[10] = 0;
    m_state[11] = 0;
    // Kite position
    m_state[0] = m_state[9] + R*sin(init_theta)*cos(init_phi);
    m_state[1] = R*sin(init_theta)*sin(init_phi);
    m_state[2] = R*cos(init_theta);
	// Kite velocity
    m_state[3] = m_state[10] + R*cos(init_theta)*cos(init_phi)*init_dtheta - R*sin(init_theta)*sin(init_phi)*init_dphi;
    m_state[4] = R*cos(init_theta)*sin(init_phi)*init_dtheta + R*sin(init_theta)*cos(init_phi)*init_dphi;
    m_state[5] = -R*sin(init_theta)*init_dtheta;
    // Kite acceleration
    m_state[6] = R*(m_state[11]/R -sin(init_theta)*cos(init_phi)*(init_dtheta*init_dtheta + init_dphi*init_dphi) - \
                 2*cos(init_theta)*sin(init_phi)*init_dtheta*init_dphi);
    m_state[7] = R*(sin(init_theta)*sin(init_phi)*(init_dtheta*init_dtheta + init_dphi*init_dphi) + \
                 2*cos(init_theta)*cos(init_phi)*init_dtheta*init_dphi);
    m_state[8] = -R*cos(init_theta)*init_dtheta*init_dtheta;

    x_diff = m_state[0] - m_state[9];
    double* v_wind = (*wind).velocity(m_state[0], m_state[1], m_state[2]);
    double va[3] = { m_state[3] - v_wind[0], m_state[4] - v_wind[1], m_state[5] - v_wind[2] };
    beta = atan2(va[2], va[0]);
    theta = atan2(sqrt(x_diff*x_diff + m_state[1]*m_state[1]), m_state[2]);
    phi = atan2(m_state[1], x_diff);
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

    x_diff = m_state[0] - m_state[9];
    theta = atan2(sqrt(x_diff*x_diff + m_state[1]*m_state[1]), m_state[2]);
    phi = atan2(m_state[1], x_diff);

    // Aeroynamical forces
    compute_F_aer();

    // Tension and friction
    if ( fabs(m_state[10]) < v_threshold ) // BLOCK NOT MOVING
        compute_tension_still();
    else // BLOCK MOVING
        compute_tension_move();

    // Update positions, velocities and accelerations
    update_state();

    // Check if a terminal state is reached. Kite fallen
    if (m_state[2] <= 0) return true;
    return false;
}


/* Compute the aerodynamical forces */
void Kite3d::compute_F_aer(){
    // Apparent velocity
    double* v_wind = (*wind).velocity(m_state[0], m_state[1], m_state[2]);
    double va[3] = { m_state[3] - v_wind[0], m_state[4] - v_wind[1], m_state[5] - v_wind[2] };
    double va_mod = sqrt(va[0]*va[0] + va[1]*va[1] + va[2]*va[2]);
    beta = atan2(va[2], va[0]);

    double t1[3] = {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)}; 
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

    double vx_diff = m_state[3] - m_state[10];
    double aux_A = (f_aer[0]*x_diff + f_aer[1]*m_state[1] + f_aer[2]*m_state[2])/m_kite + (vx_diff*vx_diff + m_state[4]*m_state[4] + m_state[5]*m_state[5]);
    double aux_B = coef_friction*(cos(phi)*x_diff + sin(phi)*m_state[1]);
    double aux_C = R*(m_kite + m_block)/(m_kite*m_block);
    // |Mg| > |Tz|
    double T1 = (aux_A - g*(m_state[2] - aux_B)) / (aux_C - cos(theta)/m_block*(m_state[2] - aux_B));
    // |Mg| < |Tz|
    double T2 = (aux_A - g*(m_state[2] + aux_B)) / (aux_C - cos(theta)/m_block*(m_state[2] + aux_B));

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
        double T = (aux_A - g*m_state[2]) / (aux_C - sin(theta)/m_block*(cos(phi)*x_diff + sin(phi)*m_state[1]) - cos(theta)/m_block*m_state[2]);
        tension[0] = T * sin(theta)*cos(phi);
        tension[1] = T * sin(theta)*sin(phi);
        tension[2] = T * cos(theta);
        friction = -tension[0];
    }   
}


/* Compute the tension and the friction of a moving block */
void Kite3d::compute_tension_move(){

    double vx_diff = m_state[3] - m_state[10];
    double aux_A = (f_aer[0]*x_diff + f_aer[1]*m_state[1] + f_aer[2]*m_state[2])/m_kite + (vx_diff*vx_diff + m_state[4]*m_state[4] + m_state[5]*m_state[5]);
    double aux_B = coef_friction/fabs(m_state[10])*x_diff*m_state[10];
    double aux_C = R*(m_kite + m_block)/(m_kite*m_block);
    // |Mg| > |Tz|
    double T1 = (aux_A - g*(m_state[2] - aux_B)) / (aux_C - cos(theta)/m_block*(m_state[2] - aux_B));
    // |Mg| < |Tz|
    double T2 = (aux_A - g*(m_state[2] + aux_B)) / (aux_C - cos(theta)/m_block*(m_state[2] + aux_B));

    double T;
    if ( m_block*g > T1*cos(theta) ) T = T1;
    else if ( m_block*g <= T2*cos(theta) ) T = T2;
    else  throw std::runtime_error ( "Tension error for moving block\n" );
    tension[0] = T * sin(theta)*cos(phi);
    tension[1] = T * sin(theta)*sin(phi);
    tension[2] = T * cos(theta);

    double N = m_block*g - tension[2];
    friction = -coef_friction*fabs(N)*m_state[10]/fabs(m_state[10]);
}


void Kite3d::update_state(){

    // Block acc
    m_state[11] = ( tension[0] + friction ) / m_block;
    // Block vel
    m_state[10] = m_state[10] + h*m_state[11];
    // Block pos
    m_state[9] = m_state[9] + h*m_state[10];

    // Kite acc
    m_state[6] = (f_aer[0] - tension[0])/m_kite;
    m_state[7] = (f_aer[1] - tension[1])/m_kite;
    m_state[8] = (f_aer[2] - tension[2] - m_kite*g)/m_kite;
    // Kite vel
    m_state[3] = m_state[3] + h*m_state[6];
    m_state[4] = m_state[4] + h*m_state[7];
    m_state[5] = m_state[5] + h*m_state[8];
    // Kite pos
    m_state[0] = m_state[0] + h*m_state[3];
    m_state[1] = m_state[1] + h*m_state[4];
    m_state[2] = m_state[2] + h*m_state[5];

    // imposing the rigid thread constraint
    double r_diff_modulo = sqrt(x_diff*x_diff + m_state[1]*m_state[1] + m_state[2]*m_state[2]);
    m_state[0] = m_state[9] + (m_state[0] - m_state[9])/r_diff_modulo*R;
    m_state[1] = m_state[1]/r_diff_modulo*R;
    m_state[2] = m_state[2]/r_diff_modulo*R;
    //std::cout << m_state[0] << " " << m_state[1] << " " << m_state[6] << " " << m_state[7] << " " << curr_alpha_ind << "\n";
}


double Kite3d::get_rew(int steps_from_training) {
    return m_state[10]*h*(steps_from_training+1);
}


double Kite3d::terminal_reward(double gamma){
    if (m_state[2] <= 0)
        return -fall_penalty;
    else
        return 0;
}


Kite3d_vrel::Kite3d_vrel(const param& params, Wind3d* wind, std::mt19937& generator) :
Kite3d{params, wind, generator} {
    beta_bins = params.vecd.at("beta_bins");
	m_aggr_state_descr = vecs(0);
	for (int a=0; a<n_alphas(); a++) 
        for (int p=0; p<n_banks(); p++) 
            for (int b=0; b<n_betas(); b++){
                double beta = (beta_bins[b+1] + beta_bins[b]) / 2.0;
		        m_aggr_state_descr.push_back("attack_ang_"+std::to_string(alphas[a])+",bank_angle_"+std::to_string(bank[p])+",vrel_angle_"+std::to_string(beta));
            }
	m_n_aggr_state = m_aggr_state_descr.size();
}


int Kite3d_vrel::aggr_state() const {
    int b;
    for (b=0; b<n_betas(); b++){
        if (beta >= beta_bins[b] && beta < beta_bins[b+1]){
            break;
        }
    }
    return b + n_betas()*(bank_ind + n_banks()*alpha_ind);
}