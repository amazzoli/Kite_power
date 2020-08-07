#include "../lib/nac.h"
#include "../lib/envs/kite.h"


// Training launcher. It must be launched giving also the environment_name and the trial_name.
// The first refers to the environment that is trained, the second to the algorithm that trains.


void run(AC& alg, const dictd& params, std::string dir);


int main(int argc, char** argv) {

    if (argc != 3)
        throw std::runtime_error("Two strings must be passed during execution: environment name and run name");

    // Init random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    // Importing the parameters file
    std::string env_name(argv[1]);
    std::string alg_name(argv[2]);
    std::string data_dir = "../data/";
    dictd params = parse_param_file(data_dir+env_name+"/"+alg_name+"/param_env.txt"); // Def in utils

    // Constructing the environment.
    Environment* env = get_env(env_name, params, generator); // Def in kite.h
    std::cout << "Environment successfully built\n";

    // Constructing the algorithm
    dictd alg_params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_alg.txt"); 
    AC alg(env, alg_params, generator);
    std::cout << "Algorithm successfully initialized\n";

    // Running the algorithm
    Timer timer;
    std::cout << "Algorithm started\n";
    run(alg, alg_params, data_dir + env_name + "/" + alg_name);
    std::cout << "Algorithm completed in " << timer.elapsed() << " seconds\n";

    // Printing the trajectories
    alg.print_traj(data_dir + env_name + "/" + alg_name + "/");
    alg.print_best_pol_val(data_dir + env_name + "/" + alg_name + "/");
    std::cout << "Trajectories successfully printed\n";

    delete env;
    return 0;
}


void run(AC& alg, const dictd& params, std::string dir){
    if (params.at("use_init_val_from_file") <= 0 && params.at("use_init_pol_from_file") <= 0){
        // Without priors -> constant init values and flat policy
        alg.run(params.at("n_steps"), params.at("init_values"), params.at("traj_points"));
    } else if (params.at("use_init_val_from_file") > 0 && params.at("use_init_pol_from_file") <= 0){
        // With value prior
        vecd val = read_best_val(dir + "/best_value.txt");
        alg.run(params.at("n_steps"), val, params.at("traj_points"));
    } else if (params.at("use_init_val_from_file") <= 0 && params.at("use_init_pol_from_file") > 0){
        // With policy prior
        vec2d policy = read_best_pol(dir + "/best_policy.txt");
        alg.run(params.at("n_steps"), params.at("init_values"), policy, params.at("traj_points"));
    } else {
        // With policy and value priors
        vecd val = read_best_val(dir + "/best_value.txt");
        vec2d policy = read_best_pol(dir+  "/best_policy.txt");
        alg.run(params.at("n_steps"), val, policy, params.at("traj_points"));
    }
}