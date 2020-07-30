#include "../lib/nac.h"
#include "../lib/envs/kite.h"


// Training launcher. It must be launched giving also the environment_name and the trial_name.
// The first refers to the environment that is trained, the second to the algorithm that trains.



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
    NAC_AP alg(env, alg_params, generator);
    std::cout << "Algorithm successfully initialized\n";

    // Running the algorithm
    Timer timer;
    std::cout << "Algorithm started\n";
    alg.run(alg_params["n_steps"], alg_params["traj_points"], alg_params["init_values"]);
    std::cout << "Algorithm completed in " << timer.elapsed() << " seconds\n";

    // Printing the trajectories
    alg.print_traj(data_dir + env_name + "/" + alg_name + "/");
    alg.print_best_pol(data_dir + env_name + "/" + alg_name + "/");
    std::cout << "Trajectories successfully printed\n";

    return 0;
}