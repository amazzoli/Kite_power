#include "../lib/envs/kite.h"
#include "../lib/eval.h"


// Evaluate a policy. It must be launched giving the environment_name and the trial_name.


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
    dictd params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_env.txt"); // Def in utils

    // Constructing the environment.
    Environment* env = get_env(env_name, params, generator); // Def in kite.h
    std::cout << "Environment successfully built\n";

    // Importing the policy
    vec2d policy = read_best_pol(data_dir + env_name + "/" + alg_name + "/best_policy.txt");
    std::cout << "Policy imported\n";

    // Running the evaluation class
    dictd ev_params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_ev.txt");
    Eval eval(env, policy, ev_params, generator);
    int n_steps = int(ev_params["ev_time"]/params["decision_time"]);
    Timer timer;
    std::cout << "Evaluation started\n";
    eval.run(n_steps, ev_params["traj_points"]);
    std::cout << "Evaluation completed in " << timer.elapsed() << " seconds\n";

    eval.print_traj(data_dir + env_name + "/" + alg_name + "/");

    return 0;
}