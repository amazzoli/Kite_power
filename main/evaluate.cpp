#include "../lib/envs/kite.h"
#include "../lib/eval.h"


// Evaluate a policy. It must be launched giving the environment_name and the trial_name.

Environment* get_env(std::string env_name, const param& params, std::mt19937& generator);


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

    param env_params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_env.txt"); // Def in utils
    param ev_params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_ev.txt");

    // Constructing the environment.
    env_params.d.at("ep_length") = ev_params.d.at("ep_length");
    Environment* env = get_env(env_name, env_params, generator); // Def in kite.h
    std::cout << "Environment successfully built\n";

    // Importing the policy
    vec2d policy = read_best_pol(data_dir + env_name + "/" + alg_name + "/best_policy.txt");
    std::cout << "Policy imported\n";

    // Running the evaluation class
    Eval eval(env, policy, ev_params.d, generator);
    Timer timer;
    std::cout << "Evaluation started\n";
    ev_params.d["decision_time"] = env_params.d["decision_time"];
    eval.run(ev_params);
    std::cout << "Evaluation completed in " << timer.elapsed() << " seconds\n";

    eval.print_output(data_dir + env_name + "/" + alg_name + "/");

    delete env;

    return 0;
}


Environment* get_env(std::string env_name, const param& params, std::mt19937& generator) {
    if (env_name == "kite2d"){
		Wind2d* wind = get_wind2d(params);
        Environment* env = new Kite2d(params, wind, generator);
        return env;
    }
    if (env_name == "kite2d_vrel"){
		Wind2d* wind = get_wind2d(params);
        Environment* env = new Kite2d_vrel(params, wind, generator);
        return env;
    }
	if (env_name == "kite3d"){
		Wind3d* wind = get_wind3d(params);
        Environment* env = new Kite3d(params, wind, generator);
        return env;
    }
	if (env_name == "kite3d_vrel"){
		Wind3d* wind = get_wind3d(params);
        Environment* env = new Kite3d_vrel(params, wind, generator);
        return env;
    }
    else throw std::invalid_argument( "Invalid environment name" );
}