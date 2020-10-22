#include "../lib/nac.h"
#include "../lib/qalg.h"
#include "../lib/eval.h"
#include "../lib/envs/kite.h"


// Algorithm launcher. It must be launched giving also the environment_name and the trial_name.
// The first refers to the environment that is trained, the second to the algorithm that is launched.


Environment* get_env(std::string env_name, const param& params, std::mt19937& generator);
RLAlgorithm* get_alg(Environment* env, const param& params, std::mt19937& generator);


int main(int argc, char** argv) {

    if (argc != 3)
        throw std::runtime_error("Two strings must be passed during execution: environment name and run name");

    // Init random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    // Importing the parameters file
    std::string env_name(argv[1]), alg_name(argv[2]), data_dir = "../data/";
    param params_env = parse_param_file(data_dir+env_name+"/"+alg_name+"/param_env.txt"); // Def in utils

    // Constructing the environment.
    Environment* env = get_env(env_name, params_env, generator); // Def in kite.h
    std::cout << "Environment successfully built:\n" << (*env).descr() << "\n\n";

    // // Constructing the algorithm
    param alg_params = parse_param_file(data_dir + env_name + "/" + alg_name + "/param_alg.txt");
    RLAlgorithm* alg = get_alg(env, alg_params, generator);
    std::cout << "Algorithm successfully initialized:\n" << (*alg).descr() << "\n\n";

    // Running the algorithm
    Timer timer;
    std::cout << "Algorithm started\n";
    (*alg).run(alg_params);
    std::cout << "Algorithm completed in " << timer.elapsed() << " seconds\n";

    // Printing the trajectories
    (*alg).print_output(data_dir + env_name + "/" + alg_name + "/");
    std::cout << "Trajectories successfully printed\n";

    delete env;
    delete alg;

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
		Wind3d* wind = get_wind3d(params, generator);
        Environment* env = new Kite3d(params, wind, generator);
        return env;
    }
	if (env_name == "kite3d_vrel" || env_name == "kite3d_lin" || env_name == "kite3d_log"){
		Wind3d* wind = get_wind3d(params, generator);
        Environment* env = new Kite3d_vrel_old(params, wind, generator);
        return env;
    }
    if (env_name == "kite3d_turboframe" || env_name == "kite3d_lognoise" || env_name == "kite3d_couetteframe" ||
        env_name == "kite3d_turbo" ){
        Wind3d* wind = get_wind3d(params, generator);
        Environment* env = new Kite3d_vrel(params, wind, generator);
        return env;
    }
    else throw std::invalid_argument( "Invalid environment name" );
}


RLAlgorithm* get_alg(Environment* env, const param& params, std::mt19937& generator) {

    std::string alg_name = params.s.at("alg_type");

    if (alg_name == "ac"){
        return new AC(env, params, generator);
    }
    else if (alg_name == "nac"){
		return new NAC_AP(env, params, generator);
    }
    else if (alg_name == "sarsa"){
		return new SARSA_eps(env, params, generator);
    }
    else if (alg_name == "ql"){
		return new QL_eps(env, params, generator);
    }
    else if (alg_name == "eval"){
		return new Eval(env, params, generator);
    }
    else throw std::invalid_argument( "Invalid environment name" );
}
