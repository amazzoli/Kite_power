#include "../lib/wind.h"



int main(int argc, char** argv) {

    if (argc != 3)
        throw std::runtime_error("Two strings must be passed during execution: environment name and run name");

    // Init random generator
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);

    // Importing the parameters file
    std::string env_name(argv[1]), alg_name(argv[2]), data_dir = "../data/";
    param params = parse_param_file(data_dir+env_name+"/"+alg_name+"/param_env.txt"); // Def in utils

    Wind3d* wind = get_wind3d(params);

    double q[3] = {0, 50, 50};
    double* v_wind;
    double dt = 0.001;
    for (int t=0; t<1000; t++){
    	v_wind = (*wind).velocity(q[0], q[1], q[2]);
    	q[0] += v_wind[0]*dt;
    	q[1] += v_wind[1]*dt;
    	q[2] += v_wind[2]*dt;
    	std::cout << q[0] << " " << q[1] << " " << q[2] << "\n";
    }


    return 0;
}