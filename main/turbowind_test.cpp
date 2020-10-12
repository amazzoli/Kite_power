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

    Wind3d* wind = get_wind3d(params, generator);

    double q[3] = {100, 49, 30};
    double* v_wind;
    double dt = 0.001;
    double v=0;
    Timer timer;
    for(int r=0; r<1000; r++) {
        v_wind = (*wind).init(q[0], q[1], q[2]);

        for (int t=0; t<10000; t++){
        	v_wind = (*wind).velocity(q[0], q[1], q[2]);
        	q[0] += v_wind[0]*dt;
        	q[1] += (v_wind[1])*dt;
        	q[2] += (v_wind[2])*dt;
            //std::cout << q[2] << " " << v_wind[2] << " " << v_wind[2]-v << "\n";
            v = v_wind[2];
        }
    }
    //std::cout << timer.elapsed() << "\n";


    return 0;
}