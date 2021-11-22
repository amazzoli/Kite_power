# Kite_power
Reinforcement learning algorithms for harvesting wind energy

## Structure
Training and evaluations can be launched from the notebook scripts in the *main* folder.
Typically the notebooks generate the parameter files, compile and run the c++ scripts and analyse the results.
For long algorithms it is suggested to launch the c++ exe them from terminal. Differently from the notebook, 
this allows for on-line information during their execution.

Each analisis has typically one global name, the *system_name* or the *environment_name* which refers to a specific physical system, 
e.g. **2dkite**: 2d kite having the attack angles as states, and a more specific name related to the algorithm used or specific system parameter changed, 
e.g. **nac** where the natural actor critic algorithm is used.
The two names define also the directory structure in *data*, where the parameters and the output information is printed, and in *plots*, 
where the plots of the analysis are saved.

g++ -o norail.exe run_norail.cpp ../lib/alg.cpp ../lib/qalg.cpp ../lib/eval.cpp ../lib/utils.cpp ../lib/wind.cpp ../lib/envs/kite_norail.cpp ../lib/envs/kite2d.cpp ../lib/envs/kite3d_norail.cpp -std=c++11
