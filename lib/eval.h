#ifndef EVAL_H
#define EVAL_H

#include "env.h"
#include "alg.h"


/* Evaluation algorithm */
class Eval : public RLAlgorithm {

    private:

        /* Policy to evaluate */
        vec2d m_quality;
        /* Trajectory of all the states */
        vec2d m_state_traj;
        /* Aggregate states trajectory */
        veci m_aggr_st_traj;
        /* Uniform random distribution */ 
        std::uniform_real_distribution<double> unif_dist;
        /* Uniform distribution over the actions */
        std::uniform_int_distribution<int> unif_act_dist;
        /* Epsilon greedy exploration. 0 if not specified */
        double eps;

        virtual void init(const param& params);
        virtual int get_action();
        virtual void learning_update();
        virtual void build_traj();
        virtual void print_traj(std::string out_dir) const;

    public:

        /* Construct the algorithm given the parameters dictionary */
        Eval(Environment* env, const param& params, std::mt19937& generator);

        /* Description */ 
        const std::string descr() const { return "Evaluation algorithm."; }
};


#endif