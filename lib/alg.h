#ifndef ALG_H
#define ALG_H

#include "utils.h"
#include "env.h"


/* Abstract class of an algorithm */
class Algorithm {

    protected:

        /* MDP to solve */
        Environment* env; 
        /* Random number generator */ 
        std::mt19937 generator;

    public:

        /* Construct the algorithm */
        Algorithm(Environment* env, std::mt19937& generator) : env{env}, generator{generator} {};

        /* Abstract. Run the algorithm given some parameters */
        virtual void run(const param& params) = 0;

        /* Abstract. Print the output of the algorithm in the given directory */
        virtual void print_output(std::string out_dir) const = 0;
};


#endif