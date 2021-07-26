#ifndef LINE_GRAPH_HPP
#define LINE_GRAPH_HPP

#include <sstream>
#include <cstdarg>
#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <Eigen/Dense>
#include <digraph.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>

/* 
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     5/3/2021
 */
namespace boost {

namespace multiprecision {

template <typename T>
typename boost::math::tools::promote_arg<T>::type logsumexp(const T& x, const T& y)
{
    /*
     * Compute logsumexp(x, y). 
     */
    using std::exp; 
    using std::log1p;    // Boost versions will be called if T is a boost::multiprecision type  

    if (x >= y) return x + log1p(exp(y - x));
    else        return y + log1p(exp(x - y)); 
}

template <typename T>
typename boost::math::tools::promote_arg<T>::type logsumexp(const std::vector<T>& x)
{
    /*
     * Pseudo-variadic version of the above logsumexp() function.
     */
    using std::exp; 
    using std::log1p;    // Boost versions will be called if T is a boost::multiprecision type  

    T max = x[0];
    unsigned argmax = 0; 
    for (unsigned i = 1; i < x.size(); ++i)
    {
        if (max < x[i]) 
        {
            max = x[i];
            argmax = i;
        }
    }
    T total = 0;
    for (unsigned i = 0; i < x.size(); ++i)
    {
        if (i != argmax) total += exp(x[i] - max);
    }
    return max + log1p(total); 
}

}    // namespace multiprecision

}    // namespace boost 

using namespace Eigen;

template <typename T>
class LineGraph : public LabeledDigraph<T>
{
    /*
     * An implementation of the two-state grid graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        unsigned N;    // Length of the graph

        // Canonical ordering of nodes
        std::vector<Node*> order;

        // Vector of edge labels that grows with the length of the graph
        // Each array stores the labels for the edges i -> i+1 and i+1 -> i
        std::vector<std::array<T, 2> > line_labels;

    public:
        LineGraph() : LabeledDigraph<T>()
        {
            /*
             * Trivial constructor with length 0 (1 vertex named "0").
             */
            this->N = 0;
            Node* node = this->addNode("0");
            this->order.push_back(node);
        }

        LineGraph(unsigned N) : LabeledDigraph<T>()
        {
            /*
             * Trivial constructor with given length; set edge labels to unity.
             */
            // Add the zeroth node
            this->N = N;
            Node* node = this->addNode("0");
            this->order.push_back(node);

            for (unsigned i = 0; i < N; ++i)
            {
                // Add the (i+1)-th node
                std::stringstream ssi, ssj;
                ssi << i;
                ssj << i + 1;
                node = this->addNode(ssj.str());
                this->order.push_back(node);

                // Add edges i -> i+1 and i+1 -> i with labels = 1
                this->addEdge(ssi.str(), ssj.str());
                this->addEdge(ssj.str(), ssi.str());
                std::array<T, 2> labels = {1, 1};
                this->line_labels.push_back(labels);
            }
        }

        ~LineGraph()
        {
            /*
             * Trivial destructor.
             */
        }

        unsigned getN()
        {
            /* 
             * Return the length of the graph.
             */
            return this->N;
        }

        void addNodeToEnd(std::array<T, 2> labels)
        {
            /*
             * Add new node onto the end of the graph, with the two 
             * additional edges. 
             */
            // Add new node to end of graph 
            this->N++;
            std::stringstream ssi, ssj;
            ssi << this->N - 1;
            ssj << this->N;
            Node* node = this->addNode(ssj.str());
            this->order.push_back(node);

            // Add edges N-1 -> N and N -> N-1 (with incremented value for N)
            this->addEdge(ssi.str(), ssj.str(), labels[0]);
            this->addEdge(ssj.str(), ssi.str(), labels[1]);
            this->line_labels.push_back(labels);
        }

        void setLabels(unsigned i, std::array<T, 2> labels)
        {
            /*
             * Set the edge labels between the i-th and (i+1)-th nodes 
             * (i -> i+1, then i+1 -> i) to the given values.
             */
            this->line_labels[i] = labels;
            std::stringstream ssi, ssj;
            ssi << i;
            ssj << i + 1;
            this->setEdgeLabel(ssi.str(), ssj.str(), labels[0]);
            this->setEdgeLabel(ssj.str(), ssi.str(), labels[1]);
        }

        T computeUpperExitProb(T exit_rate_0 = 1, T exit_rate_N = 1)
        {
            /*
             * Compute the *log* probability of exiting the line graph through
             * the upper vertex N (Eqn. 4.1).
             *
             * This is done using log-semiring arithmetic (log-sum-exp for 
             * addition, addition for multiplication). 
             */
            using std::log; 

            // Start with log(1) = 0 ...
            std::vector<T> terms; 
            for (unsigned i = 0; i < this->N + 2; ++i) terms.push_back(0); 

            // ... then, for each vertex, get the ratio of the reverse 
            // edge label divided by the forward edge label 
            T curr = log(exit_rate_0) - log(this->line_labels[0][0]);
            terms[1] = curr; 
            for (unsigned i = 1; i < this->N; ++i)
            {
                curr += log(this->line_labels[i-1][1]) - log(this->line_labels[i][0]);
                terms[i + 1] = curr;
            }
            curr += log(this->line_labels[this->N - 1][1]) - log(exit_rate_N);
            terms[this->N + 1] = curr;

            // ... then take the log-sum-exp
            T logprob = boost::multiprecision::logsumexp<T>(terms); 

            // ... then take the reciprocal
            return -logprob;
        }

        T computeLowerExitRate(T exit_rate_0 = 1, T exit_rate_N = 1)
        {
            /*
             * Compute the *log* reciprocal of the mean first passage time
             * to exit through the lower vertex 0, in the case that lower
             * exit does occur.
             *
             * This is done using log-semiring arithmetic (log-sum-exp for 
             * addition, addition for multiplication). 
             */
            using std::log; 

            // Get the weights of 2-forests rooted at the two exit vertices
            // with path from i to lower exit, for i = N, ..., 0
            std::vector<T> log_two_forest_weights;
            for (unsigned i = 0; i <= this->N; ++i)
                log_two_forest_weights.push_back(0);
            
            // Start with the sole 2-forest with path i = N -> 0
            T log_weight = log(exit_rate_0);
            for (unsigned i = 0; i < this->N; ++i)
                log_weight += log(this->line_labels[i][1]);    // Edges i <- i+1 for i = 0, ..., N-1
            log_two_forest_weights[this->N] = log_weight;      // (1 -> 0, ..., N -> N-1)

            // Then run from i = N-1 to i = 0
            for (int i = this->N - 1; i >= 0; --i)
            {
                log_weight = log(exit_rate_0);
                for (int j = 0; j < i; ++j)
                    log_weight += log(this->line_labels[j][1]);    // Edges j <- j+1 for j = 0, ..., i-1
                for (int j = i; j < this->N - 1; ++j)    
                    log_weight += log(this->line_labels[j+1][0]);  // Edges j+1 -> j+2 for j = i, ..., N-2
                log_weight += log(exit_rate_N);
                log_two_forest_weights[i] = boost::multiprecision::logsumexp<T>(log_two_forest_weights[i+1], log_weight);
            }

            // Now get the weight of the last 2-forest, with a path from 
            // 0 to upper exit
            T log_upper_exit_weight = 0;
            for (unsigned i = 0; i < this->N; ++i)
                log_upper_exit_weight += log(this->line_labels[i][0]);
            log_upper_exit_weight += log(exit_rate_N);

            // Now get the weights of 3-forests rooted at the two exit
            // vertices and i, for i = 0, ..., N, with a path 0 -> i
            std::vector<T> log_three_forest_weights;
            for (unsigned i = 0; i <= this->N; ++i)
                log_three_forest_weights.push_back(0);

            // For each i = 0, ... N, start with the weight of the path 0 -> i
            for (unsigned i = 0; i <= this->N; ++i)
            {
                log_weight = 0;
                for (unsigned j = 0; j < i; ++j)
                    log_weight += log(this->line_labels[j][0]);

                // Then run through all the 3-forests with that path ...
                std::vector<T> log_factor_terms;
                for (unsigned j = 0; j < 1 + this->N - i; ++j)
                    log_factor_terms.push_back(0); 
                
                // Start with the last forest (with the path N -> i)
                T term = 0;
                for (int j = this->N; j > i; --j)
                    term += log(this->line_labels[j-1][1]);
                log_factor_terms[0] = term; 

                for (unsigned j = i + 1; j <= this->N; ++j)
                {
                    // Then run through all forests with path N -> upper exit
                    term = log(exit_rate_N);
                    for (unsigned k = i + 1; k < j; ++k)
                        term += log(this->line_labels[k-1][1]);
                    for (unsigned k = j; k < this->N; ++k)
                        term += log(this->line_labels[k][0]);
                    log_factor_terms[j-i] = term;
                }

                log_three_forest_weights[i] = log_weight + boost::multiprecision::logsumexp<T>(log_factor_terms);
            }

            // Finally, compute the reciprocal of the mean first passage time
            std::vector<T> log_denom_terms;
            for (unsigned i = 0; i < this->N + 1; ++i)
                log_denom_terms.push_back(0); 
            for (unsigned i = 0; i <= this->N; ++i)
                log_denom_terms[i] = log_three_forest_weights[i] + log_two_forest_weights[i];
            T log_denom = boost::multiprecision::logsumexp<T>(log_denom_terms); 

            return (
                log_two_forest_weights[0] +
                boost::multiprecision::logsumexp<T>(log_two_forest_weights[0], log_upper_exit_weight) - log_denom
            );
        }

        T computeUpperExitRate(T exit_rate_0 = 1, T exit_rate_N = 1)
        {
            /*
             * Compute the *log* reciprocal of the mean first passage time
             * to exit through the upper vertex N, in the case that upper
             * exit does occur.
             *
             * This is done using log-semiring arithmetic (log-sum-exp for 
             * addition, addition for multiplication). 
             */
            using std::log; 

            // Get the weights of 2-forests rooted at the two exit vertices
            // with path from i to upper exit, for i = 0, ..., N
            std::vector<T> log_two_forest_weights;
            for (unsigned i = 0; i <= this->N; ++i)
                log_two_forest_weights.push_back(0);
            
            // Start with i = 0
            T log_weight = 0;
            for (unsigned i = 0; i < this->N; ++i)
                log_weight += log(this->line_labels[i][0]);    // Edges i -> i+1 for i = 0, ..., N-1
            log_two_forest_weights[0] = log_weight;

            // Then run from 1 to N
            for (unsigned i = 1; i <= this->N; ++i)
            {
                log_weight = log(exit_rate_0);
                for (int j = 1; j < i; ++j)
                    log_weight += log(this->line_labels[j-1][1]);    // Edges j -> j-1 for j = 1, ..., i-1
                for (int j = i; j < this->N; ++j)
                    log_weight += log(this->line_labels[j][0]);      // Edges j -> j+1 for j = i, ..., N-1
                log_weight += log(exit_rate_N);
                log_two_forest_weights[i] = boost::multiprecision::logsumexp<T>(log_two_forest_weights[i-1], log_weight);
            }

            // Now get the weight of the last 2-forest, with a path from 
            // N to lower exit (upper exit is singleton)
            T log_lower_exit_weight = 0;
            for (unsigned i = 0; i < this->N; ++i)
                log_lower_exit_weight += log(this->line_labels[i][1]);
            log_lower_exit_weight += log(exit_rate_0);

            // Now get the weights of 3-forests rooted at the two exit
            // vertices and i, for i = 0, ..., N, with a path 0 -> i
            std::vector<T> log_three_forest_weights;
            for (unsigned i = 0; i <= this->N; ++i)
                log_three_forest_weights.push_back(0);

            // For each i = 0, ... N, start with the weight of the path 0 -> i
            for (unsigned i = 0; i <= this->N; ++i)
            {
                log_weight = 0;
                for (unsigned j = 0; j < i; ++j)
                    log_weight += log(this->line_labels[j][0]);

                // Then run through all the 3-forests with that path ...
                std::vector<T> log_factor_terms;
                for (unsigned j = 0; j < 1 + this->N - i; ++j)
                    log_factor_terms.push_back(0);
                
                // Start with the last forest (with the path N -> i)
                T term = 0;
                for (int j = this->N; j > i; --j)
                    term += log(this->line_labels[j-1][1]);
                log_factor_terms[0] = term; 

                for (unsigned j = i + 1; j <= this->N; ++j)
                {
                    // Then run through all forests with path N -> upper exit
                    term = log(exit_rate_N);
                    for (unsigned k = i + 1; k < j; ++k)
                        term += log(this->line_labels[k-1][1]);
                    for (unsigned k = j; k < this->N; ++k)
                        term += log(this->line_labels[k][0]);
                    log_factor_terms[j-i] = term;
                }

                log_three_forest_weights[i] = log_weight + boost::multiprecision::logsumexp<T>(log_factor_terms);
            }

            // Finally, compute the reciprocal of the mean first passage time
            std::vector<T> log_denom_terms;
            for (unsigned i = 0; i < this->N + 1; ++i)
                log_denom_terms.push_back(0);
            for (unsigned i = 0; i <= this->N; ++i)
                log_denom_terms[i] = log_three_forest_weights[i] + log_two_forest_weights[i];
            T log_denom = boost::multiprecision::logsumexp<T>(log_denom_terms); 

            return (
                log_two_forest_weights[0] +
                boost::multiprecision::logsumexp<T>(log_two_forest_weights[this->N], log_lower_exit_weight) - log_denom
            );
        }

        std::pair<std::vector<std::vector<int> >, std::vector<std::vector<double> > >
            simulate(boost::random::mt19937& rng, unsigned nsim, unsigned max_steps,
                     T lower_exit_rate = 1.0, T upper_exit_rate = 1.0)
        {
            /*
             * Simulate the Markov process on the graph.
             */
            std::vector<std::vector<int> > simulations; 
            std::vector<std::vector<double> > timepoints;

            // Run each simulation ...
            for (unsigned i = 0; i < nsim; ++i)
            {
                int n = 0;                              // n = 0, ..., N
                std::vector<double> rates;              // Vector of possible transition rates
                double rate_total = 0.0;                // Sum of all possible transition rates
                std::vector<int> states = {0};          // Vector of simulation states  
                std::vector<double> times = {0.0};      // Vector of transition timepoints
                unsigned nsteps = 1;                    // Number of simulation steps 

                while (!(n == -1 || n == this->N + 1) && nsteps < max_steps)
                {
                    if (n == 0)
                    {
                        rates.push_back(static_cast<double>(lower_exit_rate));           // Lower exit 
                        rates.push_back(static_cast<double>(this->line_labels[0][0]));   // 0 -> 1
                    }
                    else if (n == this->N)
                    {
                        rates.push_back(static_cast<double>(this->line_labels[this->N-1][1]));    // N -> N-1
                        rates.push_back(static_cast<double>(upper_exit_rate));                    // Upper exit
                    }
                    else
                    {
                        rates.push_back(static_cast<double>(this->line_labels[n-1][1]));    // n -> n-1
                        rates.push_back(static_cast<double>(this->line_labels[n][0]));      // n -> n+1
                    }
                    for (auto&& rate : rates) rate_total += rate;

                    // Randomly sample a waiting time 
                    boost::random::exponential_distribution<> time_dist(1.0 / rate_total);
                    double wait_time = time_dist(rng);
                    times.push_back(wait_time);

                    // Randomly sample a transition with probability proportional
                    // to its rate
                    boost::random::discrete_distribution<> transition_dist(rates);
                    int choice = transition_dist(rng);
                    if (choice == 0) n--; 
                    else             n++;
                    states.push_back(n);
                    nsteps++; 

                    // Clear rates and destinations 
                    rates.clear();
                    rate_total = 0.0;
                }

                simulations.push_back(states);
                timepoints.push_back(times);
            }

            return std::make_pair(simulations, timepoints);
        }
};

#endif 
