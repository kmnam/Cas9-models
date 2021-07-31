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
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>

/* 
 * Implementation of the line graph.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     7/30/2021
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
     * Pseudo-variadic implementation of logsumexp(), which takes as input a
     * vector of numbers. 
     */
    using std::exp; 
    using std::log1p;    // Boost versions will be called if T is a boost::multiprecision type

    // First find the maximum entry in the vector 
    T max = x[0]; 
    unsigned argmax = 0;
    for (unsigned i = 1; i < x.size(); ++i)
    {
        if (max < x[i]);
        {
            max = x[i];
            argmax = i;
        }
    }

    // Then compute the logsumexp term by term
    T total = 0;
    for (unsigned i = 0; i < x.size(); ++i)
    {
        if (i != argmax)
            total += exp(x[i] - max);
    }

    return max + log1p(total); 
}

}   // namespace multiprecision

}   // namespace boost 

using namespace Eigen;

template <typename T>
class LineGraph : public LabeledDigraph<T>
{
    /*
     * An implementation of the line graph.
     *
     * The scalar type is assumed to be a boost::multiprecision floating-point
     * type, so as to allow use of the above logsumexp functions.  
     */
    private:
        unsigned N;    // Length of the graph

        // Canonical ordering of the nodes 
        std::vector<Node*> order; 

        // Vector of edge labels that grows with the length of the graph
        // Each array stores the labels for the edges i -> i+1 and i+1 -> i
        std::vector<std::array<T, 2> > line_labels;

        // Vector of log edge labels, stored for log-scale calculations 
        std::vector<std::array<T, 2> > log_line_labels; 

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
                std::array<T, 2> log_labels = {0, 0}; 
                this->log_line_labels.push_back(log_labels); 
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
            using std::log; 

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
            std::array<T, 2> log_labels;
            log_labels[0] = log(labels[0]); 
            log_labels[1] = log(labels[1]); 
            this->log_line_labels.push_back(log_labels);  
        }

        void setLabels(unsigned i, std::array<T, 2> labels)
        {
            /*
             * Set the edge labels between the i-th and (i+1)-th nodes 
             * (i -> i+1, then i+1 -> i) to the given values.
             */
            using std::log; 
            std::stringstream ssi, ssj;
            ssi << i;
            ssj << i + 1;
            this->setEdgeLabel(ssi.str(), ssj.str(), labels[0]);
            this->setEdgeLabel(ssj.str(), ssi.str(), labels[1]);
            this->line_labels[i] = labels;
            this->log_line_labels[i][0] = log(labels[0]); 
            this->log_line_labels[i][1] = log(labels[1]); 
        }

        T computeUpperExitProb(T exit_rate_0, T exit_rate_N)
        {
            /*
             * Compute the *log* probability of exiting the line graph through
             * the upper vertex N (Eqn. 4.1).
             *
             * This is done using log-semiring arithmetic (log-sum-exp for
             * addition, addition for multiplication), as this is more 
             * accurate for small edge labels and probabilities.  
             */
            using std::log;
            T log_exit_rate_0 = log(exit_rate_0); 
            T log_exit_rate_N = log(exit_rate_N); 

            // Start with the last log-ratio ...
            T log_prob_inverse = this->log_line_labels[this->N-1][1] - log_exit_rate_N; 
            log_prob_inverse = boost::multiprecision::logsumexp<T>(0, log_prob_inverse); 

            // ... then, for each vertex, get the log ratio of the reverse 
            // edge label divided by the forward edge label 
            for (int i = this->N - 2; i >= 0; --i)
            {
                log_prob_inverse += (this->log_line_labels[i][1] - this->log_line_labels[i+1][0]);
                log_prob_inverse = boost::multiprecision::logsumexp<T>(0, log_prob_inverse); 
            }
            log_prob_inverse += (log_exit_rate_0 - this->log_line_labels[0][0]); 
            log_prob_inverse = boost::multiprecision::logsumexp<T>(0, log_prob_inverse); 

            return -log_prob_inverse; 
        }

        T computeLowerExitRate(T exit_rate_0)
        {
            /*
             * Compute the *log* reciprocal of the mean first passage time
             * to exit through the lower vertex 0 when exit through the upper
             * vertex N is impossible. 
             *
             * This is done using log-semiring arithmetic (log-sum-exp for
             * addition, addition for multiplication), as this is more 
             * accurate for small edge labels and probabilities.  
             */
            using std::log; 
            T log_exit_rate_0 = log(exit_rate_0); 

            // Start with the last log-ratio ...
            T log_rate_inverse = this->log_line_labels[this->N-1][0] - this->log_line_labels[this->N-1][1];
            log_rate_inverse = boost::multiprecision::logsumexp<T>(0, log_rate_inverse); 

            // ... then, for each vertex, get the log ratio of the reverse 
            // edge label divided by the forward edge label 
            for (int i = this->N - 2; i >= 0; --i)
            {
                log_rate_inverse += (this->log_line_labels[i][0] - this->log_line_labels[i][1]);
                log_rate_inverse = boost::multiprecision::logsumexp<T>(0, log_rate_inverse); 
            }

            return -(log_rate_inverse - log_exit_rate_0); 
        }

        T computeUpperExitRate(T exit_rate_0, T exit_rate_N)
        {
            /*
             * Compute the *log* reciprocal of the mean first passage time
             * to exit through the upper vertex N, given that exit through
             * the upper vertex does occur. 
             *
             * This is done using log-semiring arithmetic (log-sum-exp for
             * addition, addition for multiplication), as this is more 
             * accurate for small edge labels and probabilities.  
             */
            using std::log; 
            T log_exit_rate_0 = log(exit_rate_0);
            T log_exit_rate_N = log(exit_rate_N);

            // Initialize the two recurrences ...
            std::vector<T> recur1, recur2;
            for (unsigned i = 0; i <= this->N; ++i)
            {
                recur1.push_back(0); 
                recur2.push_back(0); 
            }

            // Apply the second recurrence for i = 1, ..., N and the first 
            // recurrence for i = N-1, ..., 0
            for (int i2 = 1; i2 <= this->N; ++i2)
            {
                int i1 = this->N - i2;  

                // Compute the new terms to be added for each recurrence 
                T new1 = log_exit_rate_N; 
                for (int k = i1 + 1; k < this->N; ++k)
                    new1 += this->log_line_labels[k][0];
                T new2 = log_exit_rate_0;
                for (int k = 1; k <= i2 - 1; ++k)
                    new2 += this->log_line_labels[k-1][1];
                
                // Apply the recurrences
                T res1 = recur1[i1+1] + this->log_line_labels[i1][1];
                recur1[i1] = boost::multiprecision::logsumexp(res1, new1);
                T res2 = recur2[i2-1] + this->log_line_labels[i2-1][0];
                recur2[i2] = boost::multiprecision::logsumexp(res2, new2);  
            }

            // Apply the second recurrence once more to obtain the denominator 
            T denom = recur2[this->N];
            T new2 = log_exit_rate_0; 
            for (int k = 1; k <= this->N; ++k)
                new2 += this->log_line_labels[k-1][1];
            denom += log_exit_rate_N; 
            denom = boost::multiprecision::logsumexp(denom, new2);

            // Compute the numerator
            T numer = recur1[0] + recur2[0]; 
            for (int i = 1; i <= this->N; ++i)
            {
                T term = recur1[i] + recur2[i]; 
                numer = boost::multiprecision::logsumexp(numer, term); 
            } 

            return denom - numer; 
        }

        std::vector<std::vector<std::pair<int, double> > > simulate(unsigned nsims,
                                                                    T exit_rate_0, 
                                                                    T exit_rate_N,
                                                                    boost::random::mt19937& rng)
        {
            /*
             * Simulate the Markov process on the line graph, starting at 0.
             */
            std::vector<std::vector<std::pair<int, double> > > simulations; 

            for (unsigned i = 0; i < nsims; ++i)
            {
                std::vector<std::pair<int, double> > sim; 

                // Start at 0 ...
                int curr = 0;
                double time = 0;
                sim.push_back(std::make_pair(curr, time));

                // ... and run the simulation until exit occurs at either end
                while (curr >= 0 && curr <= this->N)
                {
                    // Sample an edge leaving the current state
                    T down, up; 
                    if (curr == 0)
                    {
                        down = exit_rate_0; 
                        up = this->line_labels[0][0]; 
                    }
                    else if (curr == this->N)
                    {
                        down = this->line_labels[this->N - 1][1];
                        up = exit_rate_N; 
                    } 
                    else 
                    {
                        down = this->line_labels[curr - 1][1]; 
                        up = this->line_labels[curr][0]; 
                    }
                    double prob_up = static_cast<double>(up / (down + up));
                    boost::random::bernoulli_distribution<double> dist(prob_up);
                    bool choice = dist(rng);
                    if (choice)     // Increase the current state
                        curr++;
                    else            // Decrease the current state
                        curr--;

                    // Sample a waiting time from the exponential distribution 
                    // determined by the label on the traversed edge 
                    boost::random::exponential_distribution<double> wait(static_cast<double>(up + down));
                    double waiting_time = wait(rng);
                    time += waiting_time; 
                    
                    // Record the new state with the waiting time 
                    sim.push_back(std::make_pair(curr, time));
                }

                // Append the completed simulation
                simulations.push_back(sim); 
            }

            return simulations; 
        }
};

#endif 
