#ifndef LINE_GRAPH_HPP
#define LINE_GRAPH_HPP

#include <sstream>
#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <Eigen/Dense>
#include <digraph.hpp>

/* 
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     2/16/2021
 */
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
             * Compute the probability of exiting the line graph through
             * the upper vertex N. 
             */
            // Start with 1 ...
            T prob = 1;

            // ... then, for each vertex, get the ratio of the reverse 
            // edge label divided by the forward edge label 
            T curr = exit_rate_0 / this->line_labels[0][0];
            prob += curr;
            for (unsigned i = 1; i < this->N; ++i)
            {
                curr *= this->line_labels[i-1][1] / this->line_labels[i][0];
                prob += curr;
            }
            curr *= this->line_labels[this->N - 1][1] / exit_rate_N;
            prob += curr;

            // ... then take the reciprocal
            return (1.0 / prob);
        }

        T computeLowerExitRate(T exit_rate_0 = 1, T exit_rate_N = 1)
        {
            /*
             * Compute the reciprocal of the mean first passage time to exit
             * through the lower vertex 0, in the case that lower exit does
             * occur.
             */
            // Get the weights of 2-forests rooted at the two exit vertices
            // with path from i to lower exit, for i = N, ..., 0
            std::vector<T> two_forest_weights;
            for (unsigned i = 0; i <= this->N; ++i)
                two_forest_weights.push_back(0);
            
            // Start with i = N
            T weight = exit_rate_0;
            for (unsigned i = 0; i < this->N; ++i)
                weight *= this->line_labels[i][1];
            two_forest_weights[this->N] = weight;

            // Then run from N-1 to 0
            for (int i = this->N - 1; i >= 0; --i)
            {
                weight = exit_rate_0;
                for (int j = 0; j < i; ++j)
                    weight *= this->line_labels[j][1];
                for (int j = i; j < this->N - 1; ++j)
                    weight *= this->line_labels[j+1][0];
                weight *= exit_rate_N;
                two_forest_weights[i] = two_forest_weights[i+1] + weight;
            }

            // Now get the weight of the last 2-forest, with a path from 
            // 0 to upper exit
            T upper_exit_weight = 1;
            for (unsigned i = 0; i < this->N; ++i)
                upper_exit_weight *= this->line_labels[i][0];
            upper_exit_weight *= exit_rate_N;

            // Now get the weights of 3-forests rooted at the two exit
            // vertices and i, for i = 0, ..., N, with a path 0 -> i
            std::vector<T> three_forest_weights;
            for (unsigned i = 0; i <= this->N; ++i)
                three_forest_weights.push_back(0);

            // For each i = 0, ... N, start with the weight of the path 0 -> i
            for (unsigned i = 0; i <= this->N; ++i)
            {
                weight = 1;
                for (unsigned j = 0; j < i; ++j)
                    weight *= this->line_labels[j][0];

                // Then run through all the 3-forests with that path ...
                T total = 0;
                
                // Start with the last forest (with the path N -> i)
                T factor = 1;
                for (int j = this->N; j > i; --j)
                    factor *= this->line_labels[j-1][1];
                total += factor;

                for (unsigned j = i + 1; j <= this->N; ++j)
                {
                    // Then run through all forests with path N -> upper exit
                    factor = exit_rate_N;
                    for (unsigned k = i + 1; k < j; ++k)
                        factor *= this->line_labels[k-1][1];
                    for (unsigned k = j; k < this->N; ++k)
                        factor *= this->line_labels[k][0];
                    total += factor;
                }

                three_forest_weights[i] = weight * total;
            }

            // Finally, compute the reciprocal of the mean first passage time
            T denom = 0;
            for (unsigned i = 0; i <= this->N; ++i)
                denom += (three_forest_weights[i] * two_forest_weights[i]);

            return (two_forest_weights[0] * (two_forest_weights[0] + upper_exit_weight) / denom);
        }
};

#endif 
