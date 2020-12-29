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
 *     12/29/2020
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

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<T, 2> > line_labels;

    public:
        LineGraph() : LabeledDigraph<T>()
        {
            /*
             * Trivial constructor with length zero (one vertex, no edges).
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
                this->order.push_back(node_j);

                // Add edges i -> i+1 and i+1 -> i
                this->addEdge(ssi.str(), ssj.str());
                this->addEdge(ssj.str(), ssi.str());
                std::array<T, 2> labels = {1, 1};
                this->line_labels.push_back(labels);
            }
        }

        ~LineGraph() : ~LabeledDigraph<T>()
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
             * to the given values.
             */
            this->line_labels[i] = labels;
            std::stringstream ssi, ssj;
            ssi << i;
            ssj << i + 1;
            this->setEdgeLabel(ssi.str(), ssj.str(), labels[0]);
            this->setEdgeLabel(ssj.str(), ssi.str(), labels[1]);
        }

        template <typename U = T>
        U computeDissociationTime(U kdis = 1)
        {
            /*
             * Compute the mean first passage time to the dissociated state,
             * in the case where cleavage is abrogated. 
             */
            U time = 0;
            for (unsigned i = 0; i <= this->N; ++i)
            {
                U term = 1;
                for (unsigned j = 0; j < i; ++j)
                {
                    U forward = this->line_labels[j][0];
                    U reverse = this->line_labels[j][1];
                    term *= (forward / reverse);
                }
                time += (term / kdis);
            }
            return time;
        }

        template <typename U = T>
        Matrix<U, 2, 1> computeCleavageStatsByInverse(U kdis = 1, U kcat = 1)
        {
            /*
             * Compute probability of cleavage and (conditional) mean first passage
             * time to the cleaved state in the given model, with the specified
             * terminal rates of dissociation and cleavage, by directly solving
             * for the inverse of the modified Laplacian and its square.
             */
            // Compute the Laplacian of the graph
            Matrix<U, Dynamic, Dynamic> laplacian = -this->template getLaplacian<U>(this->order).transpose();

            // Update the Laplacian matrix with the specified terminal rates
            laplacian(0, 0) += kdis;
            laplacian(this->N, this->N) += kcat;

            // Solve matrix equation for cleavage probabilities
            Matrix<U, Dynamic, 1> term_rates = Matrix<U, Dynamic, 1>::Zero(this->N+1);
            term_rates(this->N) = kcat;
            Matrix<U, Dynamic, 1> probs = laplacian.colPivHouseholderQr().solve(term_rates);

            // Solve matrix equation for mean first passage times
            Matrix<U, Dynamic, 1> times = laplacian.colPivHouseholderQr().solve(probs);
            
            // Collect the two required quantities
            Matrix<U, 2, 1> stats;
            stats << probs(0), times(0) / probs(0);
            return stats;
        }

        template <typename U = T>
        Matrix<U, 2, 1> computeRejectionStatsByInverse(U kdis = 1, U kcat = 1)
        {
            /*
             * Compute probability of rejection and (conditional) mean first passage
             * time to the dissociated state in the given model, with the specified
             * terminal rates of dissociation and cleavage, by directly solving 
             * for the inverse of the modified Laplacian and its square.
             */
            // Compute the Laplacian of the graph
            Matrix<U, Dynamic, Dynamic> laplacian = -this->template getLaplacian<U>(this->order).transpose();

            // Update the Laplacian matrix with the specified terminal rates
            laplacian(0, 0) += kdis;
            laplacian(this->N, this->N) += kcat;

            // Solve matrix equation for cleavage probabilities
            Matrix<U, Dynamic, 1> term_rates = Matrix<U, Dynamic, 1>::Zero(this->N+1);
            term_rates(0) = kdis;
            Matrix<U, Dynamic, 1> probs = laplacian.colPivHouseholderQr().solve(term_rates);

            // Solve matrix equation for mean first passage times
            Matrix<U, Dynamic, 1> times = laplacian.colPivHouseholderQr().solve(probs);
            
            // Collect the two required quantities
            Matrix<U, 2, 1> stats;
            stats << probs(0), times(0) / probs(0);
            return stats;
        }

        template <typename U = T>
        Matrix<U, 2, 1> computeCleavageStats(U kdis = 1, U kcat = 1)
        {
            /*
             * Compute probability of cleavage and (conditional) mean first passage
             * time to the cleaved state in the given model, with the specified
             * terminal rates of dissociation and cleavage, by enumerating the
             * required spanning forests of the grid graph. 
             */
            std::function<U(int)> bi = [this, kdis, kcat](int i)
            {
                return (i < this->N ? this->line_labels[i][0] : kcat);
            };

            std::function<U(int)> di = [this, kdis, kcat](int i)
            {
                return (i == 0 ? kdis : this->line_labels[i-1][1]);
            };

            // Compute the probability of cleavage ...
            U prob = 1;
            for (int i = 0; i <= this->N; ++i)
            {
                U t = 1;
                for (int j = 0; j <= i; ++j) U *= di(j) / bi(j);
                prob += t;
            }
            prob = 1 / prob;

            // ... and the mean first passage time to the cleaved state
            U time = 0;
            for (int i = 0; i <= this->N; ++i)
            {
                U t1 = 1, t2 = 1;
                for (int j = i + 1; j <= this->N; ++j)
                {
                    U u1 = 1;
                    for (int k = i + 1; k <= j; ++k) u1 *= di(k) / bi(k);
                    t1 += u1;
                }
                for (int j = 0; j < i; ++j)
                {
                    U u2 = 1;
                    for (int k = 0; k <= j; ++k) u2 *= di(k) / bi(k);
                    t2 += u2;
                }
                time += (t1 * t2 / bi(i));
            }
            time *= prob;

            // Collect the two required quantities
            Matrix<U, 2, 1> stats;
            stats << prob, time;
            return stats; 
        }

        template <typename U>
        friend std::ostream& operator<<(std::ostream& stream, const LineGraph<U>& graph);
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const LineGraph<T>& graph)
{
    /*
     * Output to the given stream. 
     */
    MatrixXd rates(graph.N, 2);
    for (unsigned i = 0; i < graph.N; ++i)
    {
        rates(i,0) = static_cast<double>(graph.line_labels[i][0]);
        rates(i,1) = static_cast<double>(graph.line_labels[i][1]);
    }
    stream << rates;
    return stream;
} 

#endif 
