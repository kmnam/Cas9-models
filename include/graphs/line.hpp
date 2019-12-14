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
 *     12/13/2019
 */
using namespace Eigen;

template <typename T>
class LineGraph : public MarkovDigraph<T>
{
    /*
     * An implementation of the two-state grid graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        unsigned N;    // Length of the graph

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<T, 2> > line_labels;

    public:
        LineGraph() : MarkovDigraph<T>()
        {
            /*
             * Trivial constructor with length zero (one vertex, no edges).
             */
            this->addNode("0");
            this->N = 0;
        }

        LineGraph(unsigned N) : MarkovDigraph<T>()
        {
            /*
             * Trivial constructor with given length; set edge labels to unity.
             */
            this->N = N;
            for (unsigned i = 0; i < N; ++i)
            {
                std::stringstream si, sj;
                si << i;
                sj << i + 1;
                this->addEdge(si.str(), sj.str(), 1.0);
                this->addEdge(sj.str(), si.str(), 1.0);
                std::array<T, 2> labels = {1.0, 1.0};
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

        void addEndNode(std::array<T, 2> labels)
        {
            /*
             * Add new node onto the end of the graph, with the two 
             * additional edges. 
             */
            this->N++;
            this->line_labels.push_back(labels);
            std::stringstream si, sj;
            si << this->N - 1;
            sj << this->N;
            this->addEdge(si.str(), sj.str(), labels[0]);
            this->addEdge(sj.str(), si.str(), labels[1]);
        }

        void setLabels(unsigned i, std::array<T, 2> labels)
        {
            /*
             * Set the edge labels between the i-th and (i+1)-th nodes 
             * to the given values.
             *
             * That is, if i == 0, then the labels between nodes 0 and 1
             * are updated, and so on.
             */
            this->line_labels[i] = labels;
            std::stringstream si, sj;
            si << i;
            sj << i + 1;
            this->setEdgeLabel(si.str(), sj.str(), labels[0]);
            this->setEdgeLabel(sj.str(), si.str(), labels[1]);
        }

        Matrix<T, 2, 1> computeCleavageStatsByInverse(T kdis = 1.0, T kcat = 1.0)
        {
            /*
             * Compute probability of cleavage and (conditional) mean first passage
             * time to the cleaved state in the given model, with the specified
             * terminal rates of dissociation and cleavage, by directly solving
             * for the inverse of the Laplacian and its square.
             */
            // Compute the Laplacian of the graph
            Matrix<T, Dynamic, Dynamic> laplacian = -this->getLaplacian().transpose();

            // Update the Laplacian matrix with the specified terminal rates
            laplacian(0, 0) += kdis;
            laplacian(this->N, this->N) += kcat;

            // Solve matrix equation for cleavage probabilities
            Matrix<T, Dynamic, 1> term_rates = Matrix<T, Dynamic, 1>::Zero(this->N+1);
            term_rates(this->N) = kcat;
            Matrix<T, Dynamic, 1> probs = laplacian.colPivHouseholderQr().solve(term_rates);

            // Solve matrix equation for mean first passage times
            Matrix<T, Dynamic, 1> times = laplacian.colPivHouseholderQr().solve(probs);
            
            // Collect the two required quantities
            Matrix<T, 2, 1> stats;
            stats << probs(0), times(0) / probs(0);
            return stats;
        }

        Matrix<T, 2, 1> computeCleavageStats(T kdis = 1.0, T kcat = 1.0)
        {
            /*
             * Compute probability of cleavage and (conditional) mean first passage
             * time to the cleaved state in the given model, with the specified
             * terminal rates of dissociation and cleavage, by enumerating the
             * required spanning forests of the grid graph. 
             */
            auto bi = [this, kdis, kcat](int i)
            {
                return (i < this->N ? this->line_labels[i][0] : kcat);
            };

            auto di = [this, kdis, kcat](int i)
            {
                return (i == 0 ? kdis : this->line_labels[i-1][1]);
            };

            // Compute the probability of cleavage ...
            T prob = 1.0;
            for (int i = 0; i <= this->N; ++i)
            {
                T t = 1.0;
                for (int j = 0; j <= i; ++j) t *= di(j) / bi(j);
                prob += t;
            }
            prob = 1.0 / prob;

            // ... and the mean first passage time to the cleaved state
            T time = 0.0;
            for (int i = 0; i <= this->N; ++i)
            {
                T t1 = 1.0, t2 = 1.0;
                for (int j = i + 1; j <= this->N; ++j)
                {
                    T u1 = 1.0;
                    for (int k = i + 1; k <= j; ++k) u1 *= di(k) / bi(k);
                    t1 += u1;
                }
                for (int j = 0; j < i; ++j)
                {
                    T u2 = 1.0;
                    for (int k = 0; k <= j; ++k) u2 *= di(k) / bi(k);
                    t2 += u2;
                }
                time += (t1 * t2 / bi(i));
            }
            time *= prob;

            // Collect the two required quantities
            Matrix<T, 2, 1> stats;
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
        rates(i,0) = static_cast<double>(graph.labels[i][0]);
        rates(i,1) = static_cast<double>(graph.labels[i][1]);
    }
    stream << rates;
    return stream;
} 

#endif 
