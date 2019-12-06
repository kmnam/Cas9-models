#ifndef TRIANGULAR_PRISM_GRAPH_HPP
#define TRIANGULAR_PRISM_GRAPH_HPP

#include <iostream>
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
 *     12/6/2019
 */
using namespace Eigen;

template <typename T>
class TriangularPrismGraph : public MarkovDigraph<T>
{
    /*
     * An implementation of the two-state grid graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        unsigned N;    // Length of the graph

        // The A <--> B <--> C <--> A edge labels for the zeroth rung of the graph
        std::array<T, 6> start;

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<T, 12> > rung_labels;

    public:
        TriangularPrismGraph() : MarkovDigraph<T>()
        {
            /*
             * Trivial constructor with length zero; set edge labels to unity.
             */
            this->addEdge("A0", "B0", 1.0);
            this->addEdge("B0", "A0", 1.0);
            this->addEdge("B0", "C0", 1.0);
            this->addEdge("C0", "B0", 1.0);
            this->addEdge("C0", "A0", 1.0);
            this->addEdge("A0", "C0", 1.0);
            this->N = 0;
            this->start[0] = 1.0;
            this->start[1] = 1.0;
            this->start[2] = 1.0;
            this->start[3] = 1.0;
            this->start[4] = 1.0;
            this->start[5] = 1.0;
        }

        TriangularPrismGraph(unsigned N) : MarkovDigraph<T>()
        {
            /*
             * Trivial constructor with given length; set edge labels to unity.
             */
            this->addEdge("A0", "B0", 1.0);
            this->addEdge("B0", "A0", 1.0);
            this->addEdge("B0", "C0", 1.0);
            this->addEdge("C0", "B0", 1.0);
            this->addEdge("C0", "A0", 1.0);
            this->addEdge("A0", "C0", 1.0);
            this->N = N;
            this->start[0] = 1.0;
            this->start[1] = 1.0;
            this->start[2] = 1.0;
            this->start[3] = 1.0;
            this->start[4] = 1.0;
            this->start[5] = 1.0;
            for (unsigned i = 0; i < N; ++i)
            {
                std::stringstream sai, sbi, sci, saj, sbj, scj;
                sai << "A" << i;
                sbi << "B" << i;
                sci << "C" << i;
                saj << "A" << i + 1;
                sbj << "B" << i + 1;
                scj << "C" << i + 1;
                this->addEdge(sai.str(), saj.str(), 1.0);   // (A,i) <--> (A,i+1)
                this->addEdge(saj.str(), sai.str(), 1.0);
                this->addEdge(sbi.str(), sbj.str(), 1.0);   // (B,i) <--> (B,i+1)
                this->addEdge(sbj.str(), sbi.str(), 1.0);
                this->addEdge(sci.str(), scj.str(), 1.0);   // (C,i) <--> (C,i+1)
                this->addEdge(scj.str(), sci.str(), 1.0);
                this->addEdge(saj.str(), sbj.str(), 1.0);   // (A,i) <--> (B,i)
                this->addEdge(sbj.str(), saj.str(), 1.0);
                this->addEdge(sbj.str(), scj.str(), 1.0);   // (B,i) <--> (C,i)
                this->addEdge(scj.str(), sbj.str(), 1.0);
                this->addEdge(scj.str(), saj.str(), 1.0);   // (C,i) <--> (A,i)
                this->addEdge(saj.str(), scj.str(), 1.0);
                std::array<T, 12> labels = {
                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                };
                this->rung_labels.push_back(labels);
            }
        }

        ~TriangularPrismGraph()
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

        void setStartLabels(T A_to_B, T B_to_A, T B_to_C, T C_to_B, T C_to_A, T A_to_C)
        {
            /*
             * Set the edge labels of the zeroth rung of the graph to the
             * given values.
             */
            this->setEdgeLabel("A0", "B0", A_to_B);
            this->setEdgeLabel("B0", "A0", B_to_A);
            this->setEdgeLabel("B0", "C0", B_to_C);
            this->setEdgeLabel("C0", "B0", C_to_B);
            this->setEdgeLabel("C0", "A0", C_to_A);
            this->setEdgeLabel("A0", "C0", A_to_C);
            this->start[0] = A_to_B;
            this->start[1] = B_to_A;
            this->start[2] = B_to_C;
            this->start[3] = C_to_B;
            this->start[4] = C_to_A;
            this->start[5] = A_to_C;
        }

        void addRung(std::array<T, 12> labels)
        {
            /*
             * Add new rung onto the end of the graph, keeping track of the
             * six new edge labels. 
             */
            this->N++;
            this->rung_labels.push_back(labels);
            std::stringstream sai, sbi, sci, saj, sbj, scj;
            sai << "A" << i;
            sbi << "B" << i;
            sci << "C" << i;
            saj << "A" << i + 1;
            sbj << "B" << i + 1;
            scj << "C" << i + 1;
            this->addEdge(sai.str(), saj.str(), labels[0]);   // (A,i) <--> (A,i+1)
            this->addEdge(saj.str(), sai.str(), labels[1]);
            this->addEdge(sbi.str(), sbj.str(), labels[2]);   // (B,i) <--> (B,i+1)
            this->addEdge(sbj.str(), sbi.str(), labels[3]);
            this->addEdge(sci.str(), scj.str(), labels[4]);   // (C,i) <--> (C,i+1)
            this->addEdge(scj.str(), sci.str(), labels[5]);
            this->addEdge(saj.str(), sbj.str(), labels[6]);   // (A,i) <--> (B,i)
            this->addEdge(sbj.str(), saj.str(), labels[7]);
            this->addEdge(sbj.str(), scj.str(), labels[8]);   // (B,i) <--> (C,i)
            this->addEdge(scj.str(), sbj.str(), labels[9]);
            this->addEdge(scj.str(), saj.str(), labels[10]);  // (C,i) <--> (A,i)
            this->addEdge(saj.str(), scj.str(), labels[11]);
        }

        void setRungLabels(unsigned i, std::array<T, 12> labels)
        {
            /*
             * Set the edge labels for the i-th rung to the given values. 
             */
            this->rung_labels[i] = labels;
            std::stringstream sai, sbi, sci, saj, sbj, scj;
            sai << "A" << i;
            sbi << "B" << i;
            sci << "C" << i;
            saj << "A" << i + 1;
            sbj << "B" << i + 1;
            scj << "C" << i + 1;
            this->addEdge(sai.str(), saj.str(), labels[0]);   // (A,i) <--> (A,i+1)
            this->addEdge(saj.str(), sai.str(), labels[1]);
            this->addEdge(sbi.str(), sbj.str(), labels[2]);   // (B,i) <--> (B,i+1)
            this->addEdge(sbj.str(), sbi.str(), labels[3]);
            this->addEdge(sci.str(), scj.str(), labels[4]);   // (C,i) <--> (C,i+1)
            this->addEdge(scj.str(), sci.str(), labels[5]);
            this->addEdge(saj.str(), sbj.str(), labels[6]);   // (A,i) <--> (B,i)
            this->addEdge(sbj.str(), saj.str(), labels[7]);
            this->addEdge(sbj.str(), scj.str(), labels[8]);   // (B,i) <--> (C,i)
            this->addEdge(scj.str(), sbj.str(), labels[9]);
            this->addEdge(scj.str(), saj.str(), labels[10]);  // (C,i) <--> (A,i)
            this->addEdge(saj.str(), scj.str(), labels[11]);
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
            laplacian(3*this->N+2, 3*this->N+2) += kcat;

            // Solve matrix equation for cleavage probabilities
            Matrix<T, Dynamic, 1> term_rates = Matrix<T, Dynamic, 1>::Zero(3*this->N+3);
            term_rates(3*this->N+2) = kcat;
            Matrix<T, Dynamic, 1> probs = laplacian.colPivHouseholderQr().solve(term_rates);

            // Solve matrix equation for mean first passage times
            Matrix<T, Dynamic, 1> times = laplacian.colPivHouseholderQr().solve(probs);
            
            // Collect the two required quantities
            Matrix<T, 2, 1> stats;
            stats << probs(0), times(0) / probs(0);
            return stats;
        }
};

#endif 
