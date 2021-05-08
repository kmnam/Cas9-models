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
 *     5/7/2021
 */
using namespace Eigen;

template <typename T>
class TriangularPrismGraph : public LabeledDigraph<T>
{
    /*
     * An implementation of the triangular-prism graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        unsigned N;    // Length of the graph

        // The A <--> B <--> C <--> A edge labels for the zeroth rung of the graph
        std::array<T, 6> start;

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<T, 12> > rung_labels;

    public:
        TriangularPrismGraph() : LabeledDigraph<T>()
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

        TriangularPrismGraph(unsigned N) : LabeledDigraph<T>()
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
            sai << "A" << this->N - 1;
            sbi << "B" << this->N - 1;
            sci << "C" << this->N - 1;
            saj << "A" << this->N;
            sbj << "B" << this->N;
            scj << "C" << this->N;
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
            this->setEdgeLabel(sai.str(), saj.str(), labels[0]);   // (A,i) <--> (A,i+1)
            this->setEdgeLabel(saj.str(), sai.str(), labels[1]);
            this->setEdgeLabel(sbi.str(), sbj.str(), labels[2]);   // (B,i) <--> (B,i+1)
            this->setEdgeLabel(sbj.str(), sbi.str(), labels[3]);
            this->setEdgeLabel(sci.str(), scj.str(), labels[4]);   // (C,i) <--> (C,i+1)
            this->setEdgeLabel(scj.str(), sci.str(), labels[5]);
            this->setEdgeLabel(saj.str(), sbj.str(), labels[6]);   // (A,i) <--> (B,i)
            this->setEdgeLabel(sbj.str(), saj.str(), labels[7]);
            this->setEdgeLabel(sbj.str(), scj.str(), labels[8]);   // (B,i) <--> (C,i)
            this->setEdgeLabel(scj.str(), sbj.str(), labels[9]);
            this->setEdgeLabel(scj.str(), saj.str(), labels[10]);  // (C,i) <--> (A,i)
            this->setEdgeLabel(saj.str(), scj.str(), labels[11]);
        }

        std::tuple<T, T, T> computeExitStats(T exit_rate_lower_prob = 1, T exit_rate_upper_prob = 1,
                                             T exit_rate_lower_time = 1, T exit_rate_upper_time = 1)
        {
            /*
             * Compute the probability of upper exit (from (C,N)) and the 
             * rate of lower exit (from (A,0)). 
             */
            using std::log; 
            using std::exp; 
            using std::log1p; 

            Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes+2, this->numnodes+2);

            // Run through the nodes in the re-ordering
            unsigned i = 0;
            for (auto&& v : this->order)
            {
                unsigned j = 0;
                for (auto&& w : this->order)
                {
                    // Does an edge exist from v to w?
                    auto it = this->edges[v].find(w);
                    if (it != this->edges[v].end())
                    {
                        // If so, update the corresponding entry in the Laplacian 
                        laplacian(i+1, j+1) = -(it->second);
                    }
                    j++;
                }
                i++;
            }

            // Edge (A,0) -> lower 
            laplacian(1, 0) = -exit_rate_lower_prob; 

            // Edge (C,N) -> upper 
            laplacian(this->numnodes, this->numnodes+1) = -exit_rate_upper_prob; 

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each row
            for (unsigned i = 0; i < this->numnodes + 2; ++i)
                laplacian(i, i) = -(laplacian.row(i).sum());

            // Function for left-multiplying by the Laplacian matrix 
            std::function<Matrix<T, Dynamic, Dynamic>(const Ref<const Matrix<T, Dynamic, Dynamic> >)> multiply
                = [this, laplacian](const Ref<const Matrix<T, Dynamic, Dynamic> > B)
            {
                Matrix<T, Dynamic, Dynamic> product = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes+2, this->numnodes+2); 

                // First get rid of the first/last row/column 
                Matrix<T, Dynamic, Dynamic> sublaplacian = laplacian.block(1, 1, this->numnodes, this->numnodes);
                Matrix<T, 3, 3> X, Y; 

                // Compute each block product ...
                for (unsigned q = 0; q <= this->N; ++q)        // q-th block row of Laplacian
                {
                    for (unsigned r = 0; r <= this->N; ++r)    // r-th block column of B 
                    {
                        Matrix<T, 3, 3> block_product;
                        if (q == 0)
                        {
                            // j = 0 and j = 1
                            X = sublaplacian.block(3*q, 0, 3, 3); 
                            Y = B.block(1, 1 + 3*r, 3, 3);
                            block_product += X * Y; 
                            X = sublaplacian.block(3*q, 3, 3, 3);
                            Y = B.block(4, 1 + 3*r, 3, 3);
                            block_product += X * Y; 
                        }
                        else if (q == this->N)
                        {
                            // j = this->N - 1 and j = this->N
                            X = sublaplacian.block(3*q, 3*(this->N-1), 3, 3); 
                            Y = B.block(1 + 3*(this->N-1), 1 + 3*r, 3, 3); 
                            block_product += X * Y; 
                            X = sublaplacian.block(3*q, 3*this->N, 3, 3);
                            Y = B.block(1 + 3*this->N, 1 + 3*r, 3, 3);
                            block_product += X * Y; 
                        }
                        else
                        {
                            // j = q - 1, j = q, j = q + 1
                            X = sublaplacian.block(3*q, 3*(q-1), 3, 3);
                            Y = B.block(1 + 3*(q-1), 1 + 3*r, 3, 3);
                            block_product += X * Y;
                            X = sublaplacian.block(3*q, 3*q, 3, 3);
                            Y = B.block(1 + 3*q, 1 + 3*r, 3, 3);
                            block_product += X * Y; 
                            X = sublaplacian.block(3*q, 3*(q+1), 3, 3);
                            Y = B.block(1 + 3*(q+1), 1 + 3*r, 3, 3); 
                            block_product += X * Y; 
                        }
                        product.block(1 + 3*q, 1 + 3*r, 3, 3) = block_product; 
                    }
                }

                // Update the first/last row and column of the product matrix:
                // - first and last rows in Laplacian are zero, so they are also
                //   zero in the product matrix 
                // - first and last columns are not necessarily zero 
                product.row(1) += laplacian(1, 0) * B.row(0);
                product.row(this->numnodes)
                    += laplacian(this->numnodes, this->numnodes+1) * B.row(this->numnodes+1);

                return product;
            };

            // Solve the Chebotarev-Agaev recurrence for 3-forests 
            Matrix<T, Dynamic, Dynamic> forest_matrix_3
                = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
            Matrix<T, Dynamic, Dynamic> forest_matrix_curr, sigma_identity;
            for (int k = 0; k < this->numnodes - 1; ++k)
            {
                forest_matrix_curr = multiply(forest_matrix_3);
                T sigma = forest_matrix_curr.trace() / (k + 1);
                sigma_identity = sigma * Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
                forest_matrix_3 = -forest_matrix_curr + sigma_identity; 
            }

            // Then solve the recurrence for 2-forests
            forest_matrix_curr = multiply(forest_matrix_3);
            sigma_identity = (forest_matrix_curr.trace() / (this->numnodes + 1))
                * Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
            Matrix<T, Dynamic, Dynamic> forest_matrix_2 = -forest_matrix_curr + sigma_identity;

            T log_prob_upper_exit; 
            if (exit_rate_lower_prob > 0 && exit_rate_upper_prob > 0)
            {
                // Get the weight of all 2-forests rooted at the two exit vertices  
                T denom = forest_matrix_2(0, 0);

                // Get the weight of all 2-forests rooted at the two exits with 
                // a path from (A,0) -> upper
                T numer = forest_matrix_2(1, this->numnodes + 1);

                log_prob_upper_exit = log(numer) - log(denom);
            }
            else if (exit_rate_lower_prob > 0) 
            {
                log_prob_upper_exit = -std::numeric_limits<T>::infinity();
            }
            else 
            {
                log_prob_upper_exit = 0.0;
            }

            // Solve Chebotarev-Agaev recurrence again and compute the (reciprocal of the)
            // mean first passage time to the lower state from A0
            T log_rate_lower_exit, log_rate_upper_exit; 
            if (exit_rate_lower_time > 0 || exit_rate_upper_time > 0)
            {
                // Re-label terminal edges
                laplacian(1, 0) = -exit_rate_lower_time; 
                laplacian(this->numnodes, this->numnodes+1) = -exit_rate_upper_time; 

                // If both lower and upper exits are possible, then the mean FPT 
                // to lower exit can be written in terms of 3- and 2-forest weights
                forest_matrix_3 = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
                for (int k = 0; k < this->numnodes - 1; ++k)
                {
                    forest_matrix_curr = multiply(forest_matrix_3);
                    T sigma = forest_matrix_curr.trace() / (k + 1);
                    sigma_identity = sigma * Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
                    forest_matrix_3 = -forest_matrix_curr + sigma_identity; 
                }

                // Then solve the recurrence for 2-forests
                forest_matrix_curr = multiply(forest_matrix_3);
                sigma_identity = (forest_matrix_curr.trace() / (this->numnodes + 1))
                    * Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
                forest_matrix_2 = -forest_matrix_curr + sigma_identity;

                // The numerator is given by the weight of all 2-forests rooted at 
                // the lower and upper exits with a path from A0 to lower exit, 
                // multiplied by the weight of all 2-forests rooted at the lower 
                // and upper exits (period)
                T log_numer = log(forest_matrix_2(1, 0)) + log(forest_matrix_2(0, 0)); 

                // The denominator is given by the sum of all terms of the form P * Q,
                // where P is the weight of all 3-forests rooted at (Y,i), lower exit, and 
                // upper exit with a path (A,0) -> (Y,i); and Q is the weight of all 
                // 2-forests rooted at lower and upper exits with a path (Y,i) -> lower
                std::vector<T> log_denom_terms;
                T max = 0; unsigned argmax; 
                for (unsigned i = 1; i < this->numnodes + 2; ++i)
                {
                    T p = forest_matrix_3(1, i);
                    T q = forest_matrix_2(i, 0);
                    T term = log(p) + log(q); 
                    log_denom_terms.push_back(term);
                    if (term > max)
                    {
                        max = term;
                        argmax = i - 1; 
                    }
                }
                T log_denom = 0;
                for (unsigned i = 0; i < this->numnodes + 1; ++i)
                {
                    if (i != argmax)
                        log_denom += exp(log_denom_terms[i] - max);
                }
                log_denom = max + log1p(log_denom); 
                log_rate_lower_exit = log_numer - log_denom;

                // The numerator is given by the weight of all 2-forests rooted at 
                // the lower and upper exits with a path from A0 to upper exit, 
                // multiplied by the weight of all 2-forests rooted at the lower 
                // and upper exits (period)
                log_numer = log(forest_matrix_2(1, this->numnodes + 1)) + log(forest_matrix_2(0, 0)); 

                // The denominator is given by the sum of all terms of the form P * Q,
                // where P is the weight of all 3-forests rooted at (Y,i), lower exit, and 
                // upper exit with a path (A,0) -> (Y,i); and Q is the weight of all 
                // 2-forests rooted at lower and upper exits with a path (Y,i) -> upper
                max = 0;  
                for (unsigned i = 1; i < this->numnodes + 2; ++i)
                {
                    T p = forest_matrix_3(1, i);
                    T q = forest_matrix_2(i, this->numnodes + 1);
                    T term = log(p) + log(q); 
                    log_denom_terms[i-1] = term;
                    if (term > max)
                    {
                        max = term;
                        argmax = i - 1; 
                    }
                }
                log_denom = 0;
                for (unsigned i = 0; i < this->numnodes + 1; ++i)
                {
                    if (i != argmax)
                        log_denom += exp(log_denom_terms[i] - max);
                }
                log_denom = max + log1p(log_denom); 
                log_rate_upper_exit = log_numer - log_denom;
            }
            else 
            {
                throw std::invalid_argument("Both exit rates are zero, so mean FPT to either exit is infinite");
            }

            return std::make_tuple(log_prob_upper_exit, log_rate_lower_exit, log_rate_upper_exit);
        }
};

#endif 
