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
 *     4/29/2021
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

        // Reordering of nodes
        std::vector<Node*> reorder;

        void reorderNodes()
        {
            /*
             * Re-order the nodes so that the even-numbered nodes come first,
             * followed by the nodes with numbers 1, 5, 9, ..., followed by 
             * the nodes with numbers 3, 7, 11, ...
             */
            unsigned i = 0;
            while (i <= this->N)
            {
                std::stringstream ssA, ssB, ssC;
                ssA << "A" << i;
                ssB << "B" << i;
                ssC << "C" << i;
                this->reorder.push_back(this->nodes[ssA.str()]);
                this->reorder.push_back(this->nodes[ssB.str()]);
                this->reorder.push_back(this->nodes[ssC.str()]);
                i += 2;
            }
            i = 1; 
            while (i <= this->N)
            {
                std::stringstream ssA, ssB, ssC;
                ssA << "A" << i;
                ssB << "B" << i;
                ssC << "C" << i;
                this->reorder.push_back(this->nodes[ssA.str()]);
                this->reorder.push_back(this->nodes[ssB.str()]);
                this->reorder.push_back(this->nodes[ssC.str()]);
                i += 4;
            }
            i = 3; 
            while (i <= this->N)
            {
                std::stringstream ssA, ssB, ssC;
                ssA << "A" << i;
                ssB << "B" << i;
                ssC << "C" << i;
                this->reorder.push_back(this->nodes[ssA.str()]);
                this->reorder.push_back(this->nodes[ssB.str()]);
                this->reorder.push_back(this->nodes[ssC.str()]);
                i += 4;
            }
        }

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

            // Deprecated: Re-order the nodes 
            //this->reorderNodes();
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

            // Deprecated: Re-order the nodes 
            //this->reorderNodes();
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
            Matrix<T, Dynamic, Dynamic> laplacian = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes + 2, this->numnodes + 2);

            // Run through the nodes in the re-ordering
            unsigned i = 0;
            //for (auto&& v : this->reorder)
            for (auto&& v : this->order)
            {
                unsigned j = 0;
                //for (auto&& w : this->reorder)
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
            laplacian(this->numnodes, this->numnodes + 1) = -exit_rate_upper_prob; 

            // Populate diagonal entries as negative sums of the off-diagonal
            // entries in each row
            for (unsigned i = 0; i < this->numnodes + 2; ++i)
                laplacian(i, i) = -(laplacian.row(i).sum());

            // Function for left-multiplying by the Laplacian matrix 
            std::function<Matrix<T, Dynamic, Dynamic>(const Ref<const Matrix<T, Dynamic, Dynamic> >)> multiply
                = [this, laplacian](const Ref<const Matrix<T, Dynamic, Dynamic> > B)
            {
                Matrix<T, Dynamic, Dynamic> product = Matrix<T, Dynamic, Dynamic>::Zero(this->numnodes + 2, this->numnodes + 2); 

                // First get rid of the first/last row and column 
                Matrix<T, Dynamic, Dynamic> sublaplacian = laplacian.block(1, 1, this->numnodes, this->numnodes);

                // Compute each block product ...
                for (unsigned q = 0; q <= this->N; ++q)        // q-th block row of Laplacian
                {
                    for (unsigned r = 0; r <= this->N; ++r)    // r-th block column of B 
                    {
                        Matrix<T, 3, 3> block_product;
                        if (q == 0)
                        {
                            // j = 0 and j = 1
                            block_product += sublaplacian.block(3*q, 0, 3, 3) * B.block(1, 1 + 3*r, 3, 3);
                            block_product += sublaplacian.block(3*q, 3, 3, 3) * B.block(4, 1 + 3*r, 3, 3);
                        }
                        else if (q == this->N)
                        {
                            // j = this->N - 1 and j = this->N
                            block_product += sublaplacian.block(3*q, 3*(this->N-1), 3, 3) * B.block(1 + 3*(this->N-1), 1 + 3*r, 3, 3);
                            block_product += sublaplacian.block(3*q, 3*this->N, 3, 3) * B.block(1 + 3*this->N, 1 + 3*r, 3, 3);
                        }
                        else
                        {
                            // j = q - 1, j = q, j = q + 1
                            block_product += sublaplacian.block(3*q, 3*(q-1), 3, 3) * B.block(1 + 3*(q-1), 1 + 3*r, 3, 3);
                            block_product += sublaplacian.block(3*q, 3*q, 3, 3) * B.block(1 + 3*q, 1 + 3*r, 3, 3);
                            block_product += sublaplacian.block(3*q, 3*(q+1), 3, 3) * B.block(1 + 3*(q+1), 1 + 3*r, 3, 3);
                        }
                        product.block(1 + 3*q, 1 + 3*r, 3, 3) = block_product; 
                    }
                }

                // Update the first/last row and column of the product matrix:
                // - first and last rows in Laplacian are zero, so they are also
                //   zero in the product matrix 
                // - first and last columns are not necessarily zero 
                product(1, 0) = laplacian(1, 0) * B.row(0).sum();
                product(this->numnodes, this->numnodes+1) = laplacian(this->numnodes, this->numnodes+1) * B.row(this->numnodes+1).sum();

                return product; 
            };

            // Solve the Chebotarev-Agaev recurrence for 3-forests 
            Matrix<T, Dynamic, Dynamic> forest_matrix_3 = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
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

            // Finally solve the recurrence for 1-forests (trees)
            forest_matrix_curr = multiply(forest_matrix_2);
            sigma_identity = (forest_matrix_curr.trace() / (this->numnodes + 2))
                * Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes + 2, this->numnodes + 2);
            Matrix<T, Dynamic, Dynamic> forest_matrix_1 = -forest_matrix_curr + sigma_identity;

            /*
            try
            {
                this->addNode("lower");
            }
            catch (const std::runtime_error& e)
            {
                ;
            }
            this->addEdge("A0", "lower", exit_rate_lower_prob);
            try
            {
                this->addNode("upper");
            }
            catch (const std::runtime_error& e)
            {
                ;
            }
            std::stringstream ss;
            ss << "C" << this->N; 
            this->addEdge(ss.str(), "upper", exit_rate_upper_prob);

            // Compute Laplacian matrix 
            Matrix<T, Dynamic, Dynamic> laplacian = (-this->getLaplacian()).transpose();

            // Solve the Chebotarev-Agaev recurrence for 3-forests
            Matrix<T, Dynamic, Dynamic> three_forest_matrix = this->getSpanningForestMatrix(this->numnodes - 3);

            // Then compute the Chebotarev-Agaev recurrence for 2-forests ...
            Matrix<T, Dynamic, Dynamic> two_forest_matrix;
            Matrix<T, Dynamic, Dynamic> product = (-laplacian) * three_forest_matrix;
            T sigma = -product.trace() / (this->numnodes - 2);
            Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);
            two_forest_matrix = product + (sigma * identity);

            // ... then compute the recurrence for 1-forests, i.e., trees ...
            Matrix<T, Dynamic, Dynamic> tree_matrix; 
            product = (-laplacian) * two_forest_matrix; 
            sigma = -product.trace() / (this->numnodes - 1);
            tree_matrix = product + (sigma * identity);

            if (exit_rate_lower_prob > 0 && exit_rate_upper_prob > 0)
            {
                // Get the weight of all trees rooted at (A,0)
                T weight_A0 = tree_matrix(0, 0);

                // Get the weight of all trees rooted at (C,N)
                T weight_CN = tree_matrix(this->numnodes - 1, this->numnodes - 1);

                // Get the weight of all 2-forests rooted at (A,0), (C,N)
                T weight_A0_CN = two_forest_matrix(

                // Get the weight of all 2-forests rooted at the two exits with 
                // a path from (A,0) -> upper
                T numer = forest_matrix(0, 3 * (this->N + 1) + 2 - 1);

                // Get the weight of all 2-forests rooted at the two exits
                T denom = forest_matrix(3 * (this->N + 1) + 2 - 1, 3 * (this->N + 1) + 2 - 1);

                prob_upper_exit = numer / denom;
            }
            else if (exit_rate_lower_prob > 0) 
            {
                prob_upper_exit = 0.0;
            }
            else 
            {
                prob_upper_exit = 1.0;
            }

            // Solve Chebotarev-Agaev recurrence again and compute the (reciprocal of the)
            // mean first passage time to the lower state from A0
            T rate_lower_exit, rate_upper_exit; 
            Matrix<T, Dynamic, Dynamic> forest_matrix_1, forest_matrix_2;
            if (exit_rate_lower_time > 0 || exit_rate_upper_time > 0)
            {
                // Re-label terminal edges
                this->setEdgeLabel("A0", "lower", exit_rate_lower_time);
                this->setEdgeLabel(ss.str(), "upper", exit_rate_upper_time);

                // If both lower and upper exits are possible, then the mean FPT 
                // to lower exit can be written in terms of 3- and 2-forest weights
                Matrix<T, Dynamic, Dynamic> laplacian = -(this->getLaplacian().transpose()); 
                Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);

                // Compute the first spanning forest matrix ...
                forest_matrix_1 = this->getSpanningForestMatrix(3 * (this->N + 1) + 2 - 3);

                // ... then the second spanning forest matrix
                T sigma = (laplacian * forest_matrix_1).trace() / (3 * (this->N + 1) + 2 - 3 + 1); 
                forest_matrix_2 = (-laplacian) * forest_matrix_1 + sigma * identity;

                // The numerator is given by the weight of all 2-forests rooted at 
                // the lower and upper exits with a path from A0 to lower exit, 
                // multiplied by the weight of all 2-forests rooted at the lower 
                // and upper exits (period)
                T numer = forest_matrix_2(0, 3 * (this->N + 1) + 2 - 2) *
                          forest_matrix_2(3 * (this->N + 1) + 2 - 2, 3 * (this->N + 1) + 2 - 2);

                // The denominator is given by the sum of all terms of the form P * Q,
                // where P is the weight of all 3-forests rooted at (Y,i), lower exit, and 
                // upper exit with a path (A,0) -> (Y,i); and Q is the weight of all 
                // 2-forests rooted at lower and upper exits with a path (Y,i) -> lower
                T denom = 0.0;
                for (unsigned i = 0; i < 3 * (this->N + 1); ++i)
                {
                    T p = forest_matrix_1(0, i);
                    T q = forest_matrix_2(i, 3 * (this->N + 1) + 2 - 2);
                    denom += (p * q);
                }
                rate_lower_exit = numer / denom;
            }
            else 
            {
                throw std::invalid_argument("Both exit rates are zero, so mean FPT to either exit is infinite");
            }

            // Solve Chebotarev-Agaev recurrence again and compute the (reciprocal of the)
            // mean first passage time to the upper state from A0
            if (exit_rate_lower_time > 0 || exit_rate_upper_time > 0)
            {
                // Re-label terminal edges
                this->setEdgeLabel("A0", "lower", exit_rate_lower_time);
                this->setEdgeLabel(ss.str(), "upper", exit_rate_upper_time);

                // If both lower and upper exits are possible, then the mean FPT 
                // to lower exit can be written in terms of 3- and 2-forest weights
                Matrix<T, Dynamic, Dynamic> laplacian = -(this->getLaplacian().transpose()); 
                Matrix<T, Dynamic, Dynamic> identity = Matrix<T, Dynamic, Dynamic>::Identity(this->numnodes, this->numnodes);

                // Compute the first spanning forest matrix ...
                forest_matrix_1 = this->getSpanningForestMatrix(3 * (this->N + 1) + 2 - 3);

                // ... then the second spanning forest matrix
                T sigma = (laplacian * forest_matrix_1).trace() / (3 * (this->N + 1) + 2 - 3 + 1); 
                forest_matrix_2 = (-laplacian) * forest_matrix_1 + sigma * identity;

                // The numerator is given by the weight of all 2-forests rooted at 
                // the lower and upper exits with a path from A0 to upper exit, 
                // multiplied by the weight of all 2-forests rooted at the lower 
                // and upper exits (period)
                T numer = forest_matrix_2(0, 3 * (this->N + 1) + 2 - 1) *
                          forest_matrix_2(3 * (this->N + 1) + 2 - 2, 3 * (this->N + 1) + 2 - 2);

                // The denominator is given by the sum of all terms of the form P * Q,
                // where P is the weight of all 3-forests rooted at (Y,i), lower exit, and 
                // upper exit with a path (A,0) -> (Y,i); and Q is the weight of all 
                // 2-forests rooted at lower and upper exits with a path (Y,i) -> upper
                T denom = 0.0;
                for (unsigned i = 0; i < 3 * (this->N + 1); ++i)
                {
                    T p = forest_matrix_1(0, i);
                    T q = forest_matrix_2(i, 3 * (this->N + 1) + 2 - 1);
                    denom += (p * q);
                }
                rate_upper_exit = numer / denom;
            }
            else 
            {
                throw std::invalid_argument("Both exit rates are zero, so mean FPT to either exit is infinite");
            }
            */

            return std::make_tuple(1, 1, 1);
            //return std::make_tuple(prob_upper_exit, rate_lower_exit, rate_upper_exit);
        }
};

#endif 
