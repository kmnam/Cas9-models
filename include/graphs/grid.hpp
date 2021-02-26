#ifndef GRID_GRAPH_HPP
#define GRID_GRAPH_HPP

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
 *     2/24/2021
 */
using namespace Eigen;

template <typename T>
std::array<T, 3> operatorA(std::array<T, 3>& v, std::array<T, 6>& labels)
{
    /*
     * Apply operator A at the j-th rung of the grid graph. 
     */
    std::array<T, 3> w;
    T fA = labels[0];
    T rA = labels[1];
    T fB = labels[2];
    T rB = labels[3];
    T c = labels[4];
    T d = labels[5];
    w[0] = (rA*d + rB*c + rA*rB)*v[0] + (fA*rB*c)*v[1] + (fB*rA*d)*v[2];
    w[1] = rB*v[0] + (fA*rB)*v[1];
    w[2] = rA*v[0] + (fB*rA)*v[2];
   
    return w; 
}

template <typename T>
std::array<T, 9> operatorB(std::array<T, 3>& v, std::array<T, 6>& labels)
{
    /*
     * Apply operator B at the j-th rung of the grid graph. 
     */
    std::array<T, 9> w;
    T fA = labels[0];
    T rA = labels[1];
    T fB = labels[2];
    T rB = labels[3];
    T c = labels[4];
    T d = labels[5];
    w[0] = (rA*d + rB*c + rA*rB)*v[0] + (fB*rA*d)*v[2];
    w[1] = (rA*d + rB*c + rA*rB)*v[1] + (fA*rB*c)*v[2];
    w[2] = (fA*d + fA*rB)*v[0] + (fB*d)*v[1] + (fA*fB*d)*v[2];
    w[3] = (fA*c)*v[0] + (fB*c + fB*rA)*v[1] + (fA*fB*c)*v[2];
    w[4] = rB*v[0];
    w[5] = rA*v[0] + (fB*rA)*v[2];
    w[6] = rB*v[1] + (fA*rB)*v[2];
    w[7] = rA*v[1];
    w[8] = fA*v[0] + fB*v[1] + (fA*fB)*v[2];

    return w;
}

template <typename T>
std::array<T, 3> operatorC(std::array<T, 4>& v, std::array<T, 6>& labels)
{
    /*
     * Apply operator C at the j-th rung of the grid graph.
     */
    std::array<T, 3> w;
    T fA = labels[0];
    T rA = labels[1];
    T fB = labels[2];
    T rB = labels[3];
    T c = labels[4];
    T d = labels[5];
    w[0] = (d + rB)*v[0] + (fA*d + fA*rB)*v[1] + (fB*d)*v[2] + (fA*fB*d)*v[3];
    w[1] = (c + rA)*v[0] + (fA*c)*v[1] + (fB*c + fB*rA)*v[2] + (fA*fB*c)*v[3];
    w[2] = v[0] + fA*v[1] + fB*v[2] + fA*fB*v[3];

    return w;
}

template <typename T>
std::array<T, 8> operatorD(std::array<T, 4>& v, std::array<T, 6>& labels)
{
    /*
     * Apply operator D at the j-th rung of the grid graph.
     */
    std::array<T, 8> w;
    T fA = labels[0];
    T rA = labels[1];
    T fB = labels[2];
    T rB = labels[3];
    T c = labels[4];
    T d = labels[5];
    w[0] = (d + rB)*v[0] + (fB*d)*v[3];
    w[1] = (d + rB)*v[1] + (fA*d + fA*rB)*v[2];
    w[2] = (c + rA)*v[0] + (fB*c + fB*rA)*v[3];
    w[3] = (c + rA)*v[1] + (fA*c)*v[2];
    w[4] = fA*v[0] + (fA*fB)*v[2];
    w[5] = fB*v[1] + (fA*fB)*v[3];
    w[6] = v[0] + fB*v[3];
    w[7] = v[1] + fA*v[2];

    return w;
}

template <typename T>
std::array<T, 8> operatorE(std::array<T, 3>& v, std::array<T, 6>& labels)
{
    /*
     * Apply operator E at the j-th rung of the grid graph.
     */
    std::array<T, 8> w;
    T fA = labels[0];
    T rA = labels[1];
    T fB = labels[2];
    T rB = labels[3];
    T c = labels[4];
    T d = labels[5];
    w[0] = (d + rB)*v[0] + (fB*d)*v[2];
    w[1] = (c + rA)*v[0] + (fB*c + fB*rA)*v[2];
    w[2] = (d + rB)*v[0] + (fA*d + fA*rB)*v[1];
    w[3] = (c + rA)*v[0] + (fA*c)*v[1];
    w[4] = rB*v[0] + (fA*rB)*v[1];
    w[5] = rA*v[0] + (fB*rA)*v[2];
    w[6] = v[0] + fB*v[2];
    w[7] = v[0] + fA*v[1];

    return w;
}

template <typename T>
class GridGraph : public LabeledDigraph<T>
{
    /*
     * An implementation of the two-state grid graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        unsigned N;    // Length of the graph

        // Canonical ordering of the nodes 
        std::vector<Node*> order;

        // The A <-> B edge labels for the zeroth rung of the graph
        std::array<T, 2> start;

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<T, 6> > rung_labels;

    public:
        GridGraph() : LabeledDigraph<T>()
        {
            /*
             * Trivial constructor with length zero; set edge labels to unity.
             */
            // Add nodes ...
            this->N = 0;
            Node* node_A = this->addNode("A0");
            Node* node_B = this->addNode("B0");
            this->order.push_back(node_A);
            this->order.push_back(node_B);

            // ... and edges 
            this->addEdge("A0", "B0");
            this->addEdge("B0", "A0");
            this->start[0] = 1.0;
            this->start[1] = 1.0;
        }

        GridGraph(unsigned N) : LabeledDigraph<T>()
        {
            /*
             * Trivial constructor with given length; set edge labels to unity.
             */
            // Add the zeroth nodes ...
            this->N = N;
            Node* node_A = this->addNode("A0");
            Node* node_B = this->addNode("B0");
            this->order.push_back(node_A);
            this->order.push_back(node_B);

            // ... and edges 
            this->addEdge("A0", "B0");
            this->addEdge("B0", "A0");
            this->start[0] = 1.0;
            this->start[1] = 1.0;

            for (unsigned i = 0; i < N; ++i)
            {
                // Add the (i+1)-th nodes ...
                std::stringstream sai, sbi, saj, sbj;
                sai << "A" << i;
                sbi << "B" << i;
                saj << "A" << i + 1;
                sbj << "B" << i + 1;
                node_A = this->addNode(saj.str());
                node_B = this->addNode(sbj.str());
                this->order.push_back(node_A);
                this->order.push_back(node_B);

                // ... and edges
                this->addEdge(sai.str(), saj.str());
                this->addEdge(saj.str(), sai.str());
                this->addEdge(sbi.str(), sbj.str());
                this->addEdge(sbj.str(), sbi.str());
                this->addEdge(saj.str(), sbj.str());
                this->addEdge(sbj.str(), saj.str());
                std::array<T, 6> labels = {1, 1, 1, 1, 1, 1};
                this->rung_labels.push_back(labels);
            }
        }

        ~GridGraph()
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

        void setStartLabels(T A_to_B, T B_to_A)
        {
            /*
             * Set the edge labels of the zeroth rung of the graph to the
             * given values.
             */
            this->setEdgeLabel("A0", "B0", A_to_B);
            this->setEdgeLabel("B0", "A0", B_to_A);
            this->start[0] = A_to_B;
            this->start[1] = B_to_A;
        }

        void addRung(std::array<T, 6> labels)
        {
            /*
             * Add new rung onto the end of the graph, keeping track of the
             * six new edge labels. 
             */
            // Add the new nodes ...
            this->N++;
            std::stringstream sai, sbi, saj, sbj;
            sai << "A" << this->N - 1;
            sbi << "B" << this->N - 1;
            saj << "A" << this->N;
            sbj << "B" << this->N;
            Node* node_A = this->addNode(saj.str());
            Node* node_B = this->addNode(sbj.str());
            this->order.push_back(node_A);
            this->order.push_back(node_B);

            // ... and edges
            // The canonical ordering for the edge labels in each rung is:
            // (A,i) -> (A,j), (A,j) -> (A,i), (B,i) -> (B,j), (B,j) -> (B,i), (A,j) -> (B,j), (B,j) -> (A,j)
            this->addEdge(sai.str(), saj.str(), labels[0]);
            this->addEdge(saj.str(), sai.str(), labels[1]);
            this->addEdge(sbi.str(), sbj.str(), labels[2]);
            this->addEdge(sbj.str(), sbi.str(), labels[3]);
            this->addEdge(saj.str(), sbj.str(), labels[4]);
            this->addEdge(sbj.str(), saj.str(), labels[5]);
            this->rung_labels.push_back(labels);
        }

        void setRungLabels(unsigned i, std::array<T, 6> labels)
        {
            /*
             * Set the edge labels for the i-th rung to the given values. 
             */
            this->rung_labels[i] = labels;
            std::stringstream sai, sbi, saj, sbj;
            sai << "A" << i;       // i == 0 means edges between A0/B0 and A1/B1, etc.
            sbi << "B" << i;
            saj << "A" << i + 1;
            sbj << "B" << i + 1;

            // The canonical ordering for the edge labels in each rung is:
            // (A,i) -> (A,j), (A,j) -> (A,i), (B,i) -> (B,j), (B,j) -> (B,i), (A,j) -> (B,j), (B,j) -> (A,j)
            this->setEdgeLabel(sai.str(), saj.str(), labels[0]);
            this->setEdgeLabel(saj.str(), sai.str(), labels[1]);
            this->setEdgeLabel(sbi.str(), sbj.str(), labels[2]);
            this->setEdgeLabel(sbj.str(), sbi.str(), labels[3]);
            this->setEdgeLabel(saj.str(), sbj.str(), labels[4]);
            this->setEdgeLabel(sbj.str(), saj.str(), labels[5]);
        }

        std::tuple<std::vector<T>,
                   std::vector<T>,
                   T,
                   std::vector<T>,
                   T> computeAllForestWeights()
        {
            /*
             * Compute the following table of spanning forest weights:
             *
             * 0) Trees rooted at (Y,i) for Y = A,B and i = 0,...,N
             * 1) Forests rooted at (Y,i), (B,N) for Y = A,B and i = 0,...,N-1
             *    with path (A,0) -> (Y,i)
             * 2) Forests rooted at (A,N), (B,N) with path (A,0) -> (A,N)
             * 3) Forests rooted at (A,0), (B,N) with path (Y,i) -> (A,0)
             *    for Y = A,B and i = 0,...,N-1
             * 4) Forests rooted at (A,0), (B,N) with path (A,N) -> (A,0)
             */
            // Require that N >= 1
            if (this->N < 1)
                throw std::runtime_error("this->N < 1");

            // All spanning tree weights for N = 1
            T weight_A0 = this->rung_labels[0][2] * this->rung_labels[0][5] * this->rung_labels[0][1] +
                this->start[1] * this->rung_labels[0][3] * this->rung_labels[0][4] +
                this->start[1] * this->rung_labels[0][3] * this->rung_labels[0][1] +
                this->start[1] * this->rung_labels[0][1] * this->rung_labels[0][5];
            T weight_B0 = this->start[0] * this->rung_labels[0][1] * this->rung_labels[0][5] + 
                this->rung_labels[0][3] * this->rung_labels[0][4] * this->rung_labels[0][0] +
                this->start[0] * this->rung_labels[0][3] * this->rung_labels[0][1] +
                this->start[0] * this->rung_labels[0][3] * this->rung_labels[0][4];
            T weight_A1 = this->rung_labels[0][0] * this->start[1] * this->rung_labels[0][3] +
                this->rung_labels[0][5] * this->rung_labels[0][2] * this->start[0] +
                this->rung_labels[0][0] * this->rung_labels[0][5] * this->start[1] +
                this->rung_labels[0][0] * this->rung_labels[0][5] * this->rung_labels[0][2];
            T weight_B1 = this->rung_labels[0][2] * this->rung_labels[0][1] * this->start[0] +
                this->rung_labels[0][4] * this->rung_labels[0][0] * this->start[1] +
                this->rung_labels[0][2] * this->rung_labels[0][4] * this->start[0] +
                this->rung_labels[0][2] * this->rung_labels[0][4] * this->rung_labels[0][0];

            // All required spanning forest weights for N = 1 with path (A,1) -> (A,0)
            // or (B,1) -> (A,0) or (A,1) -> (B,0) or (B,1) -> (B,0)
            T weight_A0_A1_with_path_B1_to_A0 = this->start[1] * this->rung_labels[0][3];
            T weight_A0_B1_with_path_A1_to_A0 = this->rung_labels[0][1] * this->rung_labels[0][2] +
                this->start[1] * this->rung_labels[0][1];
            T weight_B0_A1_with_path_B1_to_B0 = this->rung_labels[0][0] * this->rung_labels[0][3] +
                this->start[0] * this->rung_labels[0][3];
            T weight_B0_B1_with_path_A1_to_B0 = this->start[0] * this->rung_labels[0][1];

            // Spanning forest weights for N = 1 with roots (A,1), (B,1)
            T weight_A1_B1 = this->rung_labels[0][0] * this->rung_labels[0][2] +
                this->start[1] * this->rung_labels[0][0] +
                this->start[0] * this->rung_labels[0][2];

            // All required spanning forest weights for N = 1 with paths from (A,0)
            T weight_A0_A1_with_path_A0_to_A0 = this->start[1] * this->rung_labels[0][5] +
                this->rung_labels[0][2] * this->rung_labels[0][5] +
                this->start[1] * this->rung_labels[0][3];
            T weight_A0_B1_with_path_A0_to_A0 = this->start[1] * this->rung_labels[0][4] +
                this->rung_labels[0][1] * this->rung_labels[0][2] +
                this->rung_labels[0][2] * this->rung_labels[0][4] +
                this->start[1] * this->rung_labels[0][1];
            T weight_B0_A1_with_path_A0_to_B0 = this->start[0] * this->rung_labels[0][5] +
                this->start[0] * this->rung_labels[0][3];
            T weight_B0_B1_with_path_A0_to_B0 = this->start[0] * this->rung_labels[0][4] +
                this->start[0] * this->rung_labels[0][1];
            T weight_A1_B1_with_path_A0_to_A1 = this->rung_labels[0][0] * this->rung_labels[0][2] +
                this->start[1] * this->rung_labels[0][0];
            T weight_A1_B1_with_path_A0_to_B1 = this->start[0] * this->rung_labels[0][2];
            T weight_A0_A1_B1_with_path_A0_to_A0 = this->start[1] + this->rung_labels[0][2];
            T weight_B0_A1_B1_with_path_A0_to_B0 = this->start[0];

            // All required spanning forest weights for N = 1 with paths to (A,0)
            T weight_A0_A1_with_path_B0_to_A0 = this->start[1] * this->rung_labels[0][5] +
                this->start[1] * this->rung_labels[0][3];
            T weight_A0_B1_with_path_B0_to_A0 = this->start[1] * this->rung_labels[0][4] + 
                this->start[1] * this->rung_labels[0][1];
            T weight_A0_A1_B1_with_path_B0_to_A0 = this->start[1];

            // Initialize vectors of spanning forest weights 
            std::vector<std::array<T, 3> > vA;
            vA.push_back({
                weight_A0,
                weight_A0_A1_with_path_B1_to_A0,
                weight_A0_B1_with_path_A1_to_A0
            });
            vA.push_back({
                weight_B0,
                weight_B0_A1_with_path_B1_to_B0,
                weight_B0_B1_with_path_A1_to_B0
            });
            std::array<T, 3> vB = {
                weight_A1,
                weight_B1,
                weight_A1_B1
            };
            std::vector<std::array<T, 4> > vC;
            vC.push_back({
                weight_A0,
                weight_A0_A1_with_path_A0_to_A0,
                weight_A0_B1_with_path_A0_to_A0,
                weight_A0_A1_B1_with_path_A0_to_A0
            });
            vC.push_back({
                weight_B0,
                weight_B0_A1_with_path_A0_to_B0,
                weight_B0_B1_with_path_A0_to_B0,
                weight_B0_A1_B1_with_path_A0_to_B0
            });
            std::array<T, 4> vD = {
                weight_A1,
                weight_B1,
                weight_A1_B1_with_path_A0_to_B1,
                weight_A1_B1_with_path_A0_to_A1
            };
            std::vector<std::array<T, 4> > vC1;
            vC1.push_back({
                weight_A0,
                weight_A0_A1_with_path_A0_to_A0,
                weight_A0_B1_with_path_A0_to_A0,
                weight_A0_A1_B1_with_path_A0_to_A0
            });
            vC1.push_back({
                weight_A0,
                weight_A0_A1_with_path_B0_to_A0,
                weight_A0_B1_with_path_B0_to_A0,
                weight_A0_A1_B1_with_path_B0_to_A0
            });
            std::array<T, 3> vE = {
                weight_A0,
                weight_A0_A1_with_path_B1_to_A0,
                weight_A0_B1_with_path_A1_to_A0
            };

            for (unsigned i = 1; i < this->N; ++i)  
            {
                // Apply operator A
                for (unsigned j = 0; j < vA.size(); ++j)
                {
                    std::array<T, 3> wA_j = operatorA(vA[j], this->rung_labels[i]);
                    vA[j] = wA_j;
                }

                // Apply operator B
                std::array<T, 9> wB = operatorB(vB, this->rung_labels[i]);
                vA.push_back({wB[0], wB[4], wB[5]});
                vA.push_back({wB[1], wB[6], wB[7]});
                vB[0] = wB[2];
                vB[1] = wB[3];
                vB[2] = wB[8];

                // Apply operator C 
                for (unsigned j = 0; j < vC.size(); ++j)
                {
                    std::array<T, 3> wC_j = operatorC(vC[j], this->rung_labels[i]);
                    vC[j][0] = vA[j][0];
                    vC[j][1] = wC_j[0];
                    vC[j][2] = wC_j[1];
                    vC[j][3] = wC_j[2];
                }

                // Apply operator D
                std::array<T, 8> wD = operatorD(vD, this->rung_labels[i]);
                vC.push_back({vA[vA.size()-2][0], wD[0], wD[2], wD[6]});
                vC.push_back({vA[vA.size()-1][0], wD[1], wD[3], wD[7]});
                vD[0] = vB[0];
                vD[1] = vB[1];
                vD[2] = wD[4];
                vD[3] = wD[5];

                // Apply operator C again
                for (unsigned j = 0; j < vC1.size(); ++j)
                {
                    std::array<T, 3> wC1_j = operatorC(vC1[j], this->rung_labels[i]);
                    vC1[j][0] = vA[j][0];
                    vC1[j][1] = wC1_j[0];
                    vC1[j][2] = wC1_j[1];
                    vC1[j][3] = wC1_j[2];
                }

                // Apply operator E 
                std::array<T, 8> wE = operatorE(vE, this->rung_labels[i]);
                vC1.push_back({vA[vA.size()-2][0], wE[0], wE[1], wE[6]});
                vC1.push_back({vA[vA.size()-1][0], wE[2], wE[3], wE[7]});
                vE[0] = vA[0][0];
                vE[1] = wE[4];
                vE[2] = wE[5];
            }

            // 0) Write all spanning tree weights 
            std::vector<T> tree_weights;
            for (auto&& arr : vA)
                tree_weights.push_back(arr[0]);
            tree_weights.push_back(vB[0]);
            tree_weights.push_back(vB[1]);

            // 1) Write all weights of spanning forests rooted at (Y,i), (B,N)
            // with path (A,0) -> (Y,i)
            std::vector<T> forest_weights_Yi_BN_with_path_A0_to_Yi;
            for (auto&& arr : vC)
                forest_weights_Yi_BN_with_path_A0_to_Yi.push_back(arr[2]);

            // 2) Weight of spanning forests rooted at (A,N), (B,N) with path 
            // (A,0) -> (A,N)
            T forest_weight_AN_BN_with_path_A0_to_AN = vD[3];

            // 3) Write all weights of spanning forests rooted at (A,0), (B,N)
            // with path (Y,i) -> (A,0)
            std::vector<T> forest_weights_A0_BN_with_path_Yi_to_A0;
            for (auto&& arr : vC1)
                forest_weights_A0_BN_with_path_Yi_to_A0.push_back(arr[2]);

            // 4) Weight of spanning forests rooted at (A,0), (B,N) with path
            // (A,N) -> (A,0)
            T forest_weight_A0_BN_with_path_AN_to_A0 = vE[2];

            // Return all accumulated data
            return std::tie(
                tree_weights, 
                forest_weights_Yi_BN_with_path_A0_to_Yi,
                forest_weight_AN_BN_with_path_A0_to_AN,
                forest_weights_A0_BN_with_path_Yi_to_A0,
                forest_weight_A0_BN_with_path_AN_to_A0
            );
        }

        std::pair<T, T> computeExitStats(T exit_rate_lower_prob = 1, T exit_rate_upper_prob = 1,
                                         T exit_rate_lower_time = 1, T exit_rate_upper_time = 1)
        {
            /*
             * Compute the probability of upper exit (from (B,N)) and the 
             * rate of lower exit (from (A,0)). 
             */
            // Compute all spanning forest weights 
            std::tuple<std::vector<T>,
                       std::vector<T>,
                       T,
                       std::vector<T>,
                       T> weights = this->computeAllForestWeights();

            // Probability of upper exit is given by ...
            T weight_A0 = std::get<0>(weights)[0];
            T weight_AN = std::get<0>(weights)[2*(this->N+1)-2];
            T weight_BN = std::get<0>(weights)[2*(this->N+1)-1];
            T weight_A0_BN = std::get<1>(weights)[0];
            T prob_upper_exit = (weight_BN * exit_rate_upper_prob) / (
                weight_A0 * exit_rate_lower_prob +
                weight_BN * exit_rate_upper_prob +
                weight_A0_BN * exit_rate_lower_prob * exit_rate_upper_prob
            );

            // Mean first passage time to lower exit is given by ...
            T numer = 0;
            for (unsigned i = 0; i < 2*this->N; ++i)
            {
                T weight_Yi = std::get<0>(weights)[i];
                T weight_Yi_BN_with_path_A0_to_Yi = std::get<1>(weights)[i];
                T weight_A0_BN_with_path_Yi_to_A0 = std::get<3>(weights)[i];
                numer += (
                    (weight_Yi + weight_Yi_BN_with_path_A0_to_Yi * exit_rate_upper_time) *
                    (weight_A0 * exit_rate_lower_time + weight_A0_BN_with_path_Yi_to_A0 * exit_rate_lower_time * exit_rate_upper_time)
                );
            }
            T weight_AN_BN_with_path_A0_to_AN = std::get<2>(weights);
            T weight_A0_BN_with_path_AN_to_A0 = std::get<4>(weights);
            numer += (
                (weight_AN + weight_AN_BN_with_path_A0_to_AN * exit_rate_upper_time) *
                (weight_A0 * exit_rate_lower_time + weight_A0_BN_with_path_AN_to_A0 * exit_rate_lower_time * exit_rate_upper_time)
            );
            numer += (weight_BN * weight_A0 * exit_rate_lower_time);

            // Take the reciprocal of the mean first passage time to get 
            // the rate of lower exit 
            T rate_lower_exit = (
                weight_A0 * exit_rate_lower_time +
                weight_A0_BN * exit_rate_lower_time * exit_rate_upper_time
            ) * (
                weight_A0 * exit_rate_lower_time +
                weight_A0_BN * exit_rate_lower_time * exit_rate_upper_time +
                weight_BN * exit_rate_upper_time
            ) / numer;

            return std::make_pair(prob_upper_exit, rate_lower_exit);
        } 
};

#endif 
