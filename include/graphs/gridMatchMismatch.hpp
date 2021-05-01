#ifndef GRID_MATCH_MISMATCH_GRAPH_HPP
#define GRID_MATCH_MISMATCH_GRAPH_HPP

#include <iostream>
#include <sstream>
#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <digraph.hpp>
#include "grid.hpp"

/* 
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     5/1/2021
 */
template <typename T>
class GridMatchMismatchGraph : public GridGraph<T>
{
    /*
     * An implementation of the two-state grid graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        std::vector<bool> pattern;         // Pattern of matches (true) and mismatches (false)
        std::array<T, 2> match_labels;     // Match forward/reverse labels
        std::array<T, 2> mismatch_labels;  // Mismatch forward/reverse labels 

    public:
        GridMatchMismatchGraph() : GridGraph<T>()
        {
            /*
             * Trivial constructor with length zero; set edge labels to unity.
             */
            // Add match/mismatch forward/reverse labels of 1
            this->match_labels[0] = 1.0;
            this->match_labels[1] = 1.0;
            this->mismatch_labels[0] = 1.0;
            this->mismatch_labels[1] = 1.0;

            // Begin with sequence pattern of length zero 
        }

        GridMatchMismatchGraph(unsigned N) : GridGraph<T>(N)
        {
            /*
             * Trivial constructor with given length; set edge labels to unity.
             */
            // Add match/mismatch forward/reverse labels of 1
            this->match_labels[0] = 1.0;
            this->match_labels[1] = 1.0;
            this->mismatch_labels[0] = 1.0;
            this->mismatch_labels[1] = 1.0;

            // Begin with sequence pattern of N matches 
            for (unsigned i = 0; i < N; ++i)
                this->pattern.push_back(true);
        }

        ~GridMatchMismatchGraph()
        {
            /*
             * Trivial destructor.
             */
        }

        void setMatchForwardLabel(T label)
        {
            /*
             * Update the match forward transition label.
             */
            this->match_labels[0] = label; 
        }

        void setMatchReverseLabel(T label)
        {
            /*
             * Update the match reverse transition label.
             */
            this->match_labels[1] = label; 
        }

        void setMismatchForwardLabel(T label)
        {
            /*
             * Update the mismatch forward transition label.
             */
            this->mismatch_labels[0] = label; 
        }

        void setMismatchReverseLabel(T label)
        {
            /*
             * Update the mismatch reverse transition label.
             */
            this->mismatch_labels[1] = label; 
        }

        void addRung(bool match)
        {
            /*
             * Add new rung onto the end of the graph, keeping track of the
             * six new edge labels. 
             */
            // Add the new nodes ...
            this->N++;
            this->pattern.push_back(match);
            std::stringstream sai, sbi, saj, sbj;
            sai << "A" << this->N - 1;
            sbi << "B" << this->N - 1;
            saj << "A" << this->N;
            sbj << "B" << this->N;
            Node* node_A = this->addNode(saj.str());
            Node* node_B = this->addNode(sbj.str());

            // ... and edges
            // The canonical ordering for the edge labels in each rung is:
            // (A,i) -> (A,j), (A,j) -> (A,i), (B,i) -> (B,j), (B,j) -> (B,i), (A,j) -> (B,j), (B,j) -> (A,j)
            T label_forward = (match) ? this->match_labels[0] : this->mismatch_labels[0];
            T label_reverse = (match) ? this->match_labels[1] : this->mismatch_labels[1];
            this->addEdge(sai.str(), saj.str(), label_forward);
            this->addEdge(saj.str(), sai.str(), label_reverse);
            this->addEdge(sbi.str(), sbj.str(), label_forward);
            this->addEdge(sbj.str(), sbi.str(), label_reverse);
            this->addEdge(saj.str(), sbj.str(), this->start[0]);
            this->addEdge(sbj.str(), saj.str(), this->start[1]);
            this->rung_labels.emplace_back(
                {label_forward, label_reverse, label_forward, label_reverse, this->start[0], this->start[1]}
            );
        }

        void setRungLabels(unsigned i, bool match) 
        {
            /*
             * Set the edge labels for the i-th rung to the given values. 
             */
            this->pattern[i] = match;
            std::stringstream sai, sbi, saj, sbj;
            sai << "A" << i;       // i == 0 means edges between A0/B0 and A1/B1, etc.
            sbi << "B" << i;
            saj << "A" << i + 1;
            sbj << "B" << i + 1;

            // The canonical ordering for the edge labels in each rung is:
            // (A,i) -> (A,j), (A,j) -> (A,i), (B,i) -> (B,j), (B,j) -> (B,i), (A,j) -> (B,j), (B,j) -> (A,j)
            T label_forward = (match) ? this->match_labels[0] : this->mismatch_labels[0];
            T label_reverse = (match) ? this->match_labels[1] : this->mismatch_labels[1];
            this->setEdgeLabel(sai.str(), saj.str(), label_forward);
            this->setEdgeLabel(saj.str(), sai.str(), label_reverse);
            this->setEdgeLabel(sbi.str(), sbj.str(), label_forward);
            this->setEdgeLabel(sbj.str(), sbi.str(), label_reverse);
            this->setEdgeLabel(saj.str(), sbj.str(), this->start[0]);
            this->setEdgeLabel(sbj.str(), saj.str(), this->start[1]);
            this->rung_labels[i][0] = label_forward;
            this->rung_labels[i][1] = label_reverse;
            this->rung_labels[i][2] = label_forward;
            this->rung_labels[i][3] = label_reverse;
            this->rung_labels[i][4] = this->start[0];
            this->rung_labels[i][5] = this->start[1];
        }

        std::tuple<std::vector<T>, std::vector<T>, T, std::vector<T>, T> computeAllForestWeights()
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

            // Define match and mismatch constants
            T match_e = this->match_labels[1] * (this->start[1] + this->start[0] + this->match_labels[1]);
            T mismatch_e = this->mismatch_labels[1] * (this->start[1] + this->start[0] + this->mismatch_labels[1]);
            T match_g = this->match_labels[0] * this->match_labels[1];
            T mismatch_g = this->mismatch_labels[0] * this->mismatch_labels[1];
            T match_h = match_g; 
            T mismatch_h = mismatch_g;
            T match_k = this->start[0] + this->match_labels[1];
            T mismatch_k = this->start[0] + this->mismatch_labels[1]; 
            T match_l = this->start[1] + this->match_labels[1];
            T mismatch_l = this->start[1] + this->mismatch_labels[1];
            T e, g, h, k, l, f, r;

            for (unsigned i = 1; i < this->N; ++i)  
            {
                bool match_i = this->pattern[i];

                // Apply operator A
                for (unsigned j = 0; j < vA.size(); ++j)
                {
                    // Choose whether to use the match or mismatch constants
                    bool match_j = this->pattern[j];
                    e = (match_j) ? match_e : mismatch_e; 
                    g = (match_j) ? match_g : mismatch_g;
                    h = (match_j) ? match_h : mismatch_h;
                    k = (match_j) ? match_k : mismatch_k;
                    l = (match_j) ? match_l : mismatch_l;
                    f = (match_j) ? this->match_labels[0] : this->mismatch_labels[0]; 
                    r = (match_j) ? this->match_labels[1] : this->mismatch_labels[1];

                    // Apply operator A 
                    std::array<T, 3> wA, yA; 
                    wA[0] = vA[j][0];
                    wA[1] = g * vA[j][1];
                    wA[2] = h * vA[j][2];
                    yA[0] = e * wA[0] + this->start[0] * wA[1] + this->start[1] * wA[2]; 
                    yA[1] = r * wA[0] + wA[1]; 
                    yA[2] = r * wA[0] + wA[2]; 
                    vA[j][0] = yA[0];
                    vA[j][1] = yA[1];
                    vA[j][2] = yA[2];
                }

                // Apply operator B
                e = (match_i) ? match_e : mismatch_e; 
                g = (match_i) ? match_g : mismatch_g;
                h = (match_i) ? match_h : mismatch_h;
                k = (match_i) ? match_k : mismatch_k;
                l = (match_i) ? match_l : mismatch_l;
                f = (match_i) ? this->match_labels[0] : this->mismatch_labels[0]; 
                r = (match_i) ? this->match_labels[1] : this->mismatch_labels[1];
                std::array<T, 3> vA_first, vA_second, vB_next;
                vA_first[0] = e * vB[0] + h * this->start[1] * vB[2];
                vA_first[1] = r * vB[0];
                vA_first[2] = r * vB[0] + h * vB[2];
                vA_second[0] = e * vB[1] + g * this->start[0] * vB[2];
                vA_second[1] = r * vB[1] + g * vB[2];
                vA_second[2] = r * vB[1]; 
                vB_next[0] = f * (l * vB[0] + this->start[1] * (vB[1] + f * vB[2]));
                vB_next[1] = f * (this->start[0] * (vB[0] + f * vB[2]) + k * vB[1]);
                vB_next[2] = f * (vB[0] + vB[1] + f * vB[2]); 
                vA.push_back(vA_first);
                vA.push_back(vA_second);
                vB[0] = vB_next[0];
                vB[1] = vB_next[1];
                vB[2] = vB_next[2];

                // Apply operator D (before C!)
                std::array<T, 4> vC_first, vC_second;
                std::array<T, 2> vD_next;
                T u0 = vD[0] + f * vD[3];
                T u1 = vD[1] + f * vD[2]; 
                vC_first[0] = vA_first[0];
                vC_first[1] = l * vD[0] + f * this->start[1] * vD[3];
                vC_first[2] = k * u0;
                vC_first[3] = u0;
                vC_second[0] = vA_second[0];
                vC_second[1] = l * u1;
                vC_second[2] = k * vD[1] + f * this->start[0] * vD[2];
                vC_second[3] = u1;
                vD_next[0] = f * (vD[0] + f * vD[2]); 
                vD_next[1] = f * (vD[1] + f * vD[3]);
                vD[0] = vB[0];
                vD[1] = vB[1];
                vD[2] = vD_next[0];
                vD[3] = vD_next[1];

                // Apply operator E (before C!)
                std::array<T, 4> vC1_first, vC1_second;
                std::array<T, 2> vE_next;
                T x1 = vE[0] + f * vE[1]; 
                T x2 = vE[0] + f * vE[2];
                vC1_first[0] = vA_first[0];
                vC1_first[1] = l * vE[0] + f * this->start[1] * vE[2];
                vC1_first[2] = k * x2;
                vC1_first[3] = x2;
                vC1_second[0] = vA_second[0];
                vC1_second[1] = l * x1;
                vC1_second[2] = k * vE[0] + f * this->start[0] * vE[1];
                vC1_second[3] = x1;
                vE_next[0] = r * x1;
                vE_next[1] = r * x2;
                vE[0] = vA[0][0];
                vE[1] = vE_next[0];
                vE[2] = vE_next[1];

                // Apply operator C 
                for (unsigned j = 0; j < vC.size(); ++j)
                {
                    // Choose whether to use the match or mismatch constants
                    bool match_j = this->pattern[j];
                    e = (match_j) ? match_e : mismatch_e; 
                    g = (match_j) ? match_g : mismatch_g;
                    h = (match_j) ? match_h : mismatch_h;
                    k = (match_j) ? match_k : mismatch_k;
                    l = (match_j) ? match_l : mismatch_l;
                    f = (match_j) ? this->match_labels[0] : this->mismatch_labels[0]; 
                    r = (match_j) ? this->match_labels[1] : this->mismatch_labels[1];

                    // Apply operator C 
                    std::array<T, 3> wC; 
                    wC[0] = l * vC[j][0] + f * (l * vC[j][1] + this->start[1] * (vC[j][2] + f * vC[j][3]));
                    wC[1] = k * vC[j][0] + f * (this->start[0] * (vC[j][1] + f * vC[j][3]) + k * vC[j][2]);
                    wC[2] = vC[j][0] + f * (vC[j][1] + vC[j][2] + f * vC[j][3]);
                    vC[j][0] = vA[j][0];
                    vC[j][1] = wC[0];
                    vC[j][2] = wC[1];
                    vC[j][3] = wC[2]; 
                }

                // Apply operator C again
                for (unsigned j = 0; j < vC1.size(); ++j)
                {
                    // Choose whether to use the match or mismatch constants
                    bool match_j = this->pattern[j];
                    e = (match_j) ? match_e : mismatch_e; 
                    g = (match_j) ? match_g : mismatch_g;
                    h = (match_j) ? match_h : mismatch_h;
                    k = (match_j) ? match_k : mismatch_k;
                    l = (match_j) ? match_l : mismatch_l;
                    f = (match_j) ? this->match_labels[0] : this->mismatch_labels[0]; 
                    r = (match_j) ? this->match_labels[1] : this->mismatch_labels[1];

                    // Apply operator C 
                    std::array<T, 3> wC1; 
                    wC1[0] = l * vC1[j][0] + f * (l * vC1[j][1] + this->start[1] * (vC1[j][2] + f * vC1[j][3]));
                    wC1[1] = k * vC1[j][0] + f * (this->start[0] * (vC1[j][1] + f * vC1[j][3]) + k * vC1[j][2]);
                    wC1[2] = vC1[j][0] + f * (vC1[j][1] + vC1[j][2] + f * vC1[j][3]);
                    vC1[j][0] = vA[j][0];
                    vC1[j][1] = wC1[0];
                    vC1[j][2] = wC1[1];
                    vC1[j][3] = wC1[2]; 
                }

                // Append the new vectors to be added to vC and vC1
                vC.push_back(vC_first);
                vC.push_back(vC_second);
                vC1.push_back(vC1_first);
                vC1.push_back(vC1_second);
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

        std::tuple<T, T, T> computeExitStats(T exit_rate_lower_prob = 1, T exit_rate_upper_prob = 1,
                                             T exit_rate_lower_time = 1, T exit_rate_upper_time = 1)
        {
            /*
             * Compute the probability of upper exit, the rate of lower exit,
             * and the rate of upper exit, all starting from (A,0). 
             */
            // Compute all spanning forest weights 
            std::tuple<std::vector<T>, std::vector<T>, T, std::vector<T>, T> weights = this->computeAllForestWeights();

            T weight_A0 = std::get<0>(weights)[0];
            T weight_AN = std::get<0>(weights)[2*(this->N+1)-2];
            T weight_BN = std::get<0>(weights)[2*(this->N+1)-1];
            T weight_A0_BN = std::get<1>(weights)[0];
            
            // Get weight of all forests rooted at exit vertices
            T two_forest_weight = (
                weight_A0 * exit_rate_lower_prob +
                weight_BN * exit_rate_upper_prob +
                weight_A0_BN * exit_rate_lower_prob * exit_rate_upper_prob
            );

            // Get weight of all forests rooted at exit vertices with path (A,0) -> lower exit
            T two_forest_weight_A0_to_lower = (
                weight_A0 * exit_rate_lower_prob +
                weight_A0_BN * exit_rate_lower_prob * exit_rate_upper_prob
            );

            // Get weight of all forests rooted at exit vertices with path (A,0) -> upper
            T two_forest_weight_A0_to_upper = two_forest_weight - two_forest_weight_A0_to_lower;

            // Probability of upper exit is given by ...
            T prob_upper_exit = (weight_BN * exit_rate_upper_prob) / two_forest_weight;

            // Re-compute weight of all forests rooted at exit vertices
            two_forest_weight = (
                weight_A0 * exit_rate_lower_time +
                weight_BN * exit_rate_upper_time +
                weight_A0_BN * exit_rate_lower_time * exit_rate_upper_time
            );

            // Re-compute weight of all forests rooted at exit vertices with path (A,0) -> lower exit
            two_forest_weight_A0_to_lower = (
                weight_A0 * exit_rate_lower_time +
                weight_A0_BN * exit_rate_lower_time * exit_rate_upper_time
            );

            // Re-compute weight of all forests rooted at exit vertices with path (A,0) -> upper
            two_forest_weight_A0_to_upper = two_forest_weight - two_forest_weight_A0_to_lower;
            
            // Mean first passage times to lower/upper exit are given by ...
            T numer_lower_exit = 0;
            T numer_upper_exit = 0;
            for (unsigned i = 0; i < 2*this->N; ++i)
            {
                T weight_Yi = std::get<0>(weights)[i];
                T weight_Yi_BN_with_path_A0_to_Yi = std::get<1>(weights)[i];
                T weight_A0_BN_with_path_Yi_to_A0 = std::get<3>(weights)[i];
               
                // Get weight of all 2-forests rooted at exit vertices with 
                // path (Y,i) -> lower exit
                T two_forest_weight_Yi_to_lower = (
                    weight_A0 * exit_rate_lower_time +
                    weight_A0_BN_with_path_Yi_to_A0 * exit_rate_lower_time * exit_rate_upper_time
                );

                // Get weight of all 2-forests rooted at exit vertices with 
                // path (Y,i) -> upper exit
                T two_forest_weight_Yi_to_upper = two_forest_weight - two_forest_weight_Yi_to_lower;

                // Get weight of all 3-forests rooted at exit vertices and (Y,i)
                // with path (A,0) -> (Y,i)
                T three_forest_weight_A0_to_Yi = (
                    weight_Yi + weight_Yi_BN_with_path_A0_to_Yi * exit_rate_upper_time
                );

                // Get contribution to numerators of mean first passage time
                numer_lower_exit += (three_forest_weight_A0_to_Yi * two_forest_weight_Yi_to_lower);
                numer_upper_exit += (three_forest_weight_A0_to_Yi * two_forest_weight_Yi_to_upper);
            }

            // Get contribution to numerators for (Y,i) = (A,N)
            T weight_AN_BN_with_path_A0_to_AN = std::get<2>(weights);
            T weight_A0_BN_with_path_AN_to_A0 = std::get<4>(weights);
            T two_forest_weight_AN_to_lower = (
                weight_A0 * exit_rate_lower_time +
                weight_A0_BN_with_path_AN_to_A0 * exit_rate_lower_time * exit_rate_upper_time
            );
            T two_forest_weight_AN_to_upper = two_forest_weight - two_forest_weight_AN_to_lower;
            T three_forest_weight_A0_to_AN = (
                weight_AN + weight_AN_BN_with_path_A0_to_AN * exit_rate_upper_time
            );
            numer_lower_exit += (three_forest_weight_A0_to_AN * two_forest_weight_AN_to_lower);
            numer_upper_exit += (three_forest_weight_A0_to_AN * two_forest_weight_AN_to_upper);

            // Get contribution to numerators for (Y,i) = (B,N)
            T two_forest_weight_BN_to_lower = weight_A0 * exit_rate_lower_time;
            T two_forest_weight_BN_to_upper = two_forest_weight - two_forest_weight_BN_to_lower;
            T three_forest_weight_A0_to_BN = weight_BN;
            numer_lower_exit += (three_forest_weight_A0_to_BN * two_forest_weight_BN_to_lower);
            numer_upper_exit += (three_forest_weight_A0_to_BN * two_forest_weight_BN_to_upper);

            // Take the reciprocal of the mean first passage time to get 
            // the rate of lower exit
            T rate_lower_exit = two_forest_weight_A0_to_lower * two_forest_weight / numer_lower_exit;
            T rate_upper_exit = two_forest_weight_A0_to_upper * two_forest_weight / numer_upper_exit;

            return std::make_tuple(prob_upper_exit, rate_lower_exit, rate_upper_exit);
        } 
};

#endif 
