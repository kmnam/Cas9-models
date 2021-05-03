#ifndef GRID_MATCH_MISMATCH_GRAPH_HPP
#define GRID_MATCH_MISMATCH_GRAPH_HPP

#include <iostream>
#include <sstream>
#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <boost/multiprecision/mpfr.hpp>
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

            // Update the edge labels in the graph at each match position
            for (unsigned i = 0; i < this->N; ++i)
            {
                if (this->pattern[i])
                {
                    std::stringstream sai, sbi, saj, sbj;
                    sai << "A" << i;
                    sbi << "B" << i;
                    saj << "A" << i + 1;
                    sbj << "B" << i + 1;
                    this->setEdgeLabel(sai.str(), saj.str(), label);
                    this->setEdgeLabel(sbi.str(), sbj.str(), label);
                    this->rung_labels[i][0] = label;
                    this->rung_labels[i][2] = label;
                }
            }
        }

        void setMatchReverseLabel(T label)
        {
            /*
             * Update the match reverse transition label.
             */
            this->match_labels[1] = label;

            // Update the edge labels in the graph at each match position
            for (unsigned i = 0; i < this->N; ++i)
            {
                if (this->pattern[i])
                {
                    std::stringstream sai, sbi, saj, sbj;
                    sai << "A" << i;
                    sbi << "B" << i;
                    saj << "A" << i + 1;
                    sbj << "B" << i + 1;
                    this->setEdgeLabel(saj.str(), sai.str(), label);
                    this->setEdgeLabel(sbj.str(), sbi.str(), label);
                    this->rung_labels[i][1] = label;
                    this->rung_labels[i][3] = label;
                }
            }
        }

        void setMismatchForwardLabel(T label)
        {
            /*
             * Update the mismatch forward transition label.
             */
            this->mismatch_labels[0] = label;

            // Update the edge labels in the graph at each mismatch position
            for (unsigned i = 0; i < this->N; ++i)
            {
                if (!this->pattern[i])
                {
                    std::stringstream sai, sbi, saj, sbj;
                    sai << "A" << i;
                    sbi << "B" << i;
                    saj << "A" << i + 1;
                    sbj << "B" << i + 1;
                    this->setEdgeLabel(sai.str(), saj.str(), label);
                    this->setEdgeLabel(sbi.str(), sbj.str(), label);
                    this->rung_labels[i][0] = label;
                    this->rung_labels[i][2] = label;
                }
            }
        }

        void setMismatchReverseLabel(T label)
        {
            /*
             * Update the mismatch reverse transition label.
             */
            this->mismatch_labels[1] = label;

            // Update the edge labels in the graph at each mismatch position
            for (unsigned i = 0; i < this->N; ++i)
            {
                if (!this->pattern[i])
                {
                    std::stringstream sai, sbi, saj, sbj;
                    sai << "A" << i;
                    sbi << "B" << i;
                    saj << "A" << i + 1;
                    sbj << "B" << i + 1;
                    this->setEdgeLabel(saj.str(), sai.str(), label);
                    this->setEdgeLabel(sbj.str(), sbi.str(), label);
                    this->rung_labels[i][1] = label;
                    this->rung_labels[i][3] = label;
                }
            }
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

            // Define match and mismatch constants ...
            // e = reverse * (off + on + reverse)
            T match_e = this->match_labels[1] * (this->start[1] + this->start[0] + this->match_labels[1]);
            T mismatch_e = this->mismatch_labels[1] * (this->start[1] + this->start[0] + this->mismatch_labels[1]);
            // g = forward * reverse [fA * rB]
            T match_g = this->match_labels[0] * this->match_labels[1];
            T mismatch_g = this->mismatch_labels[0] * this->mismatch_labels[1];
            // h = forward * reverse [fB * rA]
            T match_h = match_g; 
            T mismatch_h = mismatch_g;
            // k = on + reverse [rA + c]
            T match_k = this->start[0] + this->match_labels[1];
            T mismatch_k = this->start[0] + this->mismatch_labels[1];
            // l = off + reverse [rB + d]
            T match_l = this->start[1] + this->match_labels[1];
            T mismatch_l = this->start[1] + this->mismatch_labels[1];
            T e, g, h, k, l, f, r;

            for (unsigned i = 1; i < this->N; ++i)  
            {
                // i here refers to the i-th position of the sequence (zero-indexed)
                // and so the fully hybridized states are (A,i+1), (B,i+1)
                bool match_i = this->pattern[i]; 

                // Apply operator A (Eqn. B.19)
                // Each array in vA stores: 
                // 1) Weight of all trees rooted at j-th vertex
                // 2) Weight of all 2-forests rooted at j-th vertex, (A,n) with path (B,n) -> j-th vertex
                // 3) Weight of all 2-forests rooted at j-th vertex, (B,n) with path (A,n) -> j-th vertex
                for (unsigned j = 0; j < vA.size(); ++j)
                {
                    // The j-th vertex here is (A,0), (B,0), (A,1), (B,1), ...
                    bool match_j = this->pattern[(j % 2 == 0) ? static_cast<int>(j / 2) : static_cast<int>((j - 1) / 2)];
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

                // Add the next two entries to vA ...
                // These arrays contain the forest weights pertaining to the
                // i-th (**not** fully hybridized) nodes in the graph
                // (i.e., (A,i), (B,i))
                e = (match_i) ? match_e : mismatch_e; 
                g = (match_i) ? match_g : mismatch_g;
                h = (match_i) ? match_h : mismatch_h;
                k = (match_i) ? match_k : mismatch_k;
                l = (match_i) ? match_l : mismatch_l;
                f = (match_i) ? this->match_labels[0] : this->mismatch_labels[0]; 
                r = (match_i) ? this->match_labels[1] : this->mismatch_labels[1];
                std::array<T, 3> vA_first, vA_second;
                // vA_first stores the initial weights pertaining to (A,i)
                vA_first[0] = e * vB[0] + h * this->start[1] * vB[2];    // Trees rooted at (A,i) : Eqn. B.28, first row
                vA_first[1] = r * vB[0];                                 // (A,i), (A,i+1) with path (B,i+1) -> (A,i)
                vA_first[2] = r * vB[0] + h * vB[2];                     // (A,i), (B,i+1) with path (A,i+1) -> (A,i)
                // vA_second stores the initial weights pertaining to (B,i)
                vA_second[0] = e * vB[1] + g * this->start[0] * vB[2];   // Trees rooted at (B,i) : Eqn. B.28, second row 
                vA_second[1] = r * vB[1] + g * vB[2];                    // (B,i), (A,i+1) with path (B,i+1) -> (B,i)
                vA_second[2] = r * vB[1];                                // (B,i), (B,i+1) with path (A,i+1) -> (B,i)
                vA.push_back(vA_first);
                vA.push_back(vA_second);

                // Apply operator B (Eqn. B.27)
                // Each version of vB stores: 
                // 1) Weight of all trees rooted at (A,i+1) -- the fully hybridized node 
                // 2) Weight of all trees rooted at (B,i+1) -- the fully hybridized node
                // 3) Weight of all 2-forests rooted at (A,i+1), (B,i+1) -- both fully hybridized nodes 
                std::array<T, 3> vB_next; 
                vB_next[0] = f * (l * vB[0] + this->start[1] * (vB[1] + f * vB[2]));
                vB_next[1] = f * (this->start[0] * (vB[0] + f * vB[2]) + k * vB[1]);
                vB_next[2] = f * (vB[0] + vB[1] + f * vB[2]); 
                vB[0] = vB_next[0];
                vB[1] = vB_next[1];
                vB[2] = vB_next[2];

                // Apply operator D (before C!) (Eqn. B.33)
                // Each version of vD stores: 
                // 1) Weight of all trees rooted at (A,i+1)
                // 2) Weight of all trees rooted at (B,i+1)
                // 3) Weight of all 2-forests rooted at (A,i+1),(B,i+1) with path (A,0) -> (B,i+1)
                // 4) Weight of all 2-forests rooted at (A,i+1),(B,i+1) with path (A,0) -> (A,i+1)
                std::array<T, 2> vD_next;
                T u0 = vD[0] + f * vD[3];
                T u1 = vD[1] + f * vD[2]; 
                vD_next[0] = f * (vD[0] + f * vD[2]);  // Eqn. B.33, row 4
                vD_next[1] = f * (vD[1] + f * vD[3]);  // Eqn. B.33, row 5

                // Apply operator E (before C!)
                // Each version of vE stores: 
                // 1) Weight of all trees rooted at (A,0)
                // 2) Weight of all 2-forests rooted at (A,0),(A,i+1) with path (B,i+1) -> (A,0)
                // 3) Weight of all 2-forests rooted at (A,0),(B,i+1) with path (A,i+1) -> (A,0)
                std::array<T, 2> vE_next;
                T x1 = vE[0] + f * vE[1]; 
                T x2 = vE[0] + f * vE[2];
                vE_next[0] = r * x1;   // Eqn. B.30, row 4
                vE_next[1] = r * x2;   // Eqn. B.30, row 5

                // Here we define new arrays to be added to vC and vC1 (Eqn. B.33)
                // vC_first stores: 
                // 1) Weight of all trees rooted at (A,i)
                // 2) Weight of all 2-forests rooted at (A,i),(A,i+1) with path (A,0) -> (A,i)
                // 3) Weight of all 2-forests rooted at (A,i),(B,i+1) with path (A,0) -> (A,i)
                // 4) Weight of all 3-forests rooted at (A,i),(A,i+1),(B,i+1) with path (A,0) -> (A,i)
                std::array<T, 4> vC_first, vC_second;
                vC_first[0] = vA_first[0];    // Just take from vA_first
                vC_first[1] = l * vD[0] + f * this->start[1] * vD[3];    // Eqn. B.33, row 0
                vC_first[2] = k * u0;                                    // Eqn. B.33, row 2
                vC_first[3] = u0;                                        // Eqn. B.33, row 6
                // vC_second stores:
                // 1) Weight of all trees rooted at (B,i)
                // 2) Weight of all 2-forests rooted at (B,i),(A,i+1) with path (A,0) -> (B,i)
                // 3) Weight of all 2-forests rooted at (B,i),(B,i+1) with path (A,0) -> (B,i)
                // 4) Weight of all 3-forests rooted at (B,i),(A,i+1),(B,i+1) with path (A,0) -> (B,i)
                vC_second[0] = vA_second[0];   // Just take from vA_second
                vC_second[1] = l * u1;                                   // Eqn. B.33, row 1
                vC_second[2] = k * vD[1] + f * this->start[0] * vD[2];   // Eqn. B.33, row 3
                vC_second[3] = u1;                                       // Eqn. B.33, row 7
                // vC1_first stores:
                // 1) Weight of all trees rooted at (A,0)
                // 2) Weight of all 2-forests rooted at (A,0),(A,i+1) with path (A,i) -> (A,0)
                // 3) Weight of all 2-forests rooted at (A,0),(B,i+1) with path (A,i) -> (A,0)
                // 4) Weight of all 3-forests rooted at (A,0),(A,i+1),(B,i+1) with path (A,i) -> (A,0)
                std::array<T, 4> vC1_first, vC1_second;
                vC1_first[0] = vA[0][0];       // Just take from vA[0]
                vC1_first[1] = l * vE[0] + f * this->start[1] * vE[2];   // Eqn. B.30, row 0
                vC1_first[2] = k * x2;                                   // Eqn. B.30, row 1
                vC1_first[3] = x2;                                       // Eqn. B.30, row 6
                // vC1_second stores:
                // 1) Weight of all trees rooted at (A,0)
                // 2) Weight of all 2-forests rooted at (A,0),(A,i+1) with path (B,i) -> (A,0)
                // 3) Weight of all 2-forests rooted at (A,0),(B,i+1) with path (B,i) -> (A,0)
                // 4) Weight of all 3-forests rooted at (A,0),(A,i+1),(B,i+1) with path (B,i) -> (A,0)
                vC1_second[0] = vA[0][0];      // Just take from vA[0]
                vC1_second[1] = l * x1;                                  // Eqn. B.30, row 2
                vC1_second[2] = k * vE[0] + f * this->start[0] * vE[1];  // Eqn. B.30, row 3
                vC1_second[3] = x1;                                      // Eqn. B.30, row 7

                // Apply operator C 
                for (unsigned j = 0; j < vC.size(); ++j)
                {
                    // The j-th vertex here is (A,0), (B,0), (A,1), (B,1), ...
                    bool match_j = this->pattern[(j % 2 == 0) ? static_cast<int>(j / 2) : static_cast<int>((j - 1) / 2)];
                    e = (match_j) ? match_e : mismatch_e; 
                    g = (match_j) ? match_g : mismatch_g;
                    h = (match_j) ? match_h : mismatch_h;
                    k = (match_j) ? match_k : mismatch_k;
                    l = (match_j) ? match_l : mismatch_l;
                    f = (match_j) ? this->match_labels[0] : this->mismatch_labels[0]; 
                    r = (match_j) ? this->match_labels[1] : this->mismatch_labels[1];

                    // Apply operator C (Eqn. B.32)
                    // Each array in vC stores: 
                    // 1) Weight of all trees rooted at j-th vertex
                    // 2) Weight of all 2-forests rooted at j-th vertex, (A,i+1) with path (A,0) -> j-th vertex
                    // 3) Weight of all 2-forests rooted at j-th vertex, (B,i+1) with path (A,0) -> j-th vertex
                    // 4) Weight of all 3-forests rooted at j-th vertex, (A,i+1), (B,i+1) with path (A,0) -> j-th vertex
                    std::array<T, 3> wC; 
                    wC[0] = l * vC[j][0] + f * (l * vC[j][1] + this->start[1] * (vC[j][2] + f * vC[j][3]));
                    wC[1] = k * vC[j][0] + f * (this->start[0] * (vC[j][1] + f * vC[j][3]) + k * vC[j][2]);
                    wC[2] = vC[j][0] + f * (vC[j][1] + vC[j][2] + f * vC[j][3]);
                    vC[j][0] = vA[j][0];    // Just take from vA[j]
                    vC[j][1] = wC[0];       // Eqn. B.32, row 0
                    vC[j][2] = wC[1];       // Eqn. B.32, row 1
                    vC[j][3] = wC[2];       // Eqn. B.32, row 2
                }

                // Apply operator C again (Eqn. B.29)
                for (unsigned j = 0; j < vC1.size(); ++j)
                {
                    // The j-th vertex here is (A,0), (B,0), (A,1), (B,1), ...
                    bool match_j = this->pattern[(j % 2 == 0) ? static_cast<int>(j / 2) : static_cast<int>((j - 1) / 2)];
                    e = (match_j) ? match_e : mismatch_e; 
                    g = (match_j) ? match_g : mismatch_g;
                    h = (match_j) ? match_h : mismatch_h;
                    k = (match_j) ? match_k : mismatch_k;
                    l = (match_j) ? match_l : mismatch_l;
                    f = (match_j) ? this->match_labels[0] : this->mismatch_labels[0]; 
                    r = (match_j) ? this->match_labels[1] : this->mismatch_labels[1];

                    // Apply operator C (Eqn. B.29)
                    // Each array in vC1 stores: 
                    // 1) Weight of all trees rooted at (A,0)
                    // 2) Weight of all 2-forests rooted at (A,0), (A,i+1) with path j-th vertex -> (A,0)
                    // 3) Weight of all 2-forests rooted at (A,0), (B,i+1) with path j-th vertex -> (A,0)
                    // 4) Weight of all 3-forests rooted at (A,0), (A,i+1), (B,i+1) with path j-th vertex -> (A,0)
                    std::array<T, 3> wC1; 
                    wC1[0] = l * vC1[j][0] + f * (l * vC1[j][1] + this->start[1] * (vC1[j][2] + f * vC1[j][3]));
                    wC1[1] = k * vC1[j][0] + f * (this->start[0] * (vC1[j][1] + f * vC1[j][3]) + k * vC1[j][2]);
                    wC1[2] = vC1[j][0] + f * (vC1[j][1] + vC1[j][2] + f * vC1[j][3]);
                    vC1[j][0] = vA[0][0];    // Just take from vA[0]
                    vC1[j][1] = wC1[0];      // Eqn. B.29, row 0
                    vC1[j][2] = wC1[1];      // Eqn. B.29, row 1
                    vC1[j][3] = wC1[2];      // Eqn. B.29, row 2
                }

                // Append the new vectors to be added to vC and vC1
                vC.push_back(vC_first);
                vC.push_back(vC_second);
                vC1.push_back(vC1_first);
                vC1.push_back(vC1_second);

                // Update vD and vE
                vD[0] = vB[0];         // Just take from first two rows of vB
                vD[1] = vB[1];
                vD[2] = vD_next[0];    // Eqn. B.33, row 4
                vD[3] = vD_next[1];    // Eqn. B.33, row 5
                vE[0] = vA[0][0];      // Just take from first row of vA[0]
                vE[1] = vE_next[0];    // Eqn. B.30, row 4
                vE[2] = vE_next[1];    // Eqn. B.30, row 5
            }

            // 0) Write all spanning tree weights [stored in vA and vB]
            std::vector<T> tree_weights;
            for (auto&& arr : vA)
                tree_weights.push_back(arr[0]);
            tree_weights.push_back(vB[0]);
            tree_weights.push_back(vB[1]);

            // 1) Write all weights of spanning forests rooted at (Y,i), (B,N)
            // with path (A,0) -> (Y,i) [stored in vC] 
            std::vector<T> forest_weights_Yi_BN_with_path_A0_to_Yi;
            for (auto&& arr : vC)
                forest_weights_Yi_BN_with_path_A0_to_Yi.push_back(arr[2]);

            // 2) Weight of spanning forests rooted at (A,N), (B,N) with path 
            // (A,0) -> (A,N) [stored in vD]
            T forest_weight_AN_BN_with_path_A0_to_AN = vD[3];

            // 3) Write all weights of spanning forests rooted at (A,0), (B,N)
            // with path (Y,i) -> (A,0) [stored in vC1]
            std::vector<T> forest_weights_A0_BN_with_path_Yi_to_A0;
            for (auto&& arr : vC1)
                forest_weights_A0_BN_with_path_Yi_to_A0.push_back(arr[2]);

            // 4) Weight of spanning forests rooted at (A,0), (B,N) with path
            // (A,N) -> (A,0) [stored in vE]
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

            // Get weight of all forests rooted at exit vertices with path (A,0) -> upper exit 
            T two_forest_weight_A0_to_upper = weight_BN * exit_rate_upper_prob; 

            // Probability of upper exit is given by ...
            T prob_upper_exit = two_forest_weight_A0_to_upper / two_forest_weight; 

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

            // Re-compute weight of all forests rooted at exit vertices with path (A,0) -> upper exit 
            two_forest_weight_A0_to_upper = weight_BN * exit_rate_upper_time; 
            
            // Mean first passage times to lower/upper exit are given by ...
            Array<T, Dynamic, 1> numer_lower_exit = Array<T, Dynamic, 1>::Zero(2 * this->N + 2);
            Array<T, Dynamic, 1> numer_upper_exit = Array<T, Dynamic, 1>::Zero(2 * this->N + 2);
            T log_two_forest_weight = boost::multiprecision::log(two_forest_weight);
            for (unsigned i = 0; i < 2 * this->N; ++i)
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
                std::cout << two_forest_weight_Yi_to_lower << " "
                          << two_forest_weight << " "
                          << (two_forest_weight_Yi_to_lower > two_forest_weight) << std::endl; 

                // Get weight of all 2-forests rooted at exit vertices with 
                // path (Y,i) -> upper exit
                // ----> Compute in log-scale using the log-diff-exp function
                T log_two_forest_weight_Yi_to_lower = boost::multiprecision::log(two_forest_weight_Yi_to_lower);
                T log_two_forest_weight_Yi_to_upper = (
                    log_two_forest_weight + boost::multiprecision::log(
                        1.0 - boost::multiprecision::exp(log_two_forest_weight_Yi_to_lower - log_two_forest_weight)
                    )
                );

                // Get weight of all 3-forests rooted at exit vertices and (Y,i)
                // with path (A,0) -> (Y,i)
                T log_three_forest_weight_A0_to_Yi = boost::multiprecision::log(
                    weight_Yi + weight_Yi_BN_with_path_A0_to_Yi * exit_rate_upper_time
                );

                // Get contribution to numerators of mean first passage time
                numer_lower_exit(i) = log_three_forest_weight_A0_to_Yi + log_two_forest_weight_Yi_to_lower;
                numer_upper_exit(i) = log_three_forest_weight_A0_to_Yi + log_two_forest_weight_Yi_to_upper;
            }

            // Get contribution to numerators for (Y,i) = (A,N)
            T weight_AN_BN_with_path_A0_to_AN = std::get<2>(weights);
            T weight_A0_BN_with_path_AN_to_A0 = std::get<4>(weights);
            T log_two_forest_weight_AN_to_lower = boost::multiprecision::log(
                weight_A0 * exit_rate_lower_time +
                weight_A0_BN_with_path_AN_to_A0 * exit_rate_lower_time * exit_rate_upper_time
            );
            T log_two_forest_weight_AN_to_upper = (
                log_two_forest_weight + boost::multiprecision::log(
                    1.0 - boost::multiprecision::exp(log_two_forest_weight_AN_to_lower - log_two_forest_weight)
                )
            ); 
            T log_three_forest_weight_A0_to_AN = boost::multiprecision::log(
                weight_AN + weight_AN_BN_with_path_A0_to_AN * exit_rate_upper_time
            );
            numer_lower_exit(2 * this->N) = log_three_forest_weight_A0_to_AN + log_two_forest_weight_AN_to_lower;
            numer_upper_exit(2 * this->N) = log_three_forest_weight_A0_to_AN + log_two_forest_weight_AN_to_upper;

            // Get contribution to numerators for (Y,i) = (B,N)
            T log_two_forest_weight_BN_to_lower = boost::multiprecision::log(weight_A0 * exit_rate_lower_time);
            T log_two_forest_weight_BN_to_upper = (
                log_two_forest_weight + boost::multiprecision::log(
                    1.0 - boost::multiprecision::exp(log_two_forest_weight_BN_to_lower - log_two_forest_weight)
                )
            );
            T log_three_forest_weight_A0_to_BN = boost::multiprecision::log(weight_BN);
            numer_lower_exit(2 * this->N + 1) = log_three_forest_weight_A0_to_BN + log_two_forest_weight_BN_to_lower;
            numer_upper_exit(2 * this->N + 1) = log_three_forest_weight_A0_to_BN + log_two_forest_weight_BN_to_upper;
            std::cout << "lower " << numer_lower_exit.transpose() << std::endl; 
            std::cout << "upper " << numer_upper_exit.transpose() << std::endl; 

            // To get the sum of these contributions in log-scale, first get the maxima ...
            T max_numer_lower_exit = numer_lower_exit.maxCoeff();
            T max_numer_upper_exit = numer_upper_exit.maxCoeff();

            // ... subtract the maxima from their respective arrays ...
            numer_lower_exit -= max_numer_lower_exit; 
            numer_upper_exit -= max_numer_upper_exit; 

            // ... exponentiate, sum, take logarithms, and add back the maxima
            T numer_lower_exit_total = boost::multiprecision::log(numer_lower_exit.exp().sum()) + max_numer_lower_exit;
            T numer_upper_exit_total = boost::multiprecision::log(numer_upper_exit.exp().sum()) + max_numer_upper_exit; 
            
            // Take the reciprocal of the mean first passage time to get 
            // the rate of lower exit
            T rate_lower_exit = boost::multiprecision::exp(
                boost::multiprecision::log(two_forest_weight_A0_to_lower)
                + log_two_forest_weight - numer_lower_exit_total
            );
            T rate_upper_exit = boost::multiprecision::exp(
                boost::multiprecision::log(two_forest_weight_A0_to_upper) 
                + log_two_forest_weight - numer_upper_exit_total
            );

            return std::make_tuple(prob_upper_exit, rate_lower_exit, rate_upper_exit);
        } 
};

#endif 
