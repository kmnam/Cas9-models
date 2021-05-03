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
 *     4/30/2021
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
    protected:
        unsigned N;    // Length of the graph

        // Canonical ordering of the nodes 
        std::vector<Node*> order;

        // The A <-> B edge labels for the zeroth rung of the graph
        std::array<T, 2> start;

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<T, 6> > rung_labels;

        // Operators for computing spanning forest weights 
        std::vector<Matrix<T, 3, 3> > A;
        std::vector<Matrix<T, 9, 3> > B;
        std::vector<Matrix<T, 3, 4> > C;
        std::vector<Matrix<T, 8, 4> > D;
        std::vector<Matrix<T, 8, 3> > E;

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

            // ... and edges ... 
            this->addEdge("A0", "B0");
            this->addEdge("B0", "A0");
            this->start[0] = 1.0;
            this->start[1] = 1.0;

            // ... and **no** spanning forest operators yet 
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

                // ... and edges ...
                this->addEdge(sai.str(), saj.str());  // (A,i) -> (A,i+1)
                this->addEdge(saj.str(), sai.str());  // (A,i+1) -> (A,i)
                this->addEdge(sbi.str(), sbj.str());  // (B,i) -> (B,i+1)
                this->addEdge(sbj.str(), sbi.str());  // (B,i+1) -> (B,i)
                this->addEdge(saj.str(), sbj.str());  // (A,i+1) -> (B,i+1)
                this->addEdge(sbj.str(), saj.str());  // (B,i+1) -> (A,i+1)
                std::array<T, 6> labels = {1, 1, 1, 1, 1, 1};
                this->rung_labels.push_back(labels);

                // ... as well as the corresponding forest operators 
                Matrix<T, 3, 3> nextA;
                nextA << 3, 1, 1,
                         1, 1, 0,
                         1, 0, 1;
                this->A.push_back(nextA); 
                Matrix<T, 9, 3> nextB;
                nextB << 3, 0, 1,
                         0, 3, 1,
                         2, 1, 1,
                         1, 2, 1,
                         1, 0, 0,
                         1, 0, 1,
                         0, 1, 1,
                         0, 1, 0,
                         1, 1, 1;
                this->B.push_back(nextB);
                Matrix<T, 3, 4> nextC;
                nextC << 2, 2, 1, 1,
                         2, 1, 2, 1,
                         1, 1, 1, 1;
                this->C.push_back(nextC);
                Matrix<T, 8, 4> nextD;
                nextD << 2, 0, 0, 1,
                         0, 2, 2, 0,
                         2, 0, 0, 2,
                         0, 2, 1, 0,
                         1, 0, 1, 0,
                         0, 1, 0, 1,
                         1, 0, 0, 1,
                         0, 1, 1, 0;
                this->D.push_back(nextD);
                Matrix<T, 8, 3> nextE; 
                nextE << 2, 0, 1,
                         2, 0, 2,
                         2, 2, 0,
                         2, 1, 0,
                         1, 1, 0,
                         1, 0, 1,
                         1, 0, 1,
                         1, 1, 0;
                this->E.push_back(nextE);
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

            // ... as well as the corresponding forest operators
            T fA, rA, fB, rB, c, d, e, g, h, j, k; 
            fA = labels[0];
            rA = labels[1];
            fB = labels[2];
            rB = labels[3];
            c = labels[4];
            d = labels[5];
            e = rA*d + rB*c + rA*rB;
            g = fA*rB;
            h = fB*rA;
            j = c + rA;
            k = d + rB;
            Matrix<T, 3, 3> nextA;
            nextA <<  e, g*c, h*d,
                     rB,   g,   0,
                     rA,   0,   h;
            this->A.push_back(nextA);
            Matrix<T, 9, 3> nextB;
            nextB <<    e,    0,     h*d,
                        0,    e,     g*c,
                     fA*k, fB*d, fA*fB*d,
                     fA*c, fB*j, fA*fB*c,
                       rB,    0,       0,
                       rA,    0,       h,
                        0,   rB,       g,
                        0,   rA,       0,
                       fA,   fB,   fA*fB;
            this->B.push_back(nextB);
            Matrix<T, 3, 4> nextC;
            nextC << k, fA*k, fB*d, fA*fB*d,
                     j, fA*c, fB*j, fA*fB*c,
                     1,   fA,   fB,   fA*fB;
            this->C.push_back(nextC);
            Matrix<T, 8, 4> nextD;
            nextD <<  k,  0,     0,  fB*d,
                      0,  k,  fA*k,     0,
                      j,  0,     0,  fB*j,
                      0,  j,  fA*c,     0,
                     fA,  0, fA*fB,     0,
                      0, fB,     0, fA*fB,
                      1,  0,     0,    fB,
                      0,  1,    fA, 0;
            this->D.push_back(nextD);
            Matrix<T, 8, 3> nextE;
            nextE <<  k,    0, fB*d,
                      j,    0, fB*j,
                      k, fA*k,    0,
                      j, fA*c,    0,
                     rB,    g,    0,
                     rA,    0,    h,
                      1,    0,   fB,
                      1,   fA,    0;
            this->E.push_back(nextE);
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

            // Also update the corresponding forest operators
            T fA, rA, fB, rB, c, d, e, g, h, j, k; 
            fA = labels[0];
            rA = labels[1];
            fB = labels[2];
            rB = labels[3];
            c = labels[4];
            d = labels[5];
            e = rA*d + rB*c + rA*rB;
            g = fA*rB;
            h = fB*rA;
            j = c + rA;
            k = d + rB;
            Matrix<T, 3, 3> nextA;
            nextA <<  e, g*c, h*d,
                     rB,   g,   0,
                     rA,   0,   h;
            this->A[i] = nextA;
            Matrix<T, 9, 3> nextB;
            nextB <<    e,    0,     h*d,
                        0,    e,     g*c,
                     fA*k, fB*d, fA*fB*d,
                     fA*c, fB*j, fA*fB*c,
                       rB,    0,       0,
                       rA,    0,       h,
                        0,   rB,       g,
                        0,   rA,       0,
                       fA,   fB,   fA*fB;
            this->B[i] = nextB;
            Matrix<T, 3, 4> nextC;
            nextC << k, fA*k, fB*d, fA*fB*d,
                     j, fA*c, fB*j, fA*fB*c,
                     1,   fA,   fB,   fA*fB;
            this->C[i] = nextC;
            Matrix<T, 8, 4> nextD;
            nextD <<  k,  0,     0,  fB*d,
                      0,  k,  fA*k,     0,
                      j,  0,     0,  fB*j,
                      0,  j,  fA*c,     0,
                     fA,  0, fA*fB,     0,
                      0, fB,     0, fA*fB,
                      1,  0,     0,    fB,
                      0,  1,    fA, 0;
            this->D[i] = nextD;
            Matrix<T, 8, 3> nextE;
            nextE <<  k,    0, fB*d,
                      j,    0, fB*j,
                      k, fA*k,    0,
                      j, fA*c,    0,
                     rB,    g,    0,
                     rA,    0,    h,
                      1,    0,   fB,
                      1,   fA,    0;
            this->E[i] = nextE;
        }

        std::tuple<std::vector<T>, std::vector<T>, T, std::vector<T>, T> computeAllForestWeights()
        {
            /*
             * Compute the following table of spanning forest weights:
             *
             * 0) Trees rooted at (Y,i) for Y = A,B and i = 0,...,N
             * 1) Forests rooted at (Y,i), (B,N) for Y = A,B and i = 0,...,N-1
             *    with path (A,0) -> (Y,i)
             *    - Note that setting Y = A, i = 0 yields all forests rooted
             *      at (A,0), (B,N)
             * 2) Forests rooted at (A,N), (B,N) with path (A,0) -> (A,N)
             * 3) Forests rooted at (A,0), (B,N) with path (Y,i) -> (A,0)
             *    for Y = A,B and i = 0,...,N-1
             *    - Note that setting Y = A, i = 0 yields all forests rooted 
             *      at (A,0), (B,N)
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
            std::vector<Matrix<T, 3, 1> > vA;
            Matrix<T, 3, 1> vA0, vA1; 
            vA0 << weight_A0,
                   weight_A0_A1_with_path_B1_to_A0,
                   weight_A0_B1_with_path_A1_to_A0;
            vA1 << weight_B0,
                   weight_B0_A1_with_path_B1_to_B0,
                   weight_B0_B1_with_path_A1_to_B0;
            vA.push_back(vA0);
            vA.push_back(vA1);
            Matrix<T, 3, 1> vB; 
            vB << weight_A1, weight_B1, weight_A1_B1;
            std::vector<Matrix<T, 4, 1> > vC;
            Matrix<T, 4, 1> vC0, vC1; 
            vC0 << weight_A0,
                   weight_A0_A1_with_path_A0_to_A0,
                   weight_A0_B1_with_path_A0_to_A0,
                   weight_A0_A1_B1_with_path_A0_to_A0;
            vC1 << weight_B0,
                   weight_B0_A1_with_path_A0_to_B0,
                   weight_B0_B1_with_path_A0_to_B0,
                   weight_B0_A1_B1_with_path_A0_to_B0;
            vC.push_back(vC0);
            vC.push_back(vC1);
            Matrix<T, 4, 1> vD; 
            vD << weight_A1,
                  weight_B1,
                  weight_A1_B1_with_path_A0_to_B1,
                  weight_A1_B1_with_path_A0_to_A1;
            std::vector<Matrix<T, 4, 1> > vCC;
            Matrix<T, 4, 1> vCC0, vCC1; 
            vCC0 << weight_A0,
                    weight_A0_A1_with_path_A0_to_A0,
                    weight_A0_B1_with_path_A0_to_A0,
                    weight_A0_A1_B1_with_path_A0_to_A0;
            vCC1 << weight_A0,
                    weight_A0_A1_with_path_B0_to_A0,
                    weight_A0_B1_with_path_B0_to_A0,
                    weight_A0_A1_B1_with_path_B0_to_A0;
            vCC.push_back(vCC0);
            vCC.push_back(vCC1);
            Matrix<T, 3, 1> vE;
            vE << weight_A0,
                  weight_A0_A1_with_path_B1_to_A0,
                  weight_A0_B1_with_path_A1_to_A0;

            for (unsigned i = 1; i < this->N; ++i)  
            {
                // Apply operator A
                for (unsigned j = 0; j < vA.size(); ++j)
                    vA[j] = (this->A[i] * vA[j]).eval();

                // Apply operator B
                Matrix<T, 9, 1> wB = this->B[i] * vB;
                Matrix<T, 3, 1> wA0, wA1; 
                wA0 << wB(0), wB(4), wB(5);
                wA1 << wB(1), wB(6), wB(7);
                vA.push_back(wA0);
                vA.push_back(wA1);
                vB << wB(2), wB(3), wB(8);

                // Apply operator C 
                for (unsigned j = 0; j < vC.size(); ++j)
                {
                    Matrix<T, 3, 1> wC = this->C[i] * vC[j];
                    vC[j](0) = vA[j](0);
                    vC[j].tail(3) = wC; 
                }

                // Apply operator D
                Matrix<T, 8, 1> wD = this->D[i] * vD;
                Matrix<T, 4, 1> wC0, wC1;
                wC0 << vA[vA.size()-2](0), wD(0), wD(2), wD(6);
                wC1 << vA[vA.size()-1](0), wD(1), wD(3), wD(7);
                vC.push_back(wC0);
                vC.push_back(wC1);
                vD.head(2) = vB.head(2);
                vD(2) = wD(4);
                vD(3) = wD(5);

                // Apply operator C again
                for (unsigned j = 0; j < vCC.size(); ++j)
                {
                    Matrix<T, 3, 1> wCC = this->C[i] * vCC[j];
                    vCC[j](0) = vA[j](0);
                    vCC[j].tail(3) = wCC; 
                }

                // Apply operator E 
                Matrix<T, 8, 1> wE = this->E[i] * vE;
                Matrix<T, 4, 1> wCC0, wCC1;
                wCC0 << vA[vA.size()-2](0), wE(0), wE(1), wE(6); 
                wCC1 << vA[vA.size()-1](0), wE(2), wE(3), wE(7); 
                vCC.push_back(wCC0);
                vCC.push_back(wCC1);
                vE << vA[0](0), wE(4), wE(5);
            }

            // 0) Write all spanning tree weights 
            std::vector<T> tree_weights;
            for (auto&& arr : vA)
                tree_weights.push_back(arr(0));
            tree_weights.push_back(vB(0));
            tree_weights.push_back(vB(1));

            // 1) Write all weights of spanning forests rooted at (Y,i), (B,N)
            // with path (A,0) -> (Y,i)
            std::vector<T> forest_weights_Yi_BN_with_path_A0_to_Yi;
            for (auto&& arr : vC)
                forest_weights_Yi_BN_with_path_A0_to_Yi.push_back(arr(2));

            // 2) Weight of spanning forests rooted at (A,N), (B,N) with path 
            // (A,0) -> (A,N)
            T forest_weight_AN_BN_with_path_A0_to_AN = vD(3);

            // 3) Weight of spanning forests rooted at (A,0), (B,N) with path
            // (Y,i) -> (A,0)
            std::vector<T> forest_weights_A0_BN_with_path_Yi_to_A0;
            for (auto&& arr : vCC)
                forest_weights_A0_BN_with_path_Yi_to_A0.push_back(arr(2));

            // 4) Weight of spanning forests rooted at (A,0), (B,N) with path
            // (A,N) -> (A,0)
            T forest_weight_A0_BN_with_path_AN_to_A0 = vE(2);

            // Return all accumulated data
            return std::make_tuple(
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
            Array<T, Dynamic, 1> numer_lower_exit(2 * this->N + 2);
            Array<T, Dynamic, 1> numer_upper_exit(2 * this->N + 2);
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
            std::cout << numer_lower_exit.transpose() << std::endl; 
            std::cout << numer_upper_exit.transpose() << std::endl; 

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
