#ifndef GRID_GRAPH_HPP
#define GRID_GRAPH_HPP

#include <iostream>
#include <array>
#include <vector>
#include <utility>
#include <stdexcept>
#include <Eigen/Dense>

/* 
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     8/16/2019
 */
using namespace Eigen;
typedef Matrix<double, 3, 4> Matrix34d;

class GridGraph
{
    /*
     * An implementation of the two-state grid graph, with recurrence relations
     * for the spanning tree weights. 
     */
    private:
        unsigned N;    // Length of the graph

        // The A <--> B edge labels for the zeroth rung of the graph
        std::array<double, 2> start;

        // Array of edge labels that grows with the length of the graph
        std::vector<std::array<double, 6> > labels;

        Vector3d initRecurrenceA0()
        {
            /*
             * Initialize the recurrence for the weight of the spanning trees
             * rooted at (A,0). The graph must have length >= 1.
             */
            if (this->N >= 1)
            {
                Vector3d weights;
                // The labels for the n-th rung are ordered as follows:
                // b(A,n-1), d(A,n), b(B,n-1), b(B,n), k(AB,n), k(BA,n)
                weights(0) = (
                    this->start[1] * this->labels[0][1] * this->labels[0][5] +
                    this->start[1] * this->labels[0][1] * this->labels[0][3] +
                    this->labels[0][1] * this->labels[0][5] * this->labels[0][2] +
                    this->labels[0][4] * this->labels[0][3] * this->start[1]
                );
                weights(1) = this->start[1] * this->labels[0][3];
                weights(2) = (
                    this->labels[0][1] * this->labels[0][2] +
                    this->labels[0][1] * this->start[1]
                );
                return weights;
            }
            else throw std::exception();
        }

        Vector3d initRecurrenceAN()
        {
            /*
             * Initialize the recurrence for the weight of the spanning trees 
             * rooted at (A,N). The graph must have length >= 1.
             */
            if (this->N >= 1)
            {
                Vector3d weights;
                // The labels for the n-th rung are ordered as follows:
                // b(A,n-1), d(A,n), b(B,n-1), b(B,n), k(AB,n), k(BA,n)
                weights(0) = (
                    this->labels[0][0] * this->labels[0][2] * this->labels[0][5] +
                    this->labels[0][0] * this->start[1] * this->labels[0][5] +
                    this->labels[0][3] * this->start[1] * this->labels[0][0] +
                    this->start[0] * this->labels[0][2] * this->labels[0][5]
                );
                weights(1) = (
                    this->labels[0][0] * this->labels[0][2] * this->labels[0][4] +
                    this->labels[0][2] * this->start[0] * this->labels[0][4] +
                    this->labels[0][1] * this->start[0] * this->labels[0][2] +
                    this->start[1] * this->labels[0][0] * this->labels[0][4]
                );
                weights(2) = (
                    this->labels[0][0] * this->labels[0][2] +
                    this->start[0] * this->labels[0][2] +
                    this->start[1] * this->labels[0][0]
                );
                return weights;
            }
            else throw std::exception();
        }

        Vector3d initRecurrenceA0AN()
        {
            /*
             * Initialize the recurrence for the weight of the spanning forests 
             * rooted at (A,0) and (A,N). The graph must have length >= 1.
             */
            if (this->N >= 1)
            {
                Vector3d weights;
                // The labels for the n-th rung are ordered as follows:
                // b(A,n-1), d(A,n), b(B,n-1), b(B,n), k(AB,n), k(BA,n)
                weights(0) = (
                    this->labels[0][2] * this->labels[0][5] +
                    this->start[1] * this->labels[0][5] +
                    this->labels[0][3] * this->start[1]
                );
                weights(1) = (
                    this->labels[0][1] * this->start[1] +
                    this->labels[0][2] * this->labels[0][1] +
                    this->start[1] * this->labels[0][4] +
                    this->labels[0][2] * this->labels[0][4]
                );
                weights(2) = this->start[1] + this->labels[0][2];
                return weights;
            }
            else throw std::exception();
        }

        Vector3d recurrenceA0()
        {
            /*
             * Compute the recurrence for the weights of the spanning
             * trees rooted at (A,0).
             */
            Vector3d v = this->initRecurrenceA0();
            Matrix3d m;
            m(1,2) = 0.0;
            m(2,1) = 0.0;
            for (unsigned i = 0; i < this->N; i++)
            {
                // Define the i-th matrix in the recurrence
                m(0,0) = (
                    this->labels[i][5] * this->labels[i][1] +
                    this->labels[i][4] * this->labels[i][3] +
                    this->labels[i][1] * this->labels[i][3]
                );
                m(0,1) = this->labels[i][0] * this->labels[i][4] * this->labels[i][3];
                m(0,2) = this->labels[i][2] * this->labels[i][5] * this->labels[i][1];
                m(1,0) = this->labels[i][3];
                m(1,1) = this->labels[i][0] * this->labels[i][3];
                m(2,0) = this->labels[i][1];
                m(2,2) = this->labels[i][2] * this->labels[i][1];
                v = m * v;
            }
            return v;
        }

        Vector3d recurrenceAN()
        {
            /*
             * Compute the recurrence for the weights of the spanning
             * trees rooted at (A,N) (or (B,N)).
             */
            Vector3d v = this->initRecurrenceAN();
            Matrix3d m;
            for (unsigned i = 0; i < this->N; i++)
            {
                // Define the i-th matrix in the recurrence
                m(0,0) = (
                    this->labels[i][0] * this->labels[i][5] +
                    this->labels[i][0] * this->labels[i][3]
                );
                m(0,1) = this->labels[i][2] * this->labels[i][5];
                m(0,2) = this->labels[i][0] * this->labels[i][2] * this->labels[i][5];
                m(1,0) = this->labels[i][0] * this->labels[i][4];
                m(1,1) = (
                    this->labels[i][2] * this->labels[i][4] +
                    this->labels[i][2] * this->labels[i][1]
                );
                m(1,2) = this->labels[i][0] * this->labels[i][2] * this->labels[i][4];
                m(2,0) = this->labels[i][0];
                m(2,1) = this->labels[i][2];
                m(2,2) = this->labels[i][0] * this->labels[i][2];
                v = m * v;
            }
            return v;
        }

        Vector4d recurrenceA0AN()
        {
            /*
             * Compute the recurrence for the weights of the spanning
             * forests rooted at {(A,0),(A,N)} (or {(A,0),(B,N)}).
             */
            Vector3d u = this->initRecurrenceA0();
            Vector3d v = this->initRecurrenceA0AN();
            Vector3d w;
            Vector4d z;
            z << v(0), v(1), v(2), u(0);
            Matrix3d m1;
            Matrix34d m2;
            m1(1,2) = 0.0;
            m1(2,1) = 0.0;
            m2(2,3) = 1.0;
            for (unsigned i = 0; i < this->N; i++)
            {
                // Define the i-th matrix in the recurrence for the weight
                // of the spanning forests rooted at A0 and AN
                m2(0,0) = (
                    this->labels[i][0] * this->labels[i][3] +
                    this->labels[i][0] * this->labels[i][5]
                );
                m2(0,1) = this->labels[i][2] * this->labels[i][5];
                m2(0,2) = this->labels[i][0] * this->labels[i][2] * this->labels[i][5];
                m2(0,3) = this->labels[i][5] + this->labels[i][3];
                m2(1,0) = this->labels[i][0] * this->labels[i][4];
                m2(1,1) = (
                    this->labels[i][2] * this->labels[i][1] +
                    this->labels[i][2] * this->labels[i][4]
                );
                m2(1,2) = this->labels[i][0] * this->labels[i][2] * this->labels[i][4];
                m2(1,3) = this->labels[i][4] + this->labels[i][1];
                m2(2,0) = this->labels[i][0];
                m2(2,1) = this->labels[i][2];
                m2(2,2) = this->labels[i][0] * this->labels[i][2];
                w = m2 * z;

                // Define the i-th matrix in the recurrence for the weight
                // of the spanning trees rooted at A0
                m1(0,0) = (
                    this->labels[i][5] * this->labels[i][1] +
                    this->labels[i][4] * this->labels[i][3] +
                    this->labels[i][1] * this->labels[i][3]
                );
                m1(0,1) = this->labels[i][0] * this->labels[i][4] * this->labels[i][3];
                m1(0,2) = this->labels[i][2] * this->labels[i][5] * this->labels[i][1];
                m1(1,0) = this->labels[i][3];
                m1(1,1) = this->labels[i][0] * this->labels[i][3];
                m1(2,0) = this->labels[i][1];
                m1(2,2) = this->labels[i][2] * this->labels[i][1];
                u = m1 * u;

                z << w(0), w(1), w(2), u(0); 
            }
            return z;
        }

    public:
        GridGraph()
        {
            /*
             * Trivial constructor with length zero; set edge labels to unity.
             */
            this->N = 0;
            this->start[0] = 1.0;
            this->start[1] = 1.0;
        }

        GridGraph(unsigned N)
        {
            /*
             * Trivial constructor with given length; set edge labels to unity.
             */
            this->N = N;
            this->start[0] = 1.0;
            this->start[1] = 1.0;
            for (unsigned i = 0; i < N; i++)
            {
                std::array<double, 6> labels = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
                this->labels.push_back(labels);
            }
        }

        ~GridGraph()
        {
            /*
             * Trivial destructor.
             */
        }

        void setStartLabels(double kAB, double kBA)
        {
            /*
             * Set the edge labels of the zeroth rung of the graph to the
             * given values.
             */
            this->start[0] = kAB;
            this->start[1] = kBA;
        }

        void addRung(std::array<double, 6> labels)
        {
            /*
             * Add new rung onto the end of the graph, keeping track of the
             * six new edge labels. 
             */
            this->N++;
            this->labels.push_back(labels);
        }

        void setRungLabels(unsigned i, std::array<double, 6> labels)
        {
            /*
             * Set the edge labels for the i-th rung to the given values. 
             */
            this->labels[i] = labels;
        }

        double weightTreesA0()
        {
            /*
             * Return the weight of the spanning trees rooted at (A,0).
             */
            return this->recurrenceA0()(0);
        }

        double weightTreesBN()
        {
            /*
             * Return the weight of the spanning trees rooted at (B,N).
             */
            return this->recurrenceAN()(1);
        }

        double weightForestsA0BN()
        {
            /*
             * Return the weight of the spanning forests rooted at (A,0) and (B,N).
             */
            return this->recurrenceA0AN()(1);
        }

        std::pair<double, double> weightTreesA0ForestsA0BN()
        {
            /*
             * Return the weights of the spanning trees rooted at (A,0) and 
             * the spanning forests rooted at {(A,0), (B,N)}.
             */
            Vector4d v = this->recurrenceA0AN();
            return std::make_pair(v(3), v(1));
        }
};

#endif 
