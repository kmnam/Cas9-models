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
 *     10/28/2019
 */
using namespace Eigen;
typedef Matrix<double, 5, 1> Vector5d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 2, 4> Matrix24d;
typedef Matrix<double, 2, 6> Matrix26d;
typedef Matrix<double, 3, 2> Matrix32d;
typedef Matrix<double, 3, 4> Matrix34d;
typedef Matrix<double, 4, 5> Matrix45d;

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
            for (unsigned i = 0; i < N; ++i)
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

        unsigned getN()
        {
            /* 
             * Return the length of the graph.
             */
            return this->N;
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

        MatrixXd laplacian()
        {
            /*
             * Return the row Laplacian matrix of the graph.
             */
            MatrixXd laplacian = MatrixXd::Zero(2*this->N+2, 2*this->N+2);
            laplacian(0, this->N+1) = -this->start[0];
            laplacian(this->N+1, 0) = -this->start[1];
            for (unsigned i = 0; i < this->N; ++i)
            {
                laplacian(i, i+1) = -this->labels[i][0];
                laplacian(i+1, i) = -this->labels[i][1];
                laplacian(this->N+i+1, this->N+i+2) = -this->labels[i][2];
                laplacian(this->N+i+2, this->N+i+1) = -this->labels[i][3];
                laplacian(i+1, this->N+i+2) = -this->labels[i][4];
                laplacian(this->N+i+2, i+1) = -this->labels[i][5];
            }
            for (unsigned i = 0; i < 2*this->N + 2; ++i)
                laplacian(i, i) = -laplacian.row(i).sum();

            return laplacian;
        }

        Vector3d recurrenceA(const Ref<const Vector3d>& v, unsigned j)
        {
            /*
             * Apply recurrence A at the j-th rung.
             */
            Matrix3d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << l*c + k*d + c*d, a*k*d, b*l*c, d, a*d, 0, c, 0, b*c;
            return m * v;
        }

        Vector3d recurrenceB(const Ref<const Vector3d>& v, unsigned j)
        {
            /*
             * Apply recurrence B at the j-th rung. 
             */
            Matrix3d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << a*(l+d), b*l, a*b*l, a*k, b*(k+c), a*b*k, a, b, a*b;
            return m * v;
        }

        Vector3d recurrenceC(const Ref<const Vector3d>& v, unsigned j)
        {
            /*
             * Apply recurrence C at the j-th rung.
             */
            Matrix3d m;
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << l*c + k*d + c*d, 0, b*l*c, d, 0, 0, c, 0, b*c;
            return m * v;
        }

        Vector3d recurrenceD(const Ref<const Vector3d>& v, unsigned j)
        {
            /*
             * Apply recurrence D at the j-th rung.
             */
            Matrix3d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << 0, l*c + k*d + c*d, a*k*d, 0, d, a*d, 0, c, 0;
            return m * v;
        }

        Vector3d recurrenceE(const Ref<const Vector4d>& v, unsigned j)
        {
            /*
             * Apply recurrence E at the j-th rung.
             */
            Matrix34d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << d+l, a*(d+l), b*l, a*b*l, c+k, a*k, b*(c+k), a*b*k, 1, a, b, a*b;
            return m * v;
        }

        Vector3d recurrenceF(const Ref<const Vector2d>& v, unsigned j)
        {
            /*
             * Apply recurrence F at the j-th rung.
             */
            Matrix32d m;
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << d+l, b*l, c+k, b*(c+k), 1, b;
            return m * v;
        }

        Vector3d recurrenceG(const Ref<const Vector2d>& v, unsigned j)
        {
            /*
             * Apply recurrence G at the j-th rung.
             */
            Matrix32d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << d+l, a*(d+l), c+k, a*k, 1, a;
            return m * v;
        }

        Vector3d recurrenceH(const Ref<const Vector4d>& v, unsigned j)
        {
            /*
             * Apply recurrence H at the j-th rung.
             */
            Matrix34d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << d+l, a*(d+l), b*l, a*b*l, c+k, a*k, b*(c+k), a*b*k, 1, a, b, a*b;
            return m * v;
        }

        Vector2d recurrenceI(const Ref<const Vector4d>& v, unsigned j)
        {
            /*
             * Apply recurrence I at the j-th rung.
             */
            Matrix24d m;
            double a = this->labels[j-1][0];
            double b = this->labels[j-1][2];
            m << 0, b, a*b, 0, a, 0, 0, a*b;
            return m * v;
        }

        Vector4d recurrenceJ(const Ref<const Vector4d>& v, unsigned j)
        {
            /*
             * Apply recurrence J at the j-th rung.
             */
            Matrix4d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << d+l, 0, 0, b*l, 0, d+l, a*(d+l), 0, c+k, 0, 0, b*(c+k), 0, c+k, a*k, 0;
            return m * v;
        }

        Vector2d recurrenceK(const Ref<const Vector4d>& v, unsigned j)
        {
            /*
             * Apply recurrence K at the j-th rung.
             */
            Matrix24d m;
            double a = this->labels[j-1][0];
            double b = this->labels[j-1][2];
            m << 1, 0, 0, b, 0, 1, a, 0;
            return m * v;
        }

        Vector4d recurrenceL(const Ref<const Vector4d>& v, unsigned j)
        {
            /*
             * Apply recurrence L at the j-th rung.
             */
            Matrix4d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << a*(d+l), b*l, a*b*l, a*b*l, a*k, b*(c+k), a*b*k, a*b*k,
                 a, 0, a*b, 0, 0, b, 0, a*b;
            return m * v;
        }

        Vector2d recurrenceM(const Ref<const Vector6d>& v, unsigned j)
        {
            /*
             * Apply recurrence M at the j-th rung.
             */
            Matrix26d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << k, a*k, b*k, a*b*k, b*c, 0, l, a*l, b*l, a*b*l, 0, a*d;
            return m * v;
        }

        Vector4d recurrenceN(const Ref<const Vector5d>& v, unsigned j)
        {
            /*
             * Apply recurrence N at the j-th rung.
             */
            Matrix45d m;
            double a = this->labels[j-1][0];
            double c = this->labels[j-1][1];
            double b = this->labels[j-1][2];
            double d = this->labels[j-1][3];
            double k = this->labels[j-1][4];
            double l = this->labels[j-1][5];
            m << a*(d+l), 0, a*b*l, b*l, 0, 0, b*l, a*b*l, 0, a*(d+l),
                 a*k, 0, a*b*k, b*(c+k), 0, 0, b*(c+k), a*b*k, 0, a*k;
            return m * v;
        }

        Vector4d recurrenceO(const Ref<const Vector5d>& v, unsigned j)
        {
            /*
             * Apply recurrence O at the j-th rung.
             */
            Matrix45d m;
            double a = this->labels[j-1][0];
            double b = this->labels[j-1][2];
            m << 0, 0, 0, b, 0, a, 0, a*b, 0, 0, 0, b, a*b, 0, 0, 0, 0, 0, 0, a;
            return m * v;
        }

        double weightTrees(const unsigned x, const unsigned i)
        {
            /*
             * Return the weight of the spanning trees rooted at the 
             * given vertex. 
             */
            // Get the initial spanning forest weights of the starting rung
            Vector3d v;
            v << this->start[1], this->start[0], 1.0;

            // Apply recurrence B as required
            for (unsigned j = 1; j <= i; ++j) v = this->recurrenceB(v, j);

            // Return the appropriate spanning tree weight if i == N
            if (i == this->N)
                return (!x) ? v(0) : v(1);

            // Apply recurrence C or D once, depending on the root identity
            v = (!x) ? this->recurrenceC(v, i + 1) : this->recurrenceD(v, i + 1);

            // Apply recurrence A until the length of the graph is reached
            for (unsigned j = i + 2; j <= this->N; ++j) v = this->recurrenceA(v, j);

            // Return the computed spanning tree weight
            return v(0);
        }

        double weightEndToEndForests()
        {
            /*
             * Return the weight of the spanning forests rooted at (A,0)
             * and (B,N). 
             */
            // Get the initial spanning forest weights of the starting two rungs
            Vector3d u, v;
            Vector4d w;
            double k0 = this->start[0];
            double l0 = this->start[1];
            double a = this->labels[0][0];
            double c = this->labels[0][1];
            double b = this->labels[0][2];
            double d = this->labels[0][3];
            double k1 = this->labels[0][4];
            double l1 = this->labels[0][5];
            u << l0*c*d + l0*l1*c + b*l1*c + k1*d*l0, d*l0, b*c + c*l0;
            w << u(0), b*l1 + d*l0 + l0*l1, b*k1 + c*l0 + b*c + l0*k1, b + l0;

            // Alternate between recurrences A and E
            for (unsigned j = 2; j <= this->N; ++j)
            {
                v = this->recurrenceE(w, j);
                u = this->recurrenceA(u, j);
                w << u(0), v(0), v(1), v(2);
            }

            return v(1);
        }

        double weightDirectedForests(const unsigned x, const unsigned i)
        {
            /*
             * Return the weight of the spanning forests rooted at the
             * given vertex and (B,N), with a path from (A,0) to the
             * given root. The given root must be distinct from (A,0)
             * or (B,N).
             */
            if ((x == 0 && i == 0) || (x == 1 && i == this->N))
                throw std::exception();

            // Get the initial spanning forest weights from the starting rung
            Vector3d a;
            Vector2d b;
            a << this->start[1], this->start[0], 1;
            b << 0, 1;

            // Apply recurrences B and I as required
            Vector4d c;
            for (unsigned j = 1; j <= i; ++j)
            {
                c << a(0), a(1), b(0), b(1);
                b = this->recurrenceI(c, j);
                a = this->recurrenceB(a, j);
            }
            if (x == 0 && i == this->N) return b(1);

            // Apply recurrences J and K once
            Vector4d d;
            Vector2d e;
            c << a(0), a(1), b(0), b(1);
            d = this->recurrenceJ(c, i + 1);
            if (i == this->N - 1)
                return ((!x) ? d(2) : d(3));
            e = this->recurrenceK(c, i + 1);

            // Apply recurrence C/D once, depending on the root identity
            Vector3d u;
            u = (!x) ? this->recurrenceC(a, i + 1) : this->recurrenceD(a, i + 1);

            // Apply recurrences A and H until the length of the graph
            Vector3d v;
            Vector4d w;
            if (!x) w << u(0), d(0), d(2), e(0);
            else    w << u(0), d(1), d(3), e(1);
            for (unsigned j = i + 2; j <= this->N; ++j)
            {
                v = this->recurrenceH(w, j);
                u = this->recurrenceA(u, j);
                w << u(0), v(0), v(1), v(2);
            }

            return v(1);
        }

        double weightDirectedForests2(const unsigned x, const unsigned i)
        {
            /*
             * Return the weight of the spanning forests rooted at the
             * given (A,0) and (B,N), with a path from the given vertex 
             * to (B,N). The given vertex must be distinct from (A,0)
             * or (B,N).
             */
            if ((x == 0 && i == 0) || (x == 1 && i == this->N))
                throw std::exception();

            // Get the initial spanning forest weights from the starting rung
            Vector3d a;
            Vector4d b;
            Vector6d c;
            a << this->start[1], 0, 1; 
            b << this->start[1], 0, 1, 0;
            c << this->start[1], 0, 1, 0, 0, 0;

            // Apply recurrences M, A, and E as required
            Vector3d d;
            Vector2d e;
            d << 0, 1, 0;
            e << 0, 0;
            for (unsigned j = 1; j <= i; ++j)
            {
                e = this->recurrenceM(c, j);
                a = this->recurrenceA(a, j);
                d = this->recurrenceE(b, j);
                b << a(0), d(0), d(1), d(2);
                c << a(0), d(0), d(1), d(2), e(0), e(1);
            }
            if (x == 0 && i == this->N)
                return e(0);

            // Apply recurrences N and O once
            Vector5d f;
            Vector4d g, h;
            f << d(0), d(1), d(2), e(0), e(1);
            g = this->recurrenceN(f, i + 1);
            h = this->recurrenceO(f, i + 1);
            if (i == this->N - 1)
                return ((!x) ? g(2) : g(3));

            // Apply recurrence L until the length of the graph
            Vector4d v;
            if (!x) v << g(0), g(2), h(0), h(1);
            else    v << g(1), g(3), h(2), h(3);
            for (unsigned j = i + 2; j <= this->N; ++j)
                v = this->recurrenceL(v, j);

            return v(1);
        }
};

#endif 
