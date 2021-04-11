#define BOOST_TEST_MODULE testTriangularPrismGraph
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <sstream>
#include <array>
#include <utility>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/graphs/triangularPrism.hpp"

/*
 * Test module for the TriangularPrismGraph class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/11/2021
 */
using namespace Eigen;

BOOST_AUTO_TEST_CASE(testUnityTwo)
{
    /*
     * Test all public methods on the triangular-prism graph of length 2 with
     * labels set to 1.
     */
    TriangularPrismGraph<double>* graph = new TriangularPrismGraph<double>(2); 

    // Test that the graph has 9 vertices
    MatrixXd laplacian = graph->getLaplacian(); 
    BOOST_TEST(laplacian.rows() == 9);
    BOOST_TEST(laplacian.cols() == 9);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i <= 2; ++i)
    {
        // Edges (A,i) <-> (B,i)
        BOOST_TEST(laplacian(3*i + 1, 3*i) == 1.0);
        BOOST_TEST(laplacian(3*i, 3*i + 1) == 1.0);

        // Edges (B,i) <-> (C,i)
        BOOST_TEST(laplacian(3*i + 2, 3*i + 1) == 1.0);
        BOOST_TEST(laplacian(3*i + 1, 3*i + 2) == 1.0);

        // Edges (A,i) <-> (C,i)
        BOOST_TEST(laplacian(3*i + 2, 3*i) == 1.0);
        BOOST_TEST(laplacian(3*i, 3*i + 2) == 1.0);

        // Edges (A,i) <-> (A,i+1) and (B,i) <-> (B,i+1) and (C,i) <-> (C,i+1)
        if (i < 2)
        {
            BOOST_TEST(laplacian(3*i + 3, 3*i) == 1.0);
            BOOST_TEST(laplacian(3*i, 3*i + 3) == 1.0);
            BOOST_TEST(laplacian(3*i + 4, 3*i + 1) == 1.0);
            BOOST_TEST(laplacian(3*i + 1, 3*i + 4) == 1.0);
            BOOST_TEST(laplacian(3*i + 5, 3*i + 2) == 1.0);
            BOOST_TEST(laplacian(3*i + 2, 3*i + 5) == 1.0);
        }
    }

    delete graph; 
}

BOOST_AUTO_TEST_CASE(testUnityFive)
{
    /*
     * Test all public methods on the triangular-prism graph of length 5 with 
     * labels set to 1. 
     */
    TriangularPrismGraph<double>* graph = new TriangularPrismGraph<double>(5);

    // Test that the graph has 18 vertices
    MatrixXd laplacian = graph->getLaplacian(); 
    BOOST_TEST(laplacian.rows() == 18);
    BOOST_TEST(laplacian.cols() == 18);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i <= 5; ++i)
    {
        // Edges (A,i) <-> (B,i)
        BOOST_TEST(laplacian(3*i + 1, 3*i) == 1.0);
        BOOST_TEST(laplacian(3*i, 3*i + 1) == 1.0);

        // Edges (B,i) <-> (C,i)
        BOOST_TEST(laplacian(3*i + 2, 3*i + 1) == 1.0);
        BOOST_TEST(laplacian(3*i + 1, 3*i + 2) == 1.0);

        // Edges (A,i) <-> (C,i)
        BOOST_TEST(laplacian(3*i + 2, 3*i) == 1.0);
        BOOST_TEST(laplacian(3*i, 3*i + 2) == 1.0);

        // Edges (A,i) <-> (A,i+1) and (B,i) <-> (B,i+1) and (C,i) <-> (C,i+1)
        if (i < 5)
        {
            BOOST_TEST(laplacian(3*i + 3, 3*i) == 1.0);
            BOOST_TEST(laplacian(3*i, 3*i + 3) == 1.0);
            BOOST_TEST(laplacian(3*i + 4, 3*i + 1) == 1.0);
            BOOST_TEST(laplacian(3*i + 1, 3*i + 4) == 1.0);
            BOOST_TEST(laplacian(3*i + 5, 3*i + 2) == 1.0);
            BOOST_TEST(laplacian(3*i + 2, 3*i + 5) == 1.0);
        }
    }

    delete graph;
}

BOOST_AUTO_TEST_CASE(testOneToFive)
{
    /*
     * Test all public methods on the grid graph of length 5 with 
     * labels set to 1, 2, ..., 5. 
     */
    TriangularPrismGraph<double>* graph = new TriangularPrismGraph<double>(5); 
    graph->setStartLabels(1, 1, 1, 1, 1, 1);
    for (unsigned i = 0; i < 5; ++i)
    {
        double k = i + 1;
        std::array<double, 12> labels = {k, k, k, k, k, k, k, k, k, k, k, k};
        graph->setRungLabels(i, labels);
    }

    // Test that the graph has 18 vertices
    MatrixXd laplacian = graph->getLaplacian(); 
    BOOST_TEST(laplacian.rows() == 18);
    BOOST_TEST(laplacian.cols() == 18);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i <= 5; ++i)
    {
        if (i == 0)
        {
            // Edges (A,0) <-> (B,0)
            BOOST_TEST(laplacian(3*i + 1, 3*i) == 1);
            BOOST_TEST(laplacian(3*i, 3*i + 1) == 1);

            // Edges (B,0) <-> (C,0)
            BOOST_TEST(laplacian(3*i + 2, 3*i + 1) == 1);
            BOOST_TEST(laplacian(3*i + 1, 3*i + 2) == 1);

            // Edges (A,0) <-> (C,0)
            BOOST_TEST(laplacian(3*i + 2, 3*i) == 1);
            BOOST_TEST(laplacian(3*i, 3*i + 2) == 1);
        }
        else
        {
            // Edges (A,i) <-> (B,i)
            BOOST_TEST(laplacian(3*i + 1, 3*i) == i);
            BOOST_TEST(laplacian(3*i, 3*i + 1) == i);

            // Edges (B,i) <-> (C,i)
            BOOST_TEST(laplacian(3*i + 2, 3*i + 1) == i);
            BOOST_TEST(laplacian(3*i + 1, 3*i + 2) == i);

            // Edges (A,i) <-> (C,i)
            BOOST_TEST(laplacian(3*i + 2, 3*i) == i);
            BOOST_TEST(laplacian(3*i, 3*i + 2) == i);
        }

        // Edges (A,i) <-> (A,i+1) and (B,i) <-> (B,i+1) and (C,i) <-> (C,i+1)
        if (i < 5)
        {
            BOOST_TEST(laplacian(3*i + 3, 3*i) == i + 1);
            BOOST_TEST(laplacian(3*i, 3*i + 3) == i + 1);
            BOOST_TEST(laplacian(3*i + 4, 3*i + 1) == i + 1);
            BOOST_TEST(laplacian(3*i + 1, 3*i + 4) == i + 1);
            BOOST_TEST(laplacian(3*i + 5, 3*i + 2) == i + 1);
            BOOST_TEST(laplacian(3*i + 2, 3*i + 5) == i + 1);
        }
    }

    delete graph;
}


