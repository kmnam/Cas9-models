#define BOOST_TEST_MODULE testGridGraph
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <sstream>
#include <array>
#include <utility>
#include <Eigen/Dense>
#include <boost/test/included/unit_test.hpp>
#include "../../include/graphs/grid.hpp"

/*
 * Test module for the GridGraph class.
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
     * Test all public methods on the grid graph of length 2 with
     * labels set to 1.
     */
    GridGraph<double>* graph = new GridGraph<double>(2);

    // Test that the graph has 6 vertices
    MatrixXd laplacian = graph->getLaplacian(); 
    BOOST_TEST(laplacian.rows() == 6);
    BOOST_TEST(laplacian.cols() == 6);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i <= 2; ++i)
    {
        // Edge (A,i) -> (B,i)
        BOOST_TEST(laplacian(2*i + 1, 2*i) == 1.0);

        // Edge (B,i) -> (A,i)
        BOOST_TEST(laplacian(2*i, 2*i + 1) == 1.0);

        // Edges (A,i) <-> (A,i+1) and (B,i) <-> (B,i+1)
        if (i < 2)
        {
            BOOST_TEST(laplacian(2*i + 2, 2*i) == 1.0);
            BOOST_TEST(laplacian(2*i, 2*i + 2) == 1.0);
            BOOST_TEST(laplacian(2*i + 3, 2*i + 1) == 1.0);
            BOOST_TEST(laplacian(2*i + 1, 2*i + 3) == 1.0);
        }
    }

    // Compute all spanning forest weights
    std::tuple<std::vector<double>, 
               std::vector<double>, 
               double, 
               std::vector<double>,
               double> weights = graph->computeAllForestWeights();

    // Count the spanning trees rooted at each vertex 
    BOOST_TEST(std::get<0>(weights)[0] == 15);    // Rooted at (A,0)
    BOOST_TEST(std::get<0>(weights)[1] == 15);    // Rooted at (B,0)
    BOOST_TEST(std::get<0>(weights)[2] == 15);    // Rooted at (A,1)
    BOOST_TEST(std::get<0>(weights)[3] == 15);    // Rooted at (B,1)
    BOOST_TEST(std::get<0>(weights)[4] == 15);    // Rooted at (A,2)
    BOOST_TEST(std::get<0>(weights)[5] == 15);    // Rooted at (B,2)

    // Count the spanning forests rooted at (Y,i), (B,N) with path (A,0) -> (Y,i)
    BOOST_TEST(std::get<1>(weights)[0] == 21);    // Rooted at (A,0), (B,2)
    BOOST_TEST(std::get<1>(weights)[1] == 15);    // Rooted at (B,0), (B,2) with (A,0) -> (B,0)
    BOOST_TEST(std::get<1>(weights)[2] == 12);    // Rooted at (A,1), (B,2) with (A,0) -> (A,1)
    BOOST_TEST(std::get<1>(weights)[3] == 9);     // Rooted at (B,1), (B,2) with (A,0) -> (B,1)

    // Count the spanning forests rooted at (A,N), (B,N) with path (A,0) -> (A,N)
    BOOST_TEST(std::get<2>(weights) == 6);        // Rooted at (A,2), (B,2) with (A,0) -> (A,2)

    // Count the spanning forests rooted at (A,0), (B,N) with path (Y,i) -> (A,0)
    BOOST_TEST(std::get<3>(weights)[0] == 21);    // Rooted at (A,0), (B,2)
    BOOST_TEST(std::get<3>(weights)[1] == 15);    // Rooted at (A,0), (B,2) with (B,0) -> (A,0)
    BOOST_TEST(std::get<3>(weights)[2] == 12);    // Rooted at (A,0), (B,2) with (A,1) -> (A,0)
    BOOST_TEST(std::get<3>(weights)[3] == 9);     // Rooted at (A,0), (B,2) with (B,1) -> (A,0)

    // Count the spanning forests rooted at (A,0), (B,N) with path (A,N) -> (A,0)
    BOOST_TEST(std::get<4>(weights) == 6);        // Rooted at (A,0), (B,2) with (A,2) -> (A,0)

    // Compute the probability of upper exit and reciprocal of the mean first
    // passage time to lower exit, with exit rates set to 1
    std::pair<double, double> exit_stats = graph->computeExitStats(1, 1, 1, 1);
    
    // Compute the probability of upper exit again with the Chebotarev-Agaev
    // recurrence
    LabeledDigraph<double>* copy = graph->copy();
    copy->addNode("lower");
    copy->addNode("upper");
    copy->addEdge("A0", "lower", 1.0);
    copy->addEdge("B2", "upper", 1.0);
    MatrixXd copy_laplacian = copy->getLaplacian();
    MatrixXd forest_matrix_1 = copy->getSpanningForestMatrix(6 + 2 - 3);
    MatrixXd forest_matrix_2 = copy->getSpanningForestMatrix(6 + 2 - 2);
    double copy_prob = forest_matrix_2(0, 6 + 2 - 1) / forest_matrix_2.row(0).sum();
    BOOST_TEST(std::abs(exit_stats.first - copy_prob) < 1e-20);

    // Compute the rate of lower exit again with the Chebotarev-Agaev recurrence
    double copy_time = 0.0;
    for (unsigned k = 0; k <= 5; ++k)
        copy_time += (forest_matrix_1(0, k) * forest_matrix_2(k, 6 + 2 - 2));
    double copy_rate = (forest_matrix_2(0, 6 + 2 - 2) * forest_matrix_2.row(0).sum()) / copy_time;
    BOOST_TEST(std::abs(exit_stats.second - copy_rate) < 1e-20);

    delete graph;
    delete copy; 
}

BOOST_AUTO_TEST_CASE(testUnityFive)
{
    /*
     * Test all public methods on the grid graph of length 5 with 
     * labels set to 1. 
     */
    GridGraph<double>* graph = new GridGraph<double>(5);

    // Test that the graph has 12 vertices
    MatrixXd laplacian = graph->getLaplacian(); 
    BOOST_TEST(laplacian.rows() == 12);
    BOOST_TEST(laplacian.cols() == 12);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i <= 5; ++i)
    {
        // Edge (A,i) -> (B,i)
        BOOST_TEST(laplacian(2*i + 1, 2*i) == 1.0);

        // Edge (B,i) -> (A,i)
        BOOST_TEST(laplacian(2*i, 2*i + 1) == 1.0);

        // Edges (A,i) <-> (A,i+1) and (B,i) <-> (B,i+1)
        if (i < 5)
        {
            BOOST_TEST(laplacian(2*i + 2, 2*i) == 1.0);
            BOOST_TEST(laplacian(2*i, 2*i + 2) == 1.0);
            BOOST_TEST(laplacian(2*i + 3, 2*i + 1) == 1.0);
            BOOST_TEST(laplacian(2*i + 1, 2*i + 3) == 1.0);
        }
    }

    // Compute all spanning forest weights
    std::tuple<std::vector<double>, 
               std::vector<double>, 
               double, 
               std::vector<double>,
               double> weights = graph->computeAllForestWeights();

    // Count the spanning trees rooted at each vertex 
    for (unsigned i = 0; i < 12; ++i)
        BOOST_TEST(std::get<0>(weights)[i] == 780);

    // Compute the probability of upper exit and reciprocal of the mean first
    // passage time to lower exit, with exit rates set to 1
    std::pair<double, double> exit_stats = graph->computeExitStats(1, 1, 1, 1);
 
    // Compute the probability of upper exit again with the Chebotarev-Agaev
    // recurrence
    LabeledDigraph<double>* copy = graph->copy();
    copy->addNode("lower");
    copy->addNode("upper");
    copy->addEdge("A0", "lower", 1.0);
    copy->addEdge("B5", "upper", 1.0);
    MatrixXd copy_laplacian = copy->getLaplacian();
    MatrixXd forest_matrix_1 = copy->getSpanningForestMatrix(12 + 2 - 3);
    MatrixXd forest_matrix_2 = copy->getSpanningForestMatrix(12 + 2 - 2);
    double copy_prob = forest_matrix_2(0, 12 + 2 - 1) / forest_matrix_2.row(0).sum();
    BOOST_TEST(std::abs(exit_stats.first - copy_prob) < 1e-20);

    // Compute the rate of lower exit again with the Chebotarev-Agaev recurrence
    double copy_time = 0.0;
    for (unsigned k = 0; k < 12; ++k)
        copy_time += (forest_matrix_1(0, k) * forest_matrix_2(k, 12 + 2 - 2));
    double copy_rate = (forest_matrix_2(0, 12 + 2 - 2) * forest_matrix_2.row(0).sum()) / copy_time;
    BOOST_TEST(std::abs(exit_stats.second - copy_rate) < 1e-20);

    delete graph;
    delete copy; 
}

BOOST_AUTO_TEST_CASE(testOneToFive)
{
    /*
     * Test all public methods on the grid graph of length 5 with 
     * labels set to 1, 2, ..., 5. 
     */
    GridGraph<double>* graph = new GridGraph<double>(5);
    graph->setStartLabels(1, 1);
    for (unsigned i = 0; i < 5; ++i)
    {
        double k = i + 1;
        std::array<double, 6> labels = {k, k, k, k, k, k};
        graph->setRungLabels(i, labels);
    }

    // Test that the graph has 12 vertices
    MatrixXd laplacian = graph->getLaplacian();
    BOOST_TEST(laplacian.rows() == 12);
    BOOST_TEST(laplacian.cols() == 12);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i <= 5; ++i)
    {
        if (i == 0)
        {
            // Edge (A,0) -> (B,0)
            BOOST_TEST(laplacian(2*i + 1, 2*i) == 1);

            // Edge (B,0) -> (A,0)
            BOOST_TEST(laplacian(2*i, 2*i + 1) == 1);
        }
        else
        {
            // Edge (A,i) -> (B,i)
            BOOST_TEST(laplacian(2*i + 1, 2*i) == i);

            // Edge (B,i) -> (A,i)
            BOOST_TEST(laplacian(2*i, 2*i + 1) == i);
        }

        // Edges (A,i) <-> (A,i+1) and (B,i) <-> (B,i+1)
        if (i < 5)
        {
            BOOST_TEST(laplacian(2*i + 2, 2*i) == i + 1);
            BOOST_TEST(laplacian(2*i, 2*i + 2) == i + 1);
            BOOST_TEST(laplacian(2*i + 3, 2*i + 1) == i + 1);
            BOOST_TEST(laplacian(2*i + 1, 2*i + 3) == i + 1);
        }
    }

    delete graph;
}


