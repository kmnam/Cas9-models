#define BOOST_TEST_MODULE testLineGraph
#define BOOST_TEST_DYN_LINK
#include <iostream>
#include <string>
#include <sstream>
#include <array>
#include <utility>
#include <boost/test/included/unit_test.hpp>
#include "../../include/graphs/line.hpp"

/*
 * Test module for the LineGraph class.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/11/2021
 */
BOOST_AUTO_TEST_CASE(testUnity)
{
    /*
     * Test all public methods on the line graph of length 5 with 
     * labels set to 1. 
     */
    LineGraph<double>* graph = new LineGraph<double>(5);

    // Test that the graph has 6 vertices
    MatrixXd laplacian = graph->getLaplacian();
    BOOST_TEST(laplacian.rows() == 6);
    BOOST_TEST(laplacian.cols() == 6);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i < 5; ++i)
    {
        BOOST_TEST(laplacian(i, i + 1) == 1.0);
        BOOST_TEST(laplacian(i + 1, i) == 1.0);
    }

    // Compute the probability of upper exit, with exit rates set to 1
    double prob = graph->computeUpperExitProb(1, 1);
    BOOST_TEST(std::abs(prob - (1.0 / 7.0) < 1e-20));

    // Compute the reciprocal of the mean first passage time to lower exit,
    // with exit rates set to 1
    double rate = graph->computeLowerExitRate(1, 1);
    BOOST_TEST(std::abs(rate - (42.0 / 91.0) < 1e-20));

    delete graph;
}

BOOST_AUTO_TEST_CASE(testOneToFive)
{
    /*
     * Test all public methods on the line graph of length 5 with 
     * labels set to 1, 2, ..., 5. 
     */
    LineGraph<double>* graph = new LineGraph<double>(5);
    for (unsigned i = 0; i < 5; ++i)
    {
        double k = i + 1;
        std::array<double, 2> labels = {k, k};
        graph->setLabels(i, labels);
    }

    // Test that the graph has 6 vertices
    MatrixXd laplacian = graph->getLaplacian();
    BOOST_TEST(laplacian.rows() == 6);
    BOOST_TEST(laplacian.cols() == 6);

    // Test that the graph has the correct set of edges
    for (unsigned i = 0; i < 5; ++i)
    {
        BOOST_TEST(laplacian(i, i + 1) == i + 1);
        BOOST_TEST(laplacian(i + 1, i) == i + 1);
    }

    // Compute the probability of upper exit, with exit rates set to 1
    double prob = graph->computeUpperExitProb(1, 1);
    BOOST_TEST(std::abs(prob - (1.0 / (1 + 1.0/1 + (1.0/1)*(1.0/2) + (1.0/1)*(1.0/2)*(2.0/3) + (1.0/1)*(1.0/2)*(2.0/3)*(3.0/4) + (1.0/1)*(1.0/2)*(2.0/3)*(3.0/4)*(4.0/5) + (1.0/1)*(1.0/2)*(2.0/3)*(3.0/4)*(4.0/5)*(5.0/1))) < 1e-20));

    // Compute the reciprocal of the mean first passage time to lower exit,
    // with exit rates set to 1
    double rate = graph->computeLowerExitRate(1, 1);
    double numer = (
        (1*1*2*3*4*5 + 1*1*2*3*4*1 + 1*1*2*3*5*1 + 1*1*2*4*5*1 + 1*1*3*4*5*1 + 1*2*3*4*5*1) *
        (1*1*2*3*4*5 + 1*1*2*3*4*1 + 1*1*2*3*5*1 + 1*1*2*4*5*1 + 1*1*3*4*5*1 + 1*2*3*4*5*1 + 1*2*3*4*5*1)
    );
    double denom = (
        (2*3*4*5*1 + 1*3*4*5*1 + 1*2*4*5*1 + 1*2*3*5*1 + 1*2*3*4*1 + 1*2*3*4*5) * (1*1*2*3*4*5 + 1*1*2*3*4*1 + 1*1*2*3*5*1 + 1*1*2*4*5*1 + 1*1*3*4*5*1 + 1*2*3*4*5*1) +
        (1*3*4*5*1 + 1*2*4*5*1 + 1*2*3*5*1 + 1*2*3*4*1 + 1*2*3*4*5) * (1*1*2*3*4*5 + 1*1*2*3*4*1 + 1*1*2*3*5*1 + 1*1*2*4*5*1 + 1*1*3*4*5*1) +
        (1*2*4*5*1 + 1*2*3*5*1 + 1*2*3*4*1 + 1*2*3*4*5) * (1*1*2*3*4*5 + 1*1*2*3*4*1 + 1*1*2*3*5*1 + 1*1*2*4*5*1) + 
        (1*2*3*5*1 + 1*2*3*4*1 + 1*2*3*4*5) * (1*1*2*3*4*5 + 1*1*2*3*4*1 + 1*1*2*3*5*1) + 
        (1*2*3*4*1 + 1*2*3*4*5) * (1*1*2*3*4*5 + 1*1*2*3*4*1) +
        (1*2*3*4*5) * (1*1*2*3*4*5)
    );
    BOOST_TEST(std::abs(rate - (numer / denom) < 1e-20));

    // Compute the probability of upper exit again with the Chebotarev-Agaev
    // recurrence
    LabeledDigraph<double>* copy = graph->copy();
    copy->addNode("lower");
    copy->addNode("upper");
    copy->addEdge("0", "lower", 1.0);
    copy->addEdge("5", "upper", 1.0);
    MatrixXd copy_laplacian = copy->getLaplacian();
    MatrixXd forest_matrix_1 = copy->getSpanningForestMatrix(6 + 2 - 3);
    MatrixXd forest_matrix_2 = copy->getSpanningForestMatrix(6 + 2 - 2);
    double copy_prob = forest_matrix_2(0, 6 + 2 - 1) / forest_matrix_2.row(0).sum();
    BOOST_TEST(std::abs(prob - copy_prob) < 1e-20);

    // Compute the rate of lower exit again with the Chebotarev-Agaev recurrence
    double copy_time = 0.0;
    for (unsigned k = 0; k <= 5; ++k)
        copy_time += (forest_matrix_1(0, k) * forest_matrix_2(k, 6 + 2 - 2));
    double copy_rate = (forest_matrix_2(0, 6 + 2 - 2) * forest_matrix_2.row(0).sum()) / copy_time;
    BOOST_TEST(std::abs(rate - copy_rate) < 1e-20);

    delete graph;
    delete copy; 
}


