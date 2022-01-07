#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <graphs/line.hpp>
#include "../include/sample.hpp"

/*
 * Computes cleavage probabilities and unbinding rates with respect to 
 * single-mismatch substrates for the line-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/lineSingleMismatch [SAMPLING POLYTOPE .delv FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/7/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

template <typename T>
MatrixXd computeStats(const Ref<const Matrix<double, Dynamic, 1> >& params)
{
    /*
     * Compute the cleavage probabilities and unbinding rates with respect to 
     * single-mismatch substrates. 
     */
    // Array of DNA/RNA match parameters
    std::pair<T, T> match_params = std::make_pair(
        static_cast<T>(std::pow(10.0, params(0))),
        static_cast<T>(std::pow(10.0, params(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<T, T> mismatch_params = std::make_pair(
        static_cast<T>(std::pow(10.0, params(2))),
        static_cast<T>(std::pow(10.0, params(3)))
    );

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_params);
    
    // Compute cleavage probability and mean first-passage time to unbound state
    T prob = model->getUpperExitProb(1, 1);
    T rate = model->getLowerExitRate(1);
    MatrixXd stats(length + 1, 2);
    stats(0, 0) = static_cast<double>(prob);
    stats(0, 1) = static_cast<double>(rate);

    // Introduce single mismatches and re-compute cleavage probability
    // and mean first-passage time
    for (unsigned j = 0; j < length; ++j)
    {
        for (unsigned k = 0; k < length; ++k)
            model->setEdgeLabels(k, match_params);
        model->setEdgeLabels(j, mismatch_params);
        prob = model->getUpperExitProb(1, 1);
        rate = model->getLowerExitRate(1);
        stats(j + 1, 0) = static_cast<double>(prob);
        stats(j + 1, 1) = static_cast<double>(rate);
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[3], "%u", &n);
    MatrixXd vertices, params;
    try
    {
        std::tie(vertices, params) = sampleFromConvexPolytopeTriangulation<double>(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // Compute cleavage probabilities and unbinding rates
    MatrixXd probs(n, length + 1);
    MatrixXd rates(n, length + 1);
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixXd stats = computeStats<number<mpfr_float_backend<100> > >(params.row(i)).transpose();
        probs.row(i) = stats.row(0);
        rates.row(i) = stats.row(1);
    }

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[2] << "-params.tsv";
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (samplefile.is_open())
    {
        for (unsigned i = 0; i < params.rows(); i++)
        {
            for (unsigned j = 0; j < params.cols() - 1; j++)
            {
                samplefile << params(i,j) << "\t";
            }
            samplefile << params(i,params.cols()-1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage probabilities
    oss << argv[2] << "-probs.tsv";
    std::ofstream probsfile(oss.str());
    probsfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (probsfile.is_open())
    {
        for (unsigned i = 0; i < probs.rows(); i++)
        {
            for (unsigned j = 0; j < probs.cols() - 1; j++)
            {
                probsfile << probs(i,j) << "\t";
            }
            probsfile << probs(i,probs.cols()-1) << std::endl;
        }
    }
    probsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of unconditional unbinding rates
    oss << argv[2] << "-rates.tsv";
    std::ofstream ratesfile(oss.str());
    ratesfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (ratesfile.is_open())
    {
        for (unsigned i = 0; i < rates.rows(); i++)
        {
            for (unsigned j = 0; j < rates.cols() - 1; j++)
            {
                ratesfile << rates(i,j) << "\t";
            }
            ratesfile << rates(i,rates.cols()-1) << std::endl;
        }
    }
    ratesfile.close();
   
    return 0;
}
