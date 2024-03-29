#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "../include/graphs/grid.hpp"
#include "../include/sample.hpp"

/*
 * Computes cleavage probabilities and unbinding rates with respect to 
 * distal-mismatch substrates for the grid-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/gridDistalMismatch [SAMPLING POLYTOPE .delv FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     2/24/2021
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

template <typename T>
Matrix<double, Dynamic, Dynamic> computeStats(const Ref<const Matrix<double, Dynamic, 1> >& params)
{
    /*
     * Compute the cleavage probabilities and unbinding rates with respect to 
     * single-mismatch substrates. 
     */
    // Array of DNA/RNA match parameters
    std::array<T, 6> match_params;
    match_params[0] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[1] = static_cast<T>(std::pow(10.0, params(1)));
    match_params[2] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[3] = static_cast<T>(std::pow(10.0, params(1)));
    match_params[4] = static_cast<T>(std::pow(10.0, params(4)));
    match_params[5] = static_cast<T>(std::pow(10.0, params(5)));

    // Array of DNA/RNA mismatch parameters
    std::array<T, 6> mismatch_params;
    mismatch_params[0] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[1] = static_cast<T>(std::pow(10.0, params(3)));
    mismatch_params[2] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[3] = static_cast<T>(std::pow(10.0, params(3)));
    mismatch_params[4] = static_cast<T>(std::pow(10.0, params(4)));
    mismatch_params[5] = static_cast<T>(std::pow(10.0, params(5)));

    // Populate each rung with DNA/RNA match parameters
    GridGraph<T>* model = new GridGraph<T>(length);
    model->setStartLabels(match_params[4], match_params[5]);
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time to unbound state
    std::pair<T, T> data = model->computeExitStats(1, 1, 1, 0);
    Matrix<double, Dynamic, Dynamic> stats(length + 1, 2);
    stats(0, 0) = static_cast<double>(data.first);
    stats(0, 1) = static_cast<double>(data.second);

    // Introduce single mismatches and re-compute cleavage probability
    // and mean first passage time
    for (int j = length - 1; j >= 0; --j)
    {
        model->setRungLabels(j, mismatch_params);
        data = model->computeExitStats(1, 1, 1, 0);
        stats(length-j, 0) = static_cast<double>(data.first);
        stats(length-j, 1) = static_cast<double>(data.second);
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[3], "%u", &n);
    Matrix<double, Dynamic, Dynamic> vertices, params;
    try
    {
        std::tie(vertices, params) = sampleFromConvexPolytopeTriangulation<double>(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // Compute cleavage probabilities and unbinding rates
    Matrix<double, Dynamic, Dynamic> probs(n, length + 1);
    Matrix<double, Dynamic, Dynamic> rates(n, length + 1);
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<double, Dynamic, Dynamic> stats = computeStats<number<mpfr_float_backend<50> > >(params.row(i));
        probs.row(i) = stats.col(0).transpose();
        rates.row(i) = stats.col(1).transpose();
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

    // Write matrix of unbinding rates
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
