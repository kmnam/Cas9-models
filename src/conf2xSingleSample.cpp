#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/random.hpp>
#include "../include/graphs/grid.hpp"
#include "../include/sample.hpp"

/*
 * Samples points uniformly from the specificity vs. times ratio region in 
 * the two-conformation Cas9 model (grid graph) with state-dependent 
 * extension rates against single-mismatch substrates.  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/8/2019
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::et_off;
typedef number<mpfr_float_backend<30>, et_off> mpfr_30_noet;
typedef Matrix<mpfr_30_noet, Dynamic, Dynamic> MatrixX30;
typedef Matrix<mpfr_30_noet, Dynamic, 1> VectorX30;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

MatrixX30 computeCleavageStats(const Ref<const VectorX30>& params)
{
    /*
     * Compute the specificity and times ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    // Array of DNA/RNA match parameters
    std::array<mpfr_30_noet, 6> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(4));
    match_params[2] = pow(10.0, params(2));
    match_params[3] = pow(10.0, params(6));
    match_params[4] = pow(10.0, params(8));
    match_params[5] = pow(10.0, params(10));

    // Array of DNA/RNA mismatch parameters
    std::array<mpfr_30_noet, 6> mismatch_params;
    mismatch_params[0] = pow(10.0, params(1));
    mismatch_params[1] = pow(10.0, params(5));
    mismatch_params[2] = pow(10.0, params(3));
    mismatch_params[3] = pow(10.0, params(7));
    mismatch_params[4] = pow(10.0, params(9));
    mismatch_params[5] = pow(10.0, params(11));

    // Populate each rung with DNA/RNA match parameters
    GridGraph<mpfr_30_noet>* model = new GridGraph<mpfr_30_noet>(length);
    model->setStartLabels(match_params[4], match_params[5]);
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<mpfr_30_noet, 2, 1> match_data = model->computeCleavageStats(1, 1).array().log10().matrix();
    MatrixX30 stats(length + 1, 2);
    stats(0, 0) = match_data(0);
    stats(0, 1) = match_data(1);

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= length; ++j)
    {
        model->setRungLabels(length - j, mismatch_params);
        Matrix<mpfr_30_noet, 2, 1> mismatch_data = model->computeCleavageStats(1, 1).array().log10().matrix();
        stats(j, 0) = mismatch_data(0);
        stats(j, 1) = mismatch_data(1);    
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[3], "%u", &n);
    MatrixX30 params(n, 12);
    std::pair<MatrixX30, MatrixX30> data;
    try
    {
        data = sampleFromConvexPolytopeTriangulation<mpfr_30_noet>(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }
    // Parameters are ordered as: a, a', b, b', c, c', d, d', k, k', l, l'
    params.block(0, 0, n, 4) = data.second.block(0, 0, n, 4);
    params.block(0, 6, n, 5) = data.second.block(0, 4, n, 5);
    params.col(4) = data.second.col(0) + data.second.col(4) - data.second.col(2);
    params.col(5) = data.second.col(1) + data.second.col(5) - data.second.col(3);
    params.col(11) = data.second.col(7) + data.second.col(8) - data.second.col(6);

    // Compute cleavage statistics
    MatrixX30 probs(n, length + 1);
    MatrixX30 times(n, length + 1);
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixX30 stats = computeCleavageStats(params.row(i));
        probs.row(i) = stats.col(0).transpose();
        times.row(i) = stats.col(1).transpose();
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

    // Write matrix of mean first passage times
    oss << argv[2] << "-times.tsv";
    std::ofstream timesfile(oss.str());
    timesfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (timesfile.is_open())
    {
        for (unsigned i = 0; i < times.rows(); i++)
        {
            for (unsigned j = 0; j < times.cols() - 1; j++)
            {
                timesfile << times(i,j) << "\t";
            }
            timesfile << times(i,times.cols()-1) << std::endl;
        }
    }
    timesfile.close();
   
    return 0;
}