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
#include "../include/graphs/triangularPrism.hpp"
#include "../include/sample.hpp"

/*
 * Samples points uniformly from the specificity vs. times ratio region in 
 * the three-conformation Cas9 model (triangular prism graph).  
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
typedef number<mpfr_float_backend<200>, et_off> mpfr_200_noet;
typedef Matrix<mpfr_200_noet, Dynamic, Dynamic> MatrixX200;
typedef Matrix<mpfr_200_noet, Dynamic, 1> VectorX200;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

MatrixX200 computeCleavageStats(const Ref<const VectorX200>& params)
{
    /*
     * Compute the specificity and times ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    using boost::multiprecision::pow;

    // Array of DNA/RNA match parameters
    std::array<mpfr_200_noet, 12> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(1));
    match_params[2] = pow(10.0, params(0));
    match_params[3] = pow(10.0, params(1));
    match_params[4] = pow(10.0, params(0));
    match_params[5] = pow(10.0, params(1));
    match_params[6] = pow(10.0, params(4));
    match_params[7] = pow(10.0, params(5));
    match_params[8] = pow(10.0, params(6));
    match_params[9] = pow(10.0, params(7));
    match_params[10] = pow(10.0, params(8));
    match_params[11] = pow(10.0, params(9));

    // Array of DNA/RNA mismatch parameters
    std::array<mpfr_200_noet, 12> mismatch_params;
    mismatch_params[0] = pow(10.0, params(2));
    mismatch_params[1] = pow(10.0, params(3));
    mismatch_params[2] = pow(10.0, params(2));
    mismatch_params[3] = pow(10.0, params(3));
    mismatch_params[4] = pow(10.0, params(2));
    mismatch_params[5] = pow(10.0, params(3));
    mismatch_params[6] = pow(10.0, params(4));
    mismatch_params[7] = pow(10.0, params(5));
    mismatch_params[8] = pow(10.0, params(6));
    mismatch_params[9] = pow(10.0, params(7));
    mismatch_params[10] = pow(10.0, params(8));
    mismatch_params[11] = pow(10.0, params(9));

    // Populate each rung with DNA/RNA match parameters
    TriangularPrismGraph<mpfr_200_noet>* model = new TriangularPrismGraph<mpfr_200_noet>(length);
    model->setStartLabels(
        match_params[6], match_params[7], match_params[8], match_params[9],
        match_params[10], match_params[11]
    );
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<mpfr_200_noet, 2, 1> match_data = model->computeCleavageStatsByInverse(1, 1).array().log10().matrix();
    MatrixX200 stats(length + 1, 2);
    stats(0, 0) = match_data(0);
    stats(0, 1) = match_data(1);

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= length; ++j)
    {
        model->setRungLabels(length - j, mismatch_params);
        Matrix<mpfr_200_noet, 2, 1> mismatch_data = model->computeCleavageStatsByInverse(1, 1).array().log10().matrix();
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
    MatrixX200 params(n, 10);
    std::pair<MatrixX200, MatrixX200> data;
    try
    {
        data = sampleFromConvexPolytopeTriangulation<mpfr_200_noet>(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }
    params.block(0, 0, n, 9) = data.second;
    params.col(9) = data.second.col(4) + data.second.col(6) + data.second.col(8) - data.second.col(5) - data.second.col(7);

    // Compute specificities and times ratios
    MatrixX200 probs(n, length + 1);
    MatrixX200 times(n, length + 1);
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixX200 stats = computeCleavageStats(params.row(i));
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

    // Write matrix of cleavage specificities
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

    // Write matrix of cleavage times ratios
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
