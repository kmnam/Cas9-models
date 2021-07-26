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
#include "../include/graphs/line.hpp"
#include "../include/sample.hpp"

/*
 * Computes cleavage probabilities and unbinding rates with respect to 
 * distal-mismatch substrates for the line-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/lineDistalMismatch [SAMPLING POLYTOPE .delv FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     7/26/2021
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
    std::array<T, 2> match_params;
    match_params[0] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[1] = static_cast<T>(std::pow(10.0, params(1)));

    // Array of DNA/RNA mismatch parameters
    std::array<T, 2> mismatch_params;
    mismatch_params[0] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[1] = static_cast<T>(std::pow(10.0, params(3)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T>* model = new LineGraph<T>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time to unbound state
    T prob = model->computeUpperExitProb(1, 1);
    T uncond_rate = model->computeLowerExitRate(1, 0);
    T cond_lower_exit = model->computeLowerExitRate(1, 1); 
    T cond_upper_exit = model->computeUpperExitRate(1, 1); 
    Matrix<double, Dynamic, Dynamic> stats(length + 1, 4);
    stats(0, 0) = static_cast<double>(prob);
    stats(0, 1) = static_cast<double>(uncond_rate);
    stats(0, 2) = static_cast<double>(cond_lower_exit);
    stats(0, 3) = static_cast<double>(cond_upper_exit);  

    // Introduce single mismatches and re-compute cleavage probability
    // and mean first passage time
    for (int j = length - 1; j >= 0; --j)
    {
        model->setLabels(j, mismatch_params);
        prob = model->computeUpperExitProb(1, 1);
        uncond_rate = model->computeLowerExitRate(1, 0);
        cond_lower_exit = model->computeLowerExitRate(1, 1); 
        cond_upper_exit = model->computeUpperExitRate(1, 1); 
        stats(j, 0) = static_cast<double>(prob);
        stats(j, 1) = static_cast<double>(uncond_rate);
        stats(j, 2) = static_cast<double>(cond_lower_exit);
        stats(j, 3) = static_cast<double>(cond_upper_exit);  
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
    Matrix<double, Dynamic, Dynamic> uncond_rates(n, length + 1);
    Matrix<double, Dynamic, Dynamic> cond_lower_rates(n, length + 1);
    Matrix<double, Dynamic, Dynamic> cond_upper_rates(n, length + 1); 
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<double, Dynamic, Dynamic> stats = computeStats<number<mpfr_float_backend<100> > >(params.row(i));
        probs.row(i) = stats.col(0).transpose();
        uncond_rates.row(i) = stats.col(1).transpose();
        cond_lower_rates.row(i) = stats.col(2).transpose();
        cond_upper_rates.row(i) = stats.col(3).transpose();
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
    oss << argv[2] << "-uncondRates.tsv";
    std::ofstream ratesfile(oss.str());
    ratesfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (ratesfile.is_open())
    {
        for (unsigned i = 0; i < uncond_rates.rows(); i++)
        {
            for (unsigned j = 0; j < uncond_rates.cols() - 1; j++)
            {
                ratesfile << uncond_rates(i,j) << "\t";
            }
            ratesfile << uncond_rates(i, uncond_rates.cols()-1) << std::endl;
        }
    }
    ratesfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of conditional unbinding rates
    oss << argv[2] << "-condLowerRates.tsv";
    std::ofstream lowerfile(oss.str());
    lowerfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (lowerfile.is_open())
    {
        for (unsigned i = 0; i < cond_lower_rates.rows(); i++)
        {
            for (unsigned j = 0; j < cond_lower_rates.cols() - 1; j++)
            {
                lowerfile << cond_lower_rates(i,j) << "\t";
            }
            lowerfile << cond_lower_rates(i, cond_lower_rates.cols()-1) << std::endl;
        }
    }
    lowerfile.close();
    oss.clear(); 
    oss.str(std::string());

    // Write matrix of conditional cleavage rates
    oss << argv[2] << "-condUpperRates.tsv";
    std::ofstream upperfile(oss.str());
    upperfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (upperfile.is_open())
    {
        for (unsigned i = 0; i < cond_upper_rates.rows(); i++)
        {
            for (unsigned j = 0; j < cond_upper_rates.cols() - 1; j++)
            {
                upperfile << cond_upper_rates(i,j) << "\t";
            }
            upperfile << cond_upper_rates(i, cond_upper_rates.cols()-1) << std::endl;
        }
    }
    upperfile.close();
    oss.clear(); 
    oss.str(std::string()); 
   
    return 0;
}
