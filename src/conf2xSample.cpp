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
 * Samples points uniformly from the specificity vs. speed ratio region in 
 * the two-conformation Cas9 model (grid graph).  
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
     * Compute the specificity and speed ratio with respect to the 
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

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    MatrixX30 stats(length, 2);
    for (unsigned j = 1; j <= length; ++j)
    {
        model->setRungLabels(length - j, mismatch_params);
        Matrix<mpfr_30_noet, 2, 1> mismatch_data = model->computeCleavageStats(1, 1).array().log10().matrix();
        
        // Compute the specificity and speed ratio
        mpfr_30_noet specificity = match_data(0) - mismatch_data(0);
        mpfr_30_noet speed_ratio = mismatch_data(1) - match_data(1);
        stats(j-1, 0) = specificity;
        stats(j-1, 1) = speed_ratio;
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[3], "%u", &n);
    mpfr_30_noet lower, upper;
    sscanf(argv[4], "%lf", &lower);
    sscanf(argv[5], "%lf", &upper);
    MatrixX30 params(n, 12);
    unsigned nsample = 0;

    while (nsample < n)
    {
        std::pair<MatrixX30, MatrixX30> data;
        try
        {
            data = sampleFromConvexPolytopeTriangulation<mpfr_30_noet>(argv[1], n - nsample, rng);
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw;
        }
        for (unsigned i = 0; i < data.second.rows(); ++i)
        {
            VectorX30 row(12);    // Ordered as: a, a', b, b', c, c', d, d', k, k', l, l'
            row(0) = data.second(i,0); 
            row(1) = data.second(i,1);
            row(2) = data.second(i,2);
            row(3) = data.second(i,3);
            row(4) = data.second(i,0) + data.second(i,4) - data.second(i,2);
            row(5) = data.second(i,1) + data.second(i,5) - data.second(i,3);
            row(6) = data.second(i,4);
            row(7) = data.second(i,5);
            row(8) = data.second(i,6);
            row(9) = data.second(i,7);
            row(10) = data.second(i,8);
            row(11) = data.second(i,7) + data.second(i,8) - data.second(i,6);

            // Check that the extra three parameters lie within the given bounds
            if (row(4) > lower && row(4) < upper && row(5) > lower && row(5) < upper && row(11) > lower && row(11) < upper)
            {
                params.row(nsample) = row;
                nsample++;
            }
        }
    }

    // Compute specificities and speed ratios 
    MatrixX30 specs(n, length);
    MatrixX30 speed(n, length);
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixX30 stats = computeCleavageStats(params.row(i));
        specs.row(i) = stats.col(0).transpose();
        speed.row(i) = stats.col(1).transpose();
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
    oss << argv[2] << "-specificities.tsv";
    std::ofstream specfile(oss.str());
    specfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (specfile.is_open())
    {
        for (unsigned i = 0; i < specs.rows(); i++)
        {
            for (unsigned j = 0; j < specs.cols() - 1; j++)
            {
                specfile << specs(i,j) << "\t";
            }
            specfile << specs(i,specs.cols()-1) << std::endl;
        }
    }
    specfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage speed ratios
    oss << argv[2] << "-speed-ratios.tsv";
    std::ofstream speedfile(oss.str());
    speedfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (speedfile.is_open())
    {
        for (unsigned i = 0; i < speed.rows(); i++)
        {
            for (unsigned j = 0; j < speed.cols() - 1; j++)
            {
                speedfile << speed(i,j) << "\t";
            }
            speedfile << speed(i,speed.cols()-1) << std::endl;
        }
    }
    speedfile.close();
   
    return 0;
}