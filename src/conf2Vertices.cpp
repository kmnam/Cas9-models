#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <tuple>
#include <random>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include "../include/graphs/grid.hpp"
#include "../include/sample.hpp"

/*
 * Computes the specificity and speed ratio at the vertices of the parameter
 * polytope for the two-conformation Cas9 model (grid graph).
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/3/2019
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::et_off;
typedef number<mpfr_float_backend<1000>, et_on> mpfr_1000;
typedef Matrix<mpfr_1000, Dynamic, 1> VectorXT;
typedef Matrix<mpfr_1000, Dynamic, Dynamic> MatrixXT;

const unsigned length = 20;

std::mt19937 rng(1234567890);

MatrixXT computeCleavageStats(const Ref<const VectorXT>& params)
{
    /*
     * Compute the specificity and speed ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    using boost::multiprecision::pow;

    // Array of DNA/RNA match parameters
    std::array<mpfr_1000, 6> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(1));
    match_params[2] = pow(10.0, params(0));
    match_params[3] = pow(10.0, params(1));
    match_params[4] = pow(10.0, params(4));
    match_params[5] = pow(10.0, params(5));

    // Array of DNA/RNA mismatch parameters
    std::array<mpfr_1000, 6> mismatch_params;
    mismatch_params[0] = pow(10.0, params(2));
    mismatch_params[1] = pow(10.0, params(3));
    mismatch_params[2] = pow(10.0, params(2));
    mismatch_params[3] = pow(10.0, params(3));
    mismatch_params[4] = pow(10.0, params(4));
    mismatch_params[5] = pow(10.0, params(5));

    // Populate each rung with DNA/RNA match parameters
    GridGraph<mpfr_1000>* model = new GridGraph<mpfr_1000>(length);
    model->setStartLabels(match_params[4], match_params[5]);
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<mpfr_1000, 2, 1> match_data = model->computeCleavageStats(1, 1).array().log10().matrix();

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    MatrixXT stats(length, 2);
    for (unsigned j = 1; j <= length; ++j)
    {
        model->setRungLabels(length - j, mismatch_params);
        Matrix<mpfr_1000, 2, 1> mismatch_data = model->computeCleavageStats(1, 1).array().log10().matrix();
        
        // Compute the specificity and speed ratio
        mpfr_1000 specificity = match_data(0) - mismatch_data(0);
        mpfr_1000 speed_ratio = mismatch_data(1) - match_data(1);
        stats(j-1, 0) = specificity;
        stats(j-1, 1) = speed_ratio;
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    MatrixXd v;
    MatrixXd p;
    try
    {
        std::tie(v, p) = sampleFromConvexPolytopeTriangulation(argv[1], 1, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }
    unsigned n = v.rows();
    unsigned d = v.cols();
    MatrixXT vertices(n, d);
    for (unsigned i = 0; i < n; ++i)
    {
        for (unsigned j = 0; j < d; ++j)
        {
            std::stringstream ss;
            ss << std::setprecision(std::numeric_limits<double>::max_digits10) << v(i,j);
            mpfr_1000 x(ss.str());
            vertices(i,j) = x;
        }
    }

    // Run the boundary-finding algorithm
    MatrixXT specs(n, length);
    MatrixXT speed(n, length);
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixXT stats = computeCleavageStats(vertices.row(i));
        specs.row(i) = stats.col(0).transpose();
        speed.row(i) = stats.col(1).transpose();
    }

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[2] << "-vertices.tsv";
    std::ofstream vertexfile(oss.str());
    vertexfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (vertexfile.is_open())
    {
        for (unsigned i = 0; i < vertices.rows(); i++)
        {
            for (unsigned j = 0; j < vertices.cols() - 1; j++)
            {
                vertexfile << vertices(i,j) << "\t";
            }
            vertexfile << vertices(i,vertices.cols()-1) << std::endl;
        }
    }
    vertexfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage specificities
    oss << argv[2] << "-vertex-specificities.tsv";
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
    oss << argv[2] << "-vertex-speed-ratios.tsv";
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
    oss.clear();
    oss.str(std::string());
   
    return 0;
}
