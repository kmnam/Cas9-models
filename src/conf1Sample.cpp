#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <random>
#include <Eigen/Dense>
#include "../include/graphs/line.hpp"
#include "../include/sample.hpp"

/*
 * Samples points uniformly from the specificity vs. speed ratio region in 
 * the single-conformation Cas9 model (line graph).  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/21/2019
 */
using namespace Eigen;

const unsigned length = 20;

// Instantiate random number generator 
std::mt19937 rng(1234567890);

MatrixXd computeCleavageStats(const Ref<const VectorXd>& params)
{
    /*
     * Compute the specificity and speed ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    // Array of DNA/RNA match parameters
    std::array<double, 2> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(1));

    // Array of DNA/RNA mismatch parameters
    std::array<double, 2> mismatch_params;
    mismatch_params[0] = pow(10.0, params(2));
    mismatch_params[1] = pow(10.0, params(3));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<double>* model = new LineGraph<double>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<double, 2, 1> match_data = model->computeCleavageStats(1, 1).array().log10().matrix();

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    MatrixXd stats(length, 2);
    for (unsigned j = 1; j <= length; ++j)
    {
        model->setLabels(length - j, mismatch_params);
        Matrix<double, 2, 1> mismatch_data = model->computeCleavageStats(1, 1).array().log10().matrix();
        
        // Compute the specificity and speed ratio
        double specificity = match_data(0) - mismatch_data(0);
        double speed_ratio = mismatch_data(1) - match_data(1);
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
    MatrixXd params(n, 4);
    try
    {
        params = sampleFromConvexPolytopeTriangulation(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    // Run the boundary-finding algorithm
    MatrixXd specs(n, length);
    MatrixXd speed(n, length);
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixXd stats = computeCleavageStats(params.row(i));
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
    oss.clear();
    oss.str(std::string());
   
    return 0;
}
