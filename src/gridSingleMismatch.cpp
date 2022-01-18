#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <graphs/grid.hpp>
#include "../include/sample.hpp"

/*
 * Computes cleavage probabilities and unbinding rates with respect to 
 * single-mismatch substrates for the grid-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/gridSingleMismatch [SAMPLING POLYTOPE .delv FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/18/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
typedef number<mpfr_float_backend<100> > PreciseType;

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

    // Array of conformational change parameters
    std::pair<T, T> switch_params = std::make_pair(
        static_cast<T>(std::pow(10.0, params(4))), 
        static_cast<T>(std::pow(10.0, params(5)))
    ); 

    // Populate each rung with DNA/RNA match parameters
    GridGraph<T, T>* model = new GridGraph<T, T>(length);
    model->setZerothLabels(switch_params.first, switch_params.second);
    for (unsigned j = 0; j < length; ++j)
    {
        std::array<T, 6> labels; 
        labels[0] = match_params.first; 
        labels[1] = match_params.second; 
        labels[2] = match_params.first; 
        labels[3] = match_params.second; 
        labels[4] = switch_params.first; 
        labels[5] = switch_params.second; 
        model->setRungLabels(j, labels); 
    } 
    
    // Compute cleavage probability, unbinding rate, and cleavage rate 
    std::tuple<T, T, T> exit_stats = model->getExitStats(1, 1); 
    T prob = std::get<0>(exit_stats); 
    T unbind_rate = std::get<1>(exit_stats); 
    T cleave_rate = std::get<2>(exit_stats); 
    MatrixXd stats(length + 1, 3);
    stats(0, 0) = static_cast<double>(prob);
    stats(0, 1) = static_cast<double>(unbind_rate);
    stats(0, 2) = static_cast<double>(cleave_rate); 

    // Introduce single mismatches and re-compute cleavage probability
    // and mean first-passage time
    for (unsigned j = 0; j < length; ++j)
    {
        std::array<T, 6> labels; 
        for (unsigned k = 0; k < length; ++k)
        {
            labels[0] = match_params.first; 
            labels[1] = match_params.second; 
            labels[2] = match_params.first; 
            labels[3] = match_params.second; 
            labels[4] = switch_params.first; 
            labels[5] = switch_params.second; 
            model->setRungLabels(k, labels); 
        }
        labels[0] = mismatch_params.first; 
        labels[1] = mismatch_params.second; 
        labels[2] = mismatch_params.first; 
        labels[3] = mismatch_params.second; 
        model->setRungLabels(j, labels); 
        exit_stats = model->getExitStats(1, 1); 
        prob = std::get<0>(exit_stats); 
        unbind_rate = std::get<1>(exit_stats); 
        cleave_rate = std::get<2>(exit_stats); 
        stats(j + 1, 0) = static_cast<double>(prob);
        stats(j + 1, 1) = static_cast<double>(unbind_rate);
        stats(j + 1, 2) = static_cast<double>(cleave_rate); 
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
    MatrixXd unbind_rates(n, length + 1);
    MatrixXd cleave_rates(n, length + 1); 
    for (unsigned i = 0; i < n; ++i)
    {
        MatrixXd stats = computeStats<PreciseType>(params.row(i)).transpose(); 
        probs.row(i) = stats.row(0);
        unbind_rates.row(i) = stats.row(1);
        cleave_rates.row(i) = stats.row(2); 
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
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                probsfile << probs(i,j) << "\t";
            }
            probsfile << probs(i, length) << std::endl; 
        }
    }
    probsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of unconditional unbinding rates
    oss << argv[2] << "-unbind-rates.tsv";
    std::ofstream unbindfile(oss.str());
    unbindfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (unbindfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                unbindfile << unbind_rates(i,j) << "\t";
            }
            unbindfile << unbind_rates(i, length) << std::endl; 
        }
    }
    unbindfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of unconditional unbinding rates
    oss << argv[2] << "-cleave-rates.tsv";
    std::ofstream cleavefile(oss.str());
    cleavefile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (cleavefile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                cleavefile << cleave_rates(i,j) << "\t";
            }
            cleavefile << cleave_rates(i, length) << std::endl; 
        }
    }
    cleavefile.close();
   
    return 0;
}
