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
 * Computes cleavage probabilities, unbinding rates, and cleavage rates with
 * respect to double-mismatch substrates for the grid-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/gridDoubleMismatch [SAMPLING POLYTOPE .delv FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/1/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10; 
typedef number<mpfr_float_backend<100> > PreciseType;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

template <typename T>
Matrix<T, Dynamic, 6> computeStats(const Ref<const Matrix<double, Dynamic, 1> >& params)
{
    /*
     * Compute the cleavage probabilities, unbinding rates, and cleavage rates
     * of randomly parametrized Cas9 enzymes with respect to double-mismatch
     * substrates. 
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
    std::array<T, 6> match_labels, mismatch_labels; 
    match_labels[0] = match_params.first; 
    match_labels[1] = match_params.second; 
    match_labels[2] = match_params.first; 
    match_labels[3] = match_params.second; 
    match_labels[4] = switch_params.first; 
    match_labels[5] = switch_params.second;
    mismatch_labels[0] = mismatch_params.first; 
    mismatch_labels[1] = mismatch_params.second; 
    mismatch_labels[2] = mismatch_params.first; 
    mismatch_labels[3] = mismatch_params.second; 
    mismatch_labels[4] = switch_params.first; 
    mismatch_labels[5] = switch_params.second; 
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_labels);
    
    // Compute cleavage probability, unbinding rate, and cleavage rate 
    std::tuple<T, T, T> exit_stats = model->getExitStats(1, 1); 
    Matrix<T, Dynamic, 6> stats = Matrix<T, Dynamic, 6>::Zero(length, 6);
    stats(0, 0) = std::get<0>(exit_stats); 
    stats(0, 1) = std::get<1>(exit_stats); 
    stats(0, 2) = std::get<2>(exit_stats); 

    // Introduce pairs of consecutive mismatches and re-compute cleavage
    // probability and mean first-passage time
    for (unsigned j = 0; j < length - 1; ++j)
    {
        for (unsigned k = 0; k < length; ++k)
             model->setRungLabels(k, match_labels); 
        model->setRungLabels(j, mismatch_labels);
        model->setRungLabels(j + 1, mismatch_labels);  
        exit_stats = model->getExitStats(1, 1);
        stats(j+1, 0) = std::get<0>(exit_stats); 
        stats(j+1, 1) = std::get<1>(exit_stats); 
        stats(j+1, 2) = std::get<2>(exit_stats);
        stats(j+1, 3) = log10(stats(0, 0)) - log10(stats(j+1, 0));
        stats(j+1, 4) = log10(stats(0, 1)) - log10(stats(j+1, 1)); 
        stats(j+1, 5) = log10(stats(0, 2)) - log10(stats(j+1, 2)); 
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[3], "%u", &n);
    MatrixXd params;
    DelaunayTriangulation tri; 
    try
    {
        std::tie(tri, params) = sampleFromConvexPolytopeTriangulation(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // Compute cleavage probabilities, unbinding rates, and cleavage rates
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> unbind_rates(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> cleave_rates(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, length - 1);
    Matrix<PreciseType, Dynamic, Dynamic> norm_unbind(n, length - 1); 
    Matrix<PreciseType, Dynamic, Dynamic> norm_cleave(n, length - 1);  
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<PreciseType, 6, Dynamic> stats = computeStats<PreciseType>(params.row(i)).transpose();
        probs.row(i) = stats.row(0);
        unbind_rates.row(i) = stats.row(1);
        cleave_rates.row(i) = stats.row(2);
        specs.row(i) = stats.block(3, 1, 1, length - 1);
        norm_unbind.row(i) = stats.block(4, 1, 1, length - 1); 
        norm_cleave.row(i) = stats.block(5, 1, 1, length - 1); 
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
            samplefile << params(i, params.cols()-1) << std::endl;
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
            for (unsigned j = 0; j < length - 1; ++j)
            {
                probsfile << probs(i,j) << "\t";
            }
            probsfile << probs(i, length-1) << std::endl; 
        }
    }
    probsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage specificities 
    oss << argv[2] << "-specs.tsv";
    std::ofstream specsfile(oss.str());
    specsfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (specsfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 2; ++j)
            {
                specsfile << specs(i,j) << "\t";
            }
            specsfile << specs(i, length-2) << std::endl; 
        }
    }
    specsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of (unnormalized) unconditional unbinding rates
    oss << argv[2] << "-unbind-rates.tsv";
    std::ofstream unbindfile(oss.str());
    unbindfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (unbindfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                unbindfile << unbind_rates(i,j) << "\t";
            }
            unbindfile << unbind_rates(i, length-1) << std::endl; 
        }
    }
    unbindfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of (unnormalized) conditional cleavage rates 
    oss << argv[2] << "-cleave-rates.tsv";
    std::ofstream cleavefile(oss.str());
    cleavefile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (cleavefile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                cleavefile << cleave_rates(i,j) << "\t";
            }
            cleavefile << cleave_rates(i, length-1) << std::endl; 
        }
    }
    cleavefile.close();
    oss.clear();
    oss.str(std::string());
  
    // Write matrix of normalized unconditional unbinding rates
    oss << argv[2] << "-norm-unbind.tsv";
    std::ofstream unbindfile2(oss.str());
    unbindfile2 << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (unbindfile2.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 2; ++j)
            {
                unbindfile2 << norm_unbind(i,j) << "\t";  
            }
            unbindfile2 << norm_unbind(i, length-2) << std::endl; 
        }
    }
    unbindfile2.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of normalized conditional cleavage rates 
    oss << argv[2] << "-norm-cleave.tsv";
    std::ofstream cleavefile2(oss.str());
    cleavefile2 << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (cleavefile2.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 2; ++j)
            {
                cleavefile2 << norm_cleave(i,j) << "\t"; 
            }
            cleavefile2 << norm_cleave(i, length-2) << std::endl; 
        }
    }
    cleavefile2.close();

    return 0;
}
