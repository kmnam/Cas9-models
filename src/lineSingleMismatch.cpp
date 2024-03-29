/**
 * Computes cleavage probabilities, unbinding rates, and cleavage rates with
 * respect to single-mismatch substrates for the line-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/lineSingleMismatch [SAMPLING POLYTOPE .vert FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     3/4/2023
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <stdexcept>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <graphs/line.hpp>
#include <polytopes.hpp>

using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
constexpr int INTERNAL_PRECISION = 100; 
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;
const PreciseType ten("10");
const int length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

/**
 * Compute the cleavage probabilities, cleavage rates, dead unbinding rates,
 * and live unbinding rates of randomly parametrized line-graph Cas9 models
 * against single-mismatch substrates. 
 */
Matrix<PreciseType, Dynamic, 5> computeCleavageStats(const Ref<const VectorXd>& logrates)
{
    // Define arrays of DNA/RNA match and mismatch parameters 
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1)))
    );
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(2))),
        pow(ten, static_cast<PreciseType>(logrates(3)))
    );

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and
    // live unbinding rate
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));
    Matrix<PreciseType, Dynamic, 5> stats = Matrix<PreciseType, Dynamic, 5>::Zero(length + 1, 5); 
    stats(0, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 2) = model->getLowerExitRate(terminal_unbind_rate);

    // Introduce single mismatches and re-compute the four output metrics 
    for (int j = 0; j < length; ++j)
    {
        for (int k = 0; k < length; ++k)
            model->setEdgeLabels(k, match_rates); 
        model->setEdgeLabels(j, mismatch_rates);
        stats(j+1, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(j+1, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(j+1, 2) = model->getLowerExitRate(terminal_unbind_rate);
        stats(j+1, 3) = log10(stats(0, 0)) - log10(stats(j+1, 0));
        stats(j+1, 4) = log10(stats(0, 1)) - log10(stats(j+1, 1)); 
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Process input arguments
    if (argc != 4)
        throw std::runtime_error("Invalid number of input arguments"); 

    // Sample model parameter combinations
    int n = std::stoi(argv[3]);
    MatrixXd params(n, 6);
    try
    {
        params = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(argv[1], n, 0, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // Compute cleavage probabilities, unbinding rates, and cleavage rates
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> cleave_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> dead_unbind_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> rapid(n, length); 
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 5> stats = computeCleavageStats(params.row(i));
        probs.row(i) = stats.col(0);
        cleave_rates.row(i) = stats.col(1);
        dead_unbind_rates.row(i) = stats.col(2);
        specs.row(i) = stats.col(3).tail(length); 
        rapid.row(i) = stats.col(4).tail(length); 
    }

    // Write sampled log-rates to file
    std::ostringstream oss;
    oss << argv[2] << "-logrates.tsv";
    std::ofstream outfile(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < params.rows(); i++)
        {
            for (unsigned j = 0; j < params.cols() - 1; j++)
            {
                outfile << params(i, j) << "\t";
            }
            outfile << params(i, params.cols()-1) << std::endl;
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage probabilities
    oss << argv[2] << "-probs.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                outfile << probs(i, j) << "\t";
            }
            outfile << probs(i, length) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage rates 
    oss << argv[2] << "-cleave.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                outfile << cleave_rates(i, j) << "\t";
            }
            outfile << cleave_rates(i, length) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of dead unbinding rates
    oss << argv[2] << "-unbind.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                outfile << dead_unbind_rates(i, j) << "\t";
            }
            outfile << dead_unbind_rates(i, length) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage specificities 
    oss << argv[2] << "-specs.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                outfile << specs(i, j) << "\t";
            }
            outfile << specs(i, length - 1) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());
  
    // Write matrix of specific rapidities
    oss << argv[2] << "-rapid.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                outfile << rapid(i, j) << "\t";  
            }
            outfile << rapid(i, length-1) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    return 0;
}
