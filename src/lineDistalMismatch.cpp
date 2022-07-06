#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <graphs/line.hpp>
#include <polytopes.hpp>

/*
 * Computes cleavage probabilities, unbinding rates, and cleavage rates with
 * respect to distal-mismatch substrates for the line-graph Cas9 model.
 *
 * Call as: 
 *     ./bin/lineDistalMismatch [SAMPLING POLYTOPE .vert FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE]
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     7/6/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
typedef number<mpfr_float_backend<1000> > PreciseType;
const int INTERNAL_PRECISION = 1000; 
const int length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

/**
 * Compute the cleavage probabilities, cleavage rates, dead unbinding rates,
 * and live unbinding rates of randomly parametrized line-graph Cas9 models
 * against distal-mismatch substrates. 
 */
template <typename T>
Matrix<T, Dynamic, 8> computeCleavageStats(const Ref<const VectorXd>& logrates)
{
    // Define arrays of DNA/RNA match and mismatch parameters 
    std::pair<T, T> match_rates = std::make_pair(
        static_cast<T>(std::pow(10.0, logrates(0))),
        static_cast<T>(std::pow(10.0, logrates(1)))
    );
    std::pair<T, T> mismatch_rates = std::make_pair(
        static_cast<T>(std::pow(10.0, logrates(2))),
        static_cast<T>(std::pow(10.0, logrates(3)))
    );

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and
    // live unbinding rate
    T terminal_unbind_rate = static_cast<T>(std::pow(10.0, logrates(4))); 
    T terminal_cleave_rate = static_cast<T>(std::pow(10.0, logrates(5)));
    Matrix<T, Dynamic, 8> stats = Matrix<T, Dynamic, 8>::Zero(length + 1, 8); 
    stats(0, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 2) = model->getLowerExitRate(terminal_unbind_rate);
    stats(0, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);  

    // Introduce distal mismatches and re-compute the four output metrics 
    for (int j = 1; j <= length; ++j)    // j = number of distal mismatches 
    {
        model->setEdgeLabels(length - j, mismatch_rates);
        stats(j, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(j, 2) = model->getLowerExitRate(terminal_unbind_rate);
        stats(j, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 4) = log10(stats(0, 0)) - log10(stats(j, 0));
        stats(j, 5) = log10(stats(0, 1)) - log10(stats(j, 1)); 
        stats(j, 6) = log10(stats(j, 2)) - log10(stats(0, 2));
        stats(j, 7) = log10(stats(j, 3)) - log10(stats(0, 3));
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    int n;
    sscanf(argv[3], "%d", &n);
    MatrixXd params;
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
    Matrix<PreciseType, Dynamic, Dynamic> live_unbind_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> rapid(n, length); 
    Matrix<PreciseType, Dynamic, Dynamic> dead_dissoc(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> live_dissoc(n, length);  
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 8> stats = computeCleavageStats<PreciseType>(params.row(i));
        probs.row(i) = stats.col(0);
        cleave_rates.row(i) = stats.col(1);
        dead_unbind_rates.row(i) = stats.col(2);
        dead_unbind_rates.row(i) = stats.col(3);
        specs.row(i) = stats.col(4).tail(length); 
        rapid.row(i) = stats.col(5).tail(length); 
        dead_dissoc.row(i) = stats.col(6).tail(length); 
        live_dissoc.row(i) = stats.col(7).tail(length); 
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
    oss << argv[2] << "-cleave-rates.tsv";
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
    oss << argv[2] << "-dead-unbind-rates.tsv";
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

    // Write matrix of live unbinding rates
    oss << argv[2] << "-live-unbind-rates.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                outfile << live_unbind_rates(i, j) << "\t";
            }
            outfile << live_unbind_rates(i, length) << std::endl; 
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

    // Write matrix of dead specific dissociativities 
    oss << argv[2] << "-dead-dissoc.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                outfile << dead_dissoc(i, j) << "\t"; 
            }
            outfile << dead_dissoc(i, length-1) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of live specific dissociativities 
    oss << argv[2] << "-live-dissoc.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                outfile << live_dissoc(i, j) << "\t"; 
            }
            outfile << live_dissoc(i, length-1) << std::endl; 
        }
    }
    outfile.close();

    return 0;
}
