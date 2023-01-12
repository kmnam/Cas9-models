/**
 * Computes cleavage probabilities, unbinding rates, and cleavage rates with
 * respect to various user-specified single- and multi-mismatch substrates for
 * the line-graph Cas9 model on the vertices of a given convex polytope in
 * parameter space.
 *
 * Call as: 
 *     ./bin/lineMultiMismatchVertices [MISMATCH PATTERNS] [POLYTOPE .vert FILE] [OUTPUT FILE PREFIX] [CAS9 CONCENTRATION]
 *
 * **Author:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/12/2023
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
const int length = 20;
const PreciseType ten("10");

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

/**
 * Compute the cleavage probabilities, cleavage rates, dead unbinding rates,
 * and live unbinding rates of randomly parametrized line-graph Cas9 models
 * against a given matrix of single- and multi-mismatch substrates (binary 
 * complementarity patterns). 
 *
 * The parameter values here are dimensioned rates, in units of inverse seconds,
 * instead of normalized ratios of rates divided by the terminal unbinding rate.  
 */
Matrix<PreciseType, Dynamic, 8> computeCleavageStats(const Ref<const Matrix<bool, Dynamic, Dynamic> >& patterns,
                                                     const Ref<const VectorXd>& logrates,
                                                     const PreciseType bind_conc)
{
    using boost::multiprecision::pow;

    // Define arrays of DNA/RNA match and mismatch parameters
    PreciseType match_fwd = pow(ten, static_cast<PreciseType>(logrates(0)));
    PreciseType match_rev = pow(ten, static_cast<PreciseType>(logrates(1)));
    PreciseType mismatch_fwd = pow(ten, static_cast<PreciseType>(logrates(2)));
    PreciseType mismatch_rev = pow(ten, static_cast<PreciseType>(logrates(3)));
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(match_fwd, match_rev);
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(mismatch_fwd, mismatch_rev);
    
    // Assign each edge with the appropriate DNA/RNA match parameter 
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    LineGraph<PreciseType, PreciseType>* model_fwd = new LineGraph<PreciseType, PreciseType>(length - 1);
    LineGraph<PreciseType, PreciseType>* model_rev = new LineGraph<PreciseType, PreciseType>(length - 1);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates);
    for (int j = 0; j < length - 1; ++j)
    {
        model_fwd->setEdgeLabels(j, match_rates);
        model_rev->setEdgeLabels(j, match_rates);
    }
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate,
    // live unbinding rate, R-loop completion rate, and R-loop dissolution
    // rate on the perfect-match substrate 
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));
    PreciseType bind_rate = pow(ten, static_cast<PreciseType>(logrates(6))) * bind_conc; 
    Matrix<PreciseType, Dynamic, 8> stats = Matrix<PreciseType, Dynamic, 8>::Zero(patterns.rows() + 1, 8); 
    stats(0, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 2) = model->getLowerExitRate(terminal_unbind_rate);
    stats(0, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
    stats(0, 4) = model_fwd->getUpperExitRateFromZero(match_fwd);     // R-loop completion rate 
    stats(0, 5) = model_rev->getLowerExitRateFromN(match_rev);        // R-loop dissolution rate
    
    // Compute the composite cleavage rate on the perfect-match substrate 
    PreciseType term = 1 / stats(0, 0);
    stats(0, 6) = 1 / ((term / bind_rate) + (1 / stats(0, 1)) + ((term - 1) / stats(0, 3))); 

    // Introduce mismatches as defined and re-compute the four output metrics 
    for (int j = 0; j < patterns.rows(); ++j)
    {
        for (int k = 0; k < length; ++k)
        {
            if (patterns(j, k))
                model->setEdgeLabels(k, match_rates); 
            else 
                model->setEdgeLabels(k, mismatch_rates);
        }
        stats(j+1, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(j+1, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(j+1, 2) = model->getLowerExitRate(terminal_unbind_rate);
        stats(j+1, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        term = 1 / stats(j+1, 0);
        stats(j+1, 6) = 1 / ((term / bind_rate) + (1 / stats(j+1, 1)) + ((term - 1) / stats(j+1, 3))); 
        stats(j+1, 7) = log10(stats(j+1, 2)) - log10(stats(0, 2));
        for (int k = 0; k < length - 1; ++k)
        {
            if (patterns(j, k))
                model_fwd->setEdgeLabels(k, match_rates);
            else
                model_fwd->setEdgeLabels(k, mismatch_rates);
            if (patterns(j, k+1))
                model_rev->setEdgeLabels(k, match_rates);
            else 
                model_rev->setEdgeLabels(k, mismatch_rates);
        }
        stats(j+1, 4) = model_fwd->getUpperExitRateFromZero(patterns(j, length - 1) ? match_fwd : mismatch_fwd);
        stats(j+1, 5) = model_rev->getLowerExitRateFromN(patterns(j, 0) ? match_rev : mismatch_rev);
    }

    delete model;
    delete model_fwd;
    delete model_rev;
    return stats;
}

int main(int argc, char** argv)
{
    // Process input arguments
    if (argc != 5)
        throw std::runtime_error("Invalid number of input arguments"); 

    // Parse input polytope and Cas9 concentration
    MatrixXd params;
    try
    {
        params = Polytopes::parseVertexCoords(argv[2]).cast<double>();
    }
    catch (const std::exception& e)
    {
        throw;
    }
    int n = params.rows();
    PreciseType bind_conc(argv[4]);  

    // Parse input binary complementarity patterns 
    int m = 0; 
    Matrix<bool, Dynamic, Dynamic> patterns(0, length);
    std::ifstream infile(argv[1]); 
    std::string line;
    if (infile.is_open())
    {
        while (std::getline(infile, line))
        {
            m++;
            patterns.conservativeResize(m, length); 
            for (int i = 0; i < length; ++i)
            {
                if (line[i] == '1')
                    patterns(m - 1, i) = true;
                else 
                    patterns(m - 1, i) = false;
            }
        }
    }

    // Compute cleavage probabilities, unbinding rates, cleavage rates,
    // R-loop completion rates, and R-loop dissolution rates
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> cleave_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> dead_unbind_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> completion_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> dissolution_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> comp_cleave_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> dead_dissoc(n, m);
    for (int i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 8> stats = computeCleavageStats(patterns, params.row(i), bind_conc);
        probs.row(i) = stats.col(0);
        cleave_rates.row(i) = stats.col(1);
        dead_unbind_rates.row(i) = stats.col(2);
        completion_rates.row(i) = stats.col(4);
        dissolution_rates.row(i) = stats.col(5);
        comp_cleave_rates.row(i) = stats.col(6);
        dead_dissoc.row(i) = stats.col(7).tail(m); 
    }

    // Write sampled log-rates to file
    std::ostringstream oss;
    oss << argv[3] << "-logrates.tsv";
    std::ofstream outfile(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (int i = 0; i < params.rows(); ++i)
        {
            for (int j = 0; j < params.cols() - 1; ++j)
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
    oss << argv[3] << "-probs.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                outfile << probs(i, j) << "\t";
            }
            outfile << probs(i, m) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage rates 
    oss << argv[3] << "-cleave.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                outfile << cleave_rates(i, j) << "\t";
            }
            outfile << cleave_rates(i, m) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of dead unbinding rates
    oss << argv[3] << "-unbind.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                outfile << dead_unbind_rates(i, j) << "\t";
            }
            outfile << dead_unbind_rates(i, m) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of R-loop completion rates
    oss << argv[3] << "-Rcompletion.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                outfile << completion_rates(i, j) << "\t";
            }
            outfile << completion_rates(i, m) << std::endl;
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of R-loop dissolution rates
    oss << argv[3] << "-Rdissolution.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                outfile << dissolution_rates(i, j) << "\t";
            }
            outfile << dissolution_rates(i, m) << std::endl;
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of composite cleavage rates 
    oss << argv[3] << "-compcleave.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                outfile << comp_cleave_rates(i, j) << "\t";
            }
            outfile << comp_cleave_rates(i, m) << std::endl;
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of dead specific dissociativities 
    oss << argv[3] << "-deaddissoc.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
        outfile << std::string(length, '1') << '\t';
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                outfile << (patterns(i, j) ? '1' : '0'); 
            }
            if (i < m - 1)
            {
                outfile << '\t';
            }
        }
        outfile << std::endl; 
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m - 1; ++j)
            {
                outfile << dead_dissoc(i, j) << "\t"; 
            }
            outfile << dead_dissoc(i, m - 1) << std::endl; 
        }
    }
    outfile.close();

    return 0;
}
