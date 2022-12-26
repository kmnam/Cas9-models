/**
 * Computes cleavage probabilities, unbinding rates, and cleavage rates with
 * respect to various user-specified single- and multi-mismatch substrates for
 * the line-graph Cas9 model on the vertices of a given convex polytope in
 * parameter space.
 *
 * Call as: 
 *     ./bin/lineMultiMismatchVertices [MISMATCH PATTERNS] [POLYTOPE .vert FILE] [OUTPUT FILE PREFIX]
 *
 * **Author:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     12/26/2022
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
typedef number<mpfr_float_backend<100> > PreciseType;
const int INTERNAL_PRECISION = 100; 
const int length = 20;

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
template <typename T>
Matrix<T, Dynamic, 10> computeCleavageStats(const Ref<const Matrix<bool, Dynamic, Dynamic> >& patterns,
                                            const Ref<const VectorXd>& logrates)
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

    // Assign each edge with the appropriate DNA/RNA match parameter 
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and
    // live unbinding rate on the perfect-match substrate 
    T terminal_unbind_rate = static_cast<T>(std::pow(10.0, logrates(4)));
    T terminal_cleave_rate = static_cast<T>(std::pow(10.0, logrates(5)));
    T bind_rate = static_cast<T>(std::pow(10.0, logrates(6))); 
    Matrix<T, Dynamic, 10> stats = Matrix<T, Dynamic, 10>::Zero(patterns.rows() + 1, 10); 
    stats(0, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 2) = model->getLowerExitRate(terminal_unbind_rate);
    stats(0, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
    
    // Compute the composite cleavage time on the perfect-match substrate 
    T term = 1 / stats(0, 0);
    stats(0, 4) = (term / bind_rate) + (1 / stats(0, 1)) + ((term - 1) / stats(0, 3)); 

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
        stats(j+1, 4) = (term / bind_rate) + (1 / stats(j+1, 1)) + ((term - 1) / stats(j+1, 3)); 
        stats(j+1, 5) = log10(stats(0, 0)) - log10(stats(j+1, 0));
        stats(j+1, 6) = log10(stats(0, 1)) - log10(stats(j+1, 1)); 
        stats(j+1, 7) = log10(stats(j+1, 2)) - log10(stats(0, 2));
        stats(j+1, 8) = log10(stats(j+1, 3)) - log10(stats(0, 3));
        stats(j+1, 9) = log10(stats(j+1, 4)) - log10(stats(0, 4));
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Process input arguments
    if (argc != 4)
        throw std::runtime_error("Invalid number of input arguments"); 

    // Parse input polytope 
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

    // Compute cleavage probabilities, unbinding rates, and cleavage rates
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> cleave_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> dead_unbind_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> live_unbind_rates(n, m + 1);
    Matrix<PreciseType, Dynamic, Dynamic> comp_cleave_times(n, m + 1); 
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, m);
    Matrix<PreciseType, Dynamic, Dynamic> rapid(n, m); 
    Matrix<PreciseType, Dynamic, Dynamic> dead_dissoc(n, m);
    Matrix<PreciseType, Dynamic, Dynamic> live_dissoc(n, m);  
    Matrix<PreciseType, Dynamic, Dynamic> comp_cleave_time_ratios(n, m); 
    for (int i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 10> stats = computeCleavageStats<PreciseType>(patterns, params.row(i));
        probs.row(i) = stats.col(0);
        cleave_rates.row(i) = stats.col(1);
        dead_unbind_rates.row(i) = stats.col(2);
        live_unbind_rates.row(i) = stats.col(3);
        comp_cleave_times.row(i) = stats.col(4);
        specs.row(i) = stats.col(5).tail(m); 
        rapid.row(i) = stats.col(6).tail(m); 
        dead_dissoc.row(i) = stats.col(7).tail(m); 
        live_dissoc.row(i) = stats.col(8).tail(m);
        comp_cleave_time_ratios.row(i) = stats.col(9).tail(m); 
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
    oss << argv[3] << "-cleaverates.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
    oss << argv[3] << "-deadunbindrates.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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

    // Write matrix of live unbinding rates
    oss << argv[3] << "-liveunbindrates.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
                outfile << live_unbind_rates(i, j) << "\t";
            }
            outfile << live_unbind_rates(i, m) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of composite cleavage times 
    oss << argv[3] << "-compcleavetimes.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
                outfile << comp_cleave_times(i, j) << "\t";
            }
            outfile << comp_cleave_times(i, m) << std::endl;
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage specificities 
    oss << argv[3] << "-specs.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
                outfile << specs(i, j) << "\t";
            }
            outfile << specs(i, m - 1) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());
  
    // Write matrix of specific rapidities
    oss << argv[3] << "-rapid.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
                outfile << rapid(i, j) << "\t";  
            }
            outfile << rapid(i, m - 1) << std::endl; 
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
    oss.clear();
    oss.str(std::string());

    // Write matrix of live specific dissociativities 
    oss << argv[3] << "-livedissoc.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
                outfile << live_dissoc(i, j) << "\t"; 
            }
            outfile << live_dissoc(i, m - 1) << std::endl; 
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string()); 

    // Write matrix of composite cleavage time ratios 
    oss << argv[3] << "-compcleavetimeratios.tsv";
    outfile.open(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        // Write complementarity patterns to file in header line  
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
                outfile << comp_cleave_time_ratios(i, j) << "\t";
            }
            outfile << comp_cleave_time_ratios(i, m - 1) << std::endl;
        }
    }
    outfile.close();

    return 0;
}
