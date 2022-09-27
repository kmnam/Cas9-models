/**
 * Using line-search SQP, identify the set of line graph parameter vectors 
 * that yields each given set of unbinding and cleavage rates in the given
 * data files.
 *
 * Abbreviations in the below comments:
 * - LG:   line graph
 * - LGPs: line graph parameters
 * - SQP:  sequential quadratic programming
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     9/27/2022
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <boostMultiprecisionEigen.hpp>
#include <linearConstraints.hpp>
#include <polytopes.hpp>
#include <SQP.hpp>
#include <graphs/line.hpp>
#include "../include/utils.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
using boost::multiprecision::pow;
using boost::multiprecision::sqrt;
constexpr int INTERNAL_PRECISION = 100; 
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;

const unsigned length = 20;
const PreciseType ten("10");

/**
 * Return a collection of index-sets for the given number of folds into which
 * to divide a dataset of the given size. 
 *
 * @param n      Number of data points. 
 * @param nfolds Number of folds into which to divide the dataset. 
 * @returns      Collection of index-sets, each corresponding to the training
 *               and test subset corresponding to each fold. 
 */
std::vector<std::pair<std::vector<int>, std::vector<int> > > getFolds(const int n,
                                                                      const int nfolds)
{
    int foldsize = static_cast<int>(n / nfolds);
    std::vector<std::pair<std::vector<int>, std::vector<int> > > fold_pairs; 

    // Define the first (nfolds - 1) folds ... 
    int start = 0;  
    for (int i = 0; i < nfolds - 1; ++i)
    {
        std::vector<int> train_fold, test_fold;
        for (int j = 0; j < start; ++j)
            train_fold.push_back(j);  
        for (int j = start; j < start + foldsize; ++j)
            test_fold.push_back(j);
        for (int j = start + foldsize; j < n; ++j)
            train_fold.push_back(j);  
        fold_pairs.emplace_back(std::make_pair(train_fold, test_fold)); 
        start += foldsize; 
    }

    // ... then define the last fold
    std::vector<int> last_train_fold, last_test_fold;
    for (int i = 0; i < start; ++i)
        last_train_fold.push_back(i);  
    for (int i = start; i < n; ++i)
        last_test_fold.push_back(i);  
    fold_pairs.emplace_back(std::make_pair(last_train_fold, last_test_fold)); 

    return fold_pairs;  
}

/**
 * Return a randomly generated permutation of the range [0, 1, ..., n - 1],
 * using the Fisher-Yates shuffle.
 *
 * @param n   Size of input range.
 * @param rng Random number generator instance.
 * @returns Permutation of the range [0, 1, ..., n - 1], as a permutation matrix.
 */
PermutationMatrix<Dynamic, Dynamic> getPermutation(const int n, boost::random::mt19937& rng)
{
    VectorXi p(n); 
    for (int i = 0; i < n; ++i)
        p(i) = i;
    
    for (int i = 0; i < n - 1; ++i)
    {
        // Generate a random integer between i and n - 1 (inclusive)
        boost::random::uniform_int_distribution<> dist(i, n - 1);
        int j = dist(rng); 
        
        // Swap p[i] and p[j]
        int tmp = p(j); 
        p(j) = p(i); 
        p(i) = tmp;
    }

    // Convert the vector into a permutation matrix 
    PermutationMatrix<Dynamic, Dynamic> perm(n);
    perm.indices() = p; 
    
    return perm; 
}

/**
 * Given a pair of DNA bases as integers (0 = A, 1 = C, 2 = G, 3 = T), return
 * 0 if the pair are the same; 1 if the pair form a transition; and 2 if the 
 * pair form a transversion. 
 *
 * @param c1 First nucleobase.
 * @param c2 Second nucleobase.
 * @returns 0 if `c1` and `c2` are the same, 1 if they form a transition, 
 *          2 if they form a transversion.
 * @throws std::invalid_argument If either `c1` or `c2` are not 0, 1, 2, or 3. 
 */
int getMutationType(const int c1, const int c2)
{
    if (c1 < 0 || c1 > 3 || c2 < 0 || c2 > 3)
        throw std::invalid_argument("Invalid input nucleobase value (must be 0, 1, 2, or 3)");
    else if (c1 == c2)
        return 0; 
    else if (std::abs(c1 - c2) == 2)   // If c1 == A and c2 == G or vice versa,
        return 1;                      // or if c1 == C and c2 == T or vice versa
    else 
        return 2;
}

/**
 * Compute cleavage statistics on the perfect-match DNA sequence, as well as 
 * all mismatched DNA sequences specified in the given matrix, for the given
 * LG with base-specific LGPs.
 *
 * Here, `logrates` is assumed to contain 34 entries:
 * - the first 16 entries are the forward rates at DNA-RNA (mis)matches of the
 *   form A/A, A/C, A/G, ...
 * - the second 16 entries are the reverse rates at DNA-RNA (mis)matches of the
 *   form A/A, A/C, A/G, ... 
 * - the last two entries are the terminal cleavage rate and binding rate, 
 *   respectively.  
 */
Matrix<PreciseType, Dynamic, 4> computeCleavageStatsBaseSpecific(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                                                                 const Ref<const MatrixXi>& seqs,
                                                                 const Ref<const VectorXi>& seq_match, 
                                                                 const PreciseType bind_conc)
{
    // Store rates in linear scale 
    Matrix<PreciseType, Dynamic, 1> rates(logrates.size()); 
    for (int i = 0; i < logrates.size(); ++i)
        rates(i) = pow(ten, logrates(i)); 

    // Exit rates from terminal nodes 
    PreciseType terminal_unbind_rate = 1;
    PreciseType terminal_cleave_rate = rates(32);   // Penultimate entry (out of 34) in the vector

    // Binding rate entering state 0
    PreciseType bind_rate = rates(33);              // Ultimate entry (out of 34) in the vector

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
    {
        // For each character, find the corresponding pair of forward/reverse
        // rates for the i-th DNA-RNA match
        std::pair<PreciseType, PreciseType> match_rates; 
        int c = seq_match(i); 
        match_rates.first = rates(c * 4 + c); 
        match_rates.second = rates(16 + c * 4 + c); 
        model->setEdgeLabels(i, match_rates);
    } 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and 
    // live unbinding rate against perfect-match substrate
    Matrix<PreciseType, 5, 1> stats_perfect;
    stats_perfect(0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(2) = model->getLowerExitRate(terminal_unbind_rate); 
    stats_perfect(3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);

    // Compute the composite cleavage time 
    stats_perfect(4) = (
        1 / (bind_conc * bind_rate)
        + (1 / stats_perfect(3) + 1 / (bind_conc * bind_rate)) * (1 - stats_perfect(0)) / stats_perfect(0)
        + 1 / stats_perfect(1)
    );

    // Re-compute cleavage probability, cleavage rate, dead unbinding rate, 
    // live unbinding rate, and composite cleavage time against each given
    // mismatched substrate
    Matrix<PreciseType, Dynamic, 4> stats(seqs.rows(), 4);  
    for (int i = 0; i < seqs.rows(); ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            // Determine whether there is a mismatch between the i-th sequence at
            // the j-th position and the perfect-match sequence
            std::pair<PreciseType, PreciseType> rates_j; 
            int c1 = seq_match(j);
            int c2 = seqs(i, j);
            rates_j.first = rates(c1 * 4 + c2); 
            rates_j.second = rates(16 + c1 * 4 + c2); 
            model->setEdgeLabels(j, rates_j); 
        }
        stats(i, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(i, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(i, 2) = model->getLowerExitRate(terminal_unbind_rate); 
        PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(i, 3) = (
            1 / (bind_conc * bind_rate)
            + (1 / unbind_rate + 1 / (bind_conc * bind_rate)) * (1 - stats(i, 0)) / stats(i, 0) + 1 / stats(i, 1)
        );

        // Inverse specificity = cleavage probability on mismatched / cleavage probability on perfect
        stats(i, 0) = log10(stats(i, 0)) - log10(stats_perfect(0)); 

        // Inverse rapidity = cleavage rate on mismatched / cleavage rate on perfect
        stats(i, 1) = log10(stats(i, 1)) - log10(stats_perfect(1));  

        // Unbinding rate on mismatched > unbinding rate on perfect, so 
        // return perfect rate / mismatched rate (inverse dissociativity) 
        stats(i, 2) = log10(stats_perfect(2)) - log10(stats(i, 2));
        
        // Cleavage *time* on mismatched > cleavage *time* on perfect (usually), 
        // so return perfect time / mismatched time (inverse composite cleavage
        // time ratio)
        stats(i, 3) = log10(stats_perfect(4)) - log10(stats(i, 3));
    }

    delete model;
    return stats;
}

/**
 * Compute cleavage statistics on the perfect-match DNA sequence, as well as 
 * all mismatched DNA sequences specified in the given matrix, for the given
 * LG with mutation-type-specific LGPs.
 *
 * Here, `logrates` is assumed to contain 8 entries:
 * - the first 3 entries are the forward rates at DNA-RNA matches, transitions,
 *   and transversions, respectively; 
 * - the second 3 entries are the reverse rates at DNA-RNA matches, transitions,
 *   and transversions, respectively; and
 * - the last two entries are the terminal cleavage rate and binding rate, 
 *   respectively.  
 */
Matrix<PreciseType, Dynamic, 4> computeCleavageStatsMutationSpecific(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                                                                     const Ref<const MatrixXi>& seqs,
                                                                     const Ref<const VectorXi>& seq_match, 
                                                                     const PreciseType bind_conc)
{
    // Store rates in linear scale 
    Matrix<PreciseType, Dynamic, 1> rates(logrates.size()); 
    for (int i = 0; i < logrates.size(); ++i)
        rates(i) = pow(ten, logrates(i)); 

    // Exit rates from terminal nodes 
    PreciseType terminal_unbind_rate = 1;
    PreciseType terminal_cleave_rate = rates(6);   // Penultimate entry (out of 8) in the vector

    // Binding rate entering state 0
    PreciseType bind_rate = rates(7);              // Ulimate entry (out of 8) in the vector

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(rates(0), rates(3)); 
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates);
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and 
    // live unbinding rate against perfect-match substrate
    Matrix<PreciseType, 5, 1> stats_perfect;
    stats_perfect(0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(2) = model->getLowerExitRate(terminal_unbind_rate); 
    stats_perfect(3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);

    // Compute the composite cleavage time 
    stats_perfect(4) = (
        1 / (bind_conc * bind_rate)
        + (1 / stats_perfect(3) + 1 / (bind_conc * bind_rate)) * (1 - stats_perfect(0)) / stats_perfect(0)
        + 1 / stats_perfect(1)
    );

    // Re-compute cleavage probability, cleavage rate, dead unbinding rate, 
    // live unbinding rate, and composite cleavage time against each given
    // mismatched substrate
    Matrix<PreciseType, Dynamic, 4> stats(seqs.rows(), 4);  
    for (int i = 0; i < seqs.rows(); ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            // Determine whether there is a match, transition, or transversion
            // between the i-th sequence at the j-th position and the perfect-
            // match sequence
            std::pair<PreciseType, PreciseType> rates_j;
            int c1 = seq_match(j);
            int c2 = seqs(i, j);
            int type = getMutationType(c1, c2); 
            rates_j.first = rates(type); 
            rates_j.second = rates(3 + type); 
            model->setEdgeLabels(j, rates_j);
        }
        stats(i, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(i, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(i, 2) = model->getLowerExitRate(terminal_unbind_rate); 
        PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(i, 3) = (
            1 / (bind_conc * bind_rate)
            + (1 / unbind_rate + 1 / (bind_conc * bind_rate)) * (1 - stats(i, 0)) / stats(i, 0) + 1 / stats(i, 1)
        );

        // Inverse specificity = cleavage probability on mismatched / cleavage probability on perfect
        stats(i, 0) = log10(stats(i, 0)) - log10(stats_perfect(0)); 

        // Inverse rapidity = cleavage rate on mismatched / cleavage rate on perfect
        stats(i, 1) = log10(stats(i, 1)) - log10(stats_perfect(1));  

        // Unbinding rate on mismatched > unbinding rate on perfect, so 
        // return perfect rate / mismatched rate (inverse dissociativity) 
        stats(i, 2) = log10(stats_perfect(2)) - log10(stats(i, 2));
        
        // Cleavage *time* on mismatched > cleavage *time* on perfect (usually), 
        // so return perfect time / mismatched time (inverse composite cleavage
        // time ratio)
        stats(i, 3) = log10(stats_perfect(4)) - log10(stats(i, 3));
    }

    delete model;
    return stats;
}

/**
 * Compute cleavage statistics on the perfect-match sequence, as well as all
 * mismatched sequences specified in the given matrix of complementarity
 * patterns, for the given LG.
 */
Matrix<PreciseType, Dynamic, 4> computeCleavageStats(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                                                     const Ref<const MatrixXi>& seqs, 
                                                     const PreciseType bind_conc)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, logrates(0)), pow(ten, logrates(1))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, logrates(2)), pow(ten, logrates(3))
    );

    // Exit rates from terminal nodes 
    PreciseType terminal_unbind_rate = 1;
    PreciseType terminal_cleave_rate = pow(ten, logrates(4)); 

    // Binding rate entering state 0
    PreciseType bind_rate = pow(ten, logrates(5));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and 
    // live unbinding rate against perfect-match substrate
    Matrix<PreciseType, 5, 1> stats_perfect;
    stats_perfect(0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(2) = model->getLowerExitRate(terminal_unbind_rate); 
    stats_perfect(3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);

    // Compute the composite cleavage time 
    stats_perfect(4) = (
        1 / (bind_conc * bind_rate)
        + (1 / stats_perfect(3) + 1 / (bind_conc * bind_rate)) * (1 - stats_perfect(0)) / stats_perfect(0)
        + 1 / stats_perfect(1)
    );

    // Re-compute cleavage probability, cleavage rate, dead unbinding rate, 
    // live unbinding rate, and composite cleavage time against each given
    // mismatched substrate
    Matrix<PreciseType, Dynamic, 4> stats(seqs.rows(), 4);  
    for (int j = 0; j < seqs.rows(); ++j)
    {
        for (int k = 0; k < length; ++k)
        {
            if (seqs(j, k))
                model->setEdgeLabels(k, match_rates);
            else
                model->setEdgeLabels(k, mismatch_rates);
        }
        stats(j, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(j, 2) = model->getLowerExitRate(terminal_unbind_rate); 
        PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 3) = (
            1 / (bind_conc * bind_rate)
            + (1 / unbind_rate + 1 / (bind_conc * bind_rate)) * (1 - stats(j, 0)) / stats(j, 0) + 1 / stats(j, 1)
        );

        // Inverse specificity = cleavage probability on mismatched / cleavage probability on perfect
        stats(j, 0) = log10(stats(j, 0)) - log10(stats_perfect(0)); 

        // Inverse rapidity = cleavage rate on mismatched / cleavage rate on perfect
        stats(j, 1) = log10(stats(j, 1)) - log10(stats_perfect(1));  

        // Unbinding rate on mismatched > unbinding rate on perfect, so 
        // return perfect rate / mismatched rate (inverse dissociativity) 
        stats(j, 2) = log10(stats_perfect(2)) - log10(stats(j, 2));
        
        // Cleavage *time* on mismatched > cleavage *time* on perfect (usually), 
        // so return perfect time / mismatched time (inverse composite cleavage
        // time ratio)
        stats(j, 3) = log10(stats_perfect(4)) - log10(stats(j, 3));
    }

    delete model;
    return stats;
}

/**
 * Compute the *unregularized* error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the given LGPs and a set of
 * experimentally determined cleavage rates/times and (dead) unbinding
 * rates/times, with respect to a set of DNA sequences. 
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.  
 */
PreciseType errorAgainstData(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                             const Ref<const MatrixXi>& cleave_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
                             const Ref<const MatrixXi>& unbind_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
                             const Ref<const VectorXi>& cleave_seq_match, 
                             const Ref<const VectorXi>& unbind_seq_match,
                             const int mode, const PreciseType bind_conc,
                             const bool logscale = false)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* cleavage metrics and corresponding error against data
    if (mode == 1)
    {
        stats1 = computeCleavageStatsMutationSpecific(
            logrates, cleave_seqs, cleave_seq_match, bind_conc
        );
        stats2 = computeCleavageStatsMutationSpecific(
            logrates, unbind_seqs, unbind_seq_match, bind_conc
        );
    }
    else if (mode == 2)
    {
        stats1 = computeCleavageStatsBaseSpecific(
            logrates, cleave_seqs, cleave_seq_match, bind_conc
        );
        stats2 = computeCleavageStatsBaseSpecific(
            logrates, unbind_seqs, unbind_seq_match, bind_conc
        );
    }
    else    // mode should be either 1 or 2
    {
        throw std::invalid_argument(
            "Invalid parametrization mode specified (should be 1 or 2)"
        ); 
    }
    PreciseType error = 0;
    if (logscale)               // Compute error function in log-scale
    {
        error += (stats1.col(3) - cleave_data.array().log10().matrix()).squaredNorm();
        error += (stats2.col(2) - unbind_data.array().log10().matrix()).squaredNorm();
    }
    else                        // Compute error function in linear scale  
    {
        Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
        Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
        for (int j = 0; j < 4; ++j)
        {
            for (int i = 0; i < stats1.rows(); ++i)
                stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
            for (int i = 0; i < stats2.rows(); ++i)
                stats2_transformed(i, j) = pow(ten, stats2(i, j));
        }
        error += (stats1_transformed.col(3) - cleave_data).squaredNorm(); 
        error += (stats2_transformed.col(2) - unbind_data).squaredNorm(); 
    }

    return error;
}

/**
 * Compute the *unregularized* error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the given LGPs and a set of
 * experimentally determined cleavage rates/times and (dead) unbinding
 * rates/times, with respect to a set of binary complementarity patterns.
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.  
 */
PreciseType errorAgainstData(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                             const Ref<const MatrixXi>& cleave_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
                             const Ref<const MatrixXi>& unbind_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
                             const PreciseType bind_conc, const bool logscale = false)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc);
    PreciseType error = 0;
    if (logscale)               // Compute error function in log-scale
    {
        error += (stats1.col(3) - cleave_data.array().log10().matrix()).squaredNorm();
        error += (stats2.col(2) - unbind_data.array().log10().matrix()).squaredNorm();
    }
    else                        // Compute error function in linear scale  
    {
        Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
        Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
        for (int j = 0; j < 4; ++j)
        {
            for (int i = 0; i < stats1.rows(); ++i)
                stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
            for (int i = 0; i < stats2.rows(); ++i)
                stats2_transformed(i, j) = pow(ten, stats2(i, j));
        }
        error += (stats1_transformed.col(3) - cleave_data).squaredNorm(); 
        error += (stats2_transformed.col(2) - unbind_data).squaredNorm(); 
    }

    return error;
}

/**
 * @param cleave_data
 * @param unbind_data
 * @param cleave_seqs
 * @param unbind_seqs
 * @param mode
 * @param cleave_seq_match
 * @param unbind_seq_match
 * @param ninit
 * @param bind_conc
 * @param logscale
 * @param rng
 * @param tau
 * @param delta
 * @param beta
 * @param max_iter
 * @param tol
 * @param method
 * @param regularize
 * @param regularize_weight
 * @param use_only_armijo
 * @param use_strong_wolfe
 * @param hessian_modify_max_iter
 * @param c1
 * @param c2
 * @param x_tol
 * @param verbose
 */
std::tuple<Matrix<PreciseType, Dynamic, Dynamic>, 
           Matrix<PreciseType, Dynamic, Dynamic>,
           Matrix<PreciseType, Dynamic, 1> >
    fitLineParamsAgainstMeasuredRates(const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data, 
                                      const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data, 
                                      const Ref<const MatrixXi>& cleave_seqs, 
                                      const Ref<const MatrixXi>& unbind_seqs,
                                      const int mode,
                                      const Ref<const VectorXi>& cleave_seq_match,
                                      const Ref<const VectorXi>& unbind_seq_match,
                                      const int ninit, const PreciseType bind_conc,
                                      const bool logscale, boost::random::mt19937& rng,
                                      const PreciseType tau, const PreciseType delta,
                                      const PreciseType beta, const int max_iter,
                                      const PreciseType tol, const QuasiNewtonMethod method, 
                                      const RegularizationMethod regularize,
                                      const PreciseType regularize_weight, 
                                      const bool use_only_armijo,
                                      const bool use_strong_wolfe, 
                                      const int hessian_modify_max_iter, 
                                      const PreciseType c1, const PreciseType c2,
                                      const PreciseType x_tol, const bool verbose) 
{
    // Set up an SQPOptimizer instance that constrains all parameters to lie 
    // between 1e-10 and 1e+10 
    std::string poly_filename, vert_filename;
    if (mode == 1)
    {
        poly_filename = "polytopes/line-10-unbindingunity-plusbind-mutationtype.poly"; 
        vert_filename = "polytopes/line-5-unbindingunity-plusbind-mutationtype.vert"; 
    }
    else if (mode == 2)
    {
        poly_filename = "polytopes/line-10-unbindingunity-plusbind-dna.poly"; 
        vert_filename = "polytopes/line-5-unbindingunity-plusbind-dna-perbase.vert"; 
    }
    else    // mode == 0
    {
        poly_filename = "polytopes/line-10-unbindingunity-plusbind.poly"; 
        vert_filename = "polytopes/line-5-unbindingunity-plusbind.vert"; 
    }
    Polytopes::LinearConstraints<mpq_rational>* constraints_opt = new Polytopes::LinearConstraints<mpq_rational>(
        Polytopes::InequalityType::GreaterThanOrEqualTo 
    );
    constraints_opt->parse(poly_filename);
    const int D = constraints_opt->getD(); 
    const int N = constraints_opt->getN(); 
    SQPOptimizer<PreciseType>* opt = new SQPOptimizer<PreciseType>(constraints_opt);

    // Sample a set of points from an initial polytope in parameter space, 
    // which constrains all parameters to lie between 1e-5 and 1e+5
    Delaunay_triangulation* tri; 
    Matrix<PreciseType, Dynamic, Dynamic> init_points(ninit, D); 
    if (mode == 0 || mode == 1)
    {
        tri = new Delaunay_triangulation(D);
        Polytopes::parseVerticesFile(vert_filename, tri);
        init_points = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, ninit, 0, rng);
    }
    else    // mode == 2
    {
        tri = new Delaunay_triangulation(8);
        Polytopes::parseVerticesFile(vert_filename, tri);

        // For each base (A, C, G, T), generate a new set of coordinates  
        Matrix<PreciseType, Dynamic, Dynamic> init_coords_per_base;
        MatrixXi indices(4, 8); 
        indices <<  0,  1,  2,  3, 16, 17, 18, 19,
                    5,  4,  6,  7, 21, 20, 22, 23,
                   10,  8,  9, 11, 26, 24, 25, 27,
                   15, 12, 13, 14, 31, 28, 29, 30;
        for (int i = 0; i < 4; ++i)
        {
            init_coords_per_base = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, ninit, 0, rng);
            init_points(Eigen::all, indices.row(i)) = init_coords_per_base;
        }

        // Sample the last two coordinates to lie within 1e-5 and 1e+5 and 
        // otherwise be unconstrained
        double min = -5; 
        double max = 5; 
        boost::random::uniform_real_distribution<double> dist(min, max);
        for (int i = 0; i < ninit; ++i)
        {
            init_points(i, 32) = static_cast<PreciseType>(dist(rng));
            init_points(i, 33) = static_cast<PreciseType>(dist(rng));
        }
    }
    delete tri; 

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<PreciseType, Dynamic, Dynamic> best_fit(ninit, D); 
    Matrix<PreciseType, Dynamic, 1> x_init, l_init;
    Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats(ninit, 4 * length);
    Matrix<PreciseType, Dynamic, 1> errors(ninit);
    MatrixXi single_mismatch_seqs = MatrixXi::Ones(length, length) - MatrixXi::Identity(length, length); 
    for (int i = 0; i < ninit; ++i)
    {
        // Assemble initial parameter values
        x_init = init_points.row(i); 
        l_init = (
            Matrix<PreciseType, Dynamic, 1>::Ones(N)
            - constraints_opt->active(x_init.cast<mpq_rational>()).template cast<PreciseType>()
        );

        // Get the best-fit parameter values from the i-th initial parameter vector
        std::function<PreciseType(const Ref<const Matrix<PreciseType, Dynamic, 1> >&)> func; 
        if (mode == 0)
        {
            func = [
                &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
                &bind_conc, &logscale
            ](
                const Ref<const Matrix<PreciseType, Dynamic, 1> >& x
            )
            {
                return errorAgainstData(
                    x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    bind_conc, logscale
                ); 
            };
        }
        else    // mode == 1 or mode == 2
        {
            func = [
                &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
                &cleave_seq_match, &unbind_seq_match, &mode, &bind_conc,
                &logscale
            ](
                const Ref<const Matrix<PreciseType, Dynamic, 1> >& x
            )
            {
                return errorAgainstData(
                    x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    cleave_seq_match, unbind_seq_match, mode, bind_conc,
                    logscale
                ); 
            };
        }
        best_fit.row(i) = opt->run(
            func, x_init, l_init, tau, delta, beta, max_iter, tol, x_tol, method,
            regularize, regularize_weight, use_only_armijo, use_strong_wolfe,
            hessian_modify_max_iter, c1, c2, verbose
        );
        if (mode == 0)
        {
            errors(i) = errorAgainstData(
                best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                bind_conc, logscale
            );
        }
        else if (mode == 1)
        {
            errors(i) = errorAgainstData(
                best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_seq_match, unbind_seq_match, 1, bind_conc, logscale
            ); 
        }
        else    // mode == 2
        {
            errors(i) = errorAgainstData(
                best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_seq_match, unbind_seq_match, 2, bind_conc, logscale
            ); 
        }
        
        // Normalize error by the number of data points 
        errors(i) /= (cleave_data.rows() + unbind_data.rows());

        // Then compute the normalized cleavage statistics of the best-fit 
        // model against all single-mismatch substrates 
        Matrix<PreciseType, Dynamic, 4> fit_stats;
        if (mode == 0)
        {
            fit_stats = computeCleavageStats(
                best_fit.row(i), single_mismatch_seqs, bind_conc
            );
        }
        else if (mode == 1)
        {
            fit_stats = computeCleavageStatsMutationSpecific(
                best_fit.row(i), single_mismatch_seqs, cleave_seq_match, bind_conc
            ); 
        }
        else    // mode == 2
        {
            fit_stats = computeCleavageStatsBaseSpecific(
                best_fit.row(i), single_mismatch_seqs, cleave_seq_match, bind_conc
            ); 
        }
        for (int j = 0; j < length; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                // Invert all returned statistics  
                fit_single_mismatch_stats(i, 4 * j + k) = -fit_stats(j, k); 
            }
        }
    }
    delete opt;

    return std::make_tuple(best_fit, fit_single_mismatch_stats, errors);
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);

    /** ------------------------------------------------------- //
     *                    PARSE CONFIGURATIONS                  //
     *  --------------------------------------------------------*/ 
    boost::json::object json_data = parseConfigFile(argv[1]).as_object();

    // Check that input/output file paths were specified 
    if (!json_data.if_contains("cleave_data_filename") && !json_data.if_contains("unbind_data_filename"))
        throw std::runtime_error("At least one dataset must be specified");
    else if (!json_data.if_contains("cleave_data_filename"))
        json_data["cleave_data_filename"] = ""; 
    else if (!json_data.if_contains("unbind_data_filename"))
        json_data["unbind_data_filename"] = "";
    if (!json_data.if_contains("output_filename"))
        throw std::runtime_error("Output file path must be specified");
    std::string cleave_infilename = json_data["cleave_data_filename"].as_string().c_str();
    std::string unbind_infilename = json_data["unbind_data_filename"].as_string().c_str();
    std::string outfilename = json_data["output_filename"].as_string().c_str();

    // Check that a valid parametrization mode was specified 
    if (!json_data.if_contains("param_mode")) 
        throw std::runtime_error("Parametrization mode must be specified");
    else if (json_data["param_mode"].as_int64() < 0 || json_data["param_mode"].as_int64() > 2)
        throw std::invalid_argument("Invalid parametrization mode specified");
    const int mode = json_data["param_mode"].as_int64();

    // Assign default values for parameters that were not specified 
    bool data_specified_as_times = false;
    int ninit = 100; 
    int nfolds = 10;
    PreciseType bind_conc = 1e-9; 
    if (json_data.if_contains("data_specified_as_times"))
    {
        data_specified_as_times = json_data["data_specified_as_times"].as_bool();
    }
    if (json_data.if_contains("n_init"))
    {
        ninit = json_data["n_init"].as_int64();
        if (ninit <= 0)
            throw std::runtime_error("Invalid number of initial parameter vectors specified"); 
    }
    if (json_data.if_contains("n_folds"))
    {
        nfolds = json_data["n_folds"].as_int64();
        if (nfolds <= 0)
            throw std::runtime_error("Invalid number of folds specified");  
    }
    if (json_data.if_contains("bind_conc"))
    {
        bind_conc = static_cast<PreciseType>(json_data["bind_conc"].as_double());
        if (bind_conc <= 0)
            throw std::runtime_error("Invalid binding concentration specified"); 
    }
    
    // Parse SQP configurations
    PreciseType tau = 0.5;
    PreciseType delta = 1e-8; 
    PreciseType beta = 1e-4; 
    int max_iter = 1000; 
    PreciseType tol = 1e-8;       // Set the y-value tolerance to be small 
    QuasiNewtonMethod method = QuasiNewtonMethod::BFGS;
    RegularizationMethod regularize = RegularizationMethod::L2; 
    PreciseType regularize_weight = 0.1; 
    bool use_only_armijo = true;
    bool use_strong_wolfe = false;
    int hessian_modify_max_iter = 10000;
    PreciseType c1 = 1e-4;
    PreciseType c2 = 0.9;
    PreciseType x_tol = 10000;    // Set the x-value tolerance to be large 
    bool verbose = true;
    if (json_data.if_contains("sqp_config"))
    {
        boost::json::object sqp_data = json_data["sqp_config"].as_object(); 
        if (sqp_data.if_contains("tau"))
        {
            tau = static_cast<PreciseType>(sqp_data["tau"].as_double()); 
            if (tau <= 0)
                throw std::runtime_error("Invalid value for tau specified"); 
        }
        if (sqp_data.if_contains("delta"))
        {
            delta = static_cast<PreciseType>(sqp_data["delta"].as_double());
            if (delta <= 0)
                throw std::runtime_error("Invalid value for delta specified"); 
        }
        if (sqp_data.if_contains("beta"))
        {
            beta = static_cast<PreciseType>(sqp_data["beta"].as_double()); 
            if (beta <= 0)
                throw std::runtime_error("Invalid value for beta specified"); 
        }
        if (sqp_data.if_contains("max_iter"))
        {
            max_iter = sqp_data["max_iter"].as_int64(); 
            if (max_iter <= 0)
                throw std::runtime_error("Invalid value for max_iter specified"); 
        }
        if (sqp_data.if_contains("tol"))
        {
            tol = static_cast<PreciseType>(sqp_data["tol"].as_double());
            if (tol <= 0)
                throw std::runtime_error("Invalid value for tol specified"); 
        }
        if (sqp_data.if_contains("quasi_newton_method"))
        {
            // Check that the value is either 0, 1, 2
            int value = sqp_data["quasi_newton_method"].as_int64(); 
            if (value < 0 || value > 2)
                throw std::runtime_error("Invalid value for quasi_newton_method specified"); 
            method = static_cast<QuasiNewtonMethod>(value); 
        }
        if (sqp_data.if_contains("regularization_method"))
        {
            // Check that the value is either 0, 1, 2
            int value = sqp_data["regularization_method"].as_int64(); 
            if (value < 0 || value > 2)
                throw std::runtime_error("Invalid value for regularization_method specified"); 
            regularize = static_cast<RegularizationMethod>(value); 
        }
        if (sqp_data.if_contains("regularization_weight"))
        {
            regularize_weight = static_cast<PreciseType>(sqp_data["regularization_weight"].as_double());
            if (regularize_weight <= 0)
                throw std::runtime_error("Invalid value for regularize_weight specified"); 
        }
        if (sqp_data.if_contains("use_only_armijo"))
        {
            use_only_armijo = sqp_data["use_only_armijo"].as_bool();
        }
        if (sqp_data.if_contains("use_strong_wolfe"))
        {
            use_strong_wolfe = sqp_data["use_strong_wolfe"].as_bool(); 
        }
        if (sqp_data.if_contains("hessian_modify_max_iter"))
        {
            hessian_modify_max_iter = sqp_data["hessian_modify_max_iter"].as_int64(); 
            if (hessian_modify_max_iter <= 0)
                throw std::runtime_error("Invalid value for hessian_modify_max_iter specified"); 
        }
        if (sqp_data.if_contains("c1"))
        {
            c1 = static_cast<PreciseType>(sqp_data["c1"].as_double());
            if (c1 <= 0)
                throw std::runtime_error("Invalid value for c1 specified"); 
        }
        if (sqp_data.if_contains("c2"))
        {
            c2 = static_cast<PreciseType>(sqp_data["c2"].as_double());
            if (c2 <= 0)
                throw std::runtime_error("Invalid value for c2 specified"); 
        }
        if (sqp_data.if_contains("x_tol"))
        {
            x_tol = static_cast<PreciseType>(sqp_data["x_tol"].as_double());
            if (x_tol <= 0)
                throw std::runtime_error("Invalid value for x_tol specified"); 
        }
        if (sqp_data.if_contains("verbose"))
        {
            verbose = sqp_data["verbose"].as_bool();
        }
    }

    // Whether to evaluate the error function in log-scale 
    const bool logscale = false;
    
    // Parse measured cleavage rates and dead unbinding rates, along with the
    // mismatched sequences on which they were measured 
    int n_cleave_data = 0; 
    int n_unbind_data = 0;
    MatrixXi cleave_seqs = MatrixXi::Zero(0, length); 
    MatrixXi unbind_seqs = MatrixXi::Zero(0, length); 
    Matrix<PreciseType, Dynamic, 1> cleave_data = Matrix<PreciseType, Dynamic, 1>::Zero(0); 
    Matrix<PreciseType, Dynamic, 1> unbind_data = Matrix<PreciseType, Dynamic, 1>::Zero(0);
    
    // Parse the input file of (composite) cleavage rates, if one is given 
    std::ifstream infile;
    std::string line, token;
    std::string cleave_seq_match = "";
    std::string unbind_seq_match = "";  
    if (cleave_infilename.size() > 0)
    {
        infile.open(cleave_infilename);

        // If mode == 1 or mode == 2, the first line should specify the 
        // perfect-match sequence for the given dataset
        if (mode == 1 || mode == 2)
        {
            std::getline(infile, line);
            std::stringstream ss;
            ss << line;
            std::getline(ss, token, '\t'); 
            if (token.compare("MATCH") != 0)
                throw std::runtime_error(
                    "Perfect-match sequence not specified for non-binary parametrization mode"
                );
            std::getline(ss, token, '\t');
            if (token.size() != length)
                throw std::runtime_error(
                    "Parsed input perfect-match sequence of invalid length (!= 20)"
                );  
            cleave_seq_match = token; 
        } 

        // Parse each subsequent line in the file 
        while (std::getline(infile, line))
        {
            std::stringstream ss;
            ss << line;
            std::getline(ss, token, '\t');
            n_cleave_data++;
            cleave_seqs.conservativeResize(n_cleave_data, length);  
            cleave_data.conservativeResize(n_cleave_data);

            // Parse the sequence, character by character
            if (token.size() != length)
                throw std::runtime_error("Parsed input sequence of invalid length (!= 20)"); 
            for (int j = 0; j < length; ++j)
            {
                if (mode == 1 || mode == 2)
                {
                    if (token[j] == 'A')
                        cleave_seqs(n_cleave_data - 1, j) = 0;
                    else if (token[j] == 'C')
                        cleave_seqs(n_cleave_data - 1, j) = 1;
                    else if (token[j] == 'G')
                        cleave_seqs(n_cleave_data - 1, j) = 2; 
                    else    // token[j] == 'T'
                        cleave_seqs(n_cleave_data - 1, j) = 3;
                }
                else
                {
                    if (token[j] == '0')
                        cleave_seqs(n_cleave_data - 1, j) = 0; 
                    else
                        cleave_seqs(n_cleave_data - 1, j) = 1;
                }
            }

            // The second entry is the cleavage rate
            std::getline(ss, token, '\t'); 
            try
            {
                cleave_data(n_cleave_data - 1) = std::stod(token);
            }
            catch (const std::out_of_range& e)
            {
                cleave_data(n_cleave_data - 1) = 0; 
            }
        }
        infile.close();
    }

    // Parse the input file of unbinding rates, if one is given
    if (unbind_infilename.size() > 0)
    {
        infile.open(unbind_infilename);

        // If mode == 1 or mode == 2, the first line should specify the 
        // perfect-match sequence for the given dataset
        if (mode == 1 || mode == 2)
        {
            std::getline(infile, line);
            std::stringstream ss;
            ss << line;
            std::getline(ss, token, '\t'); 
            if (token.compare("MATCH") != 0)
                throw std::runtime_error(
                    "Perfect-match sequence not specified for non-binary parametrization mode"
                );
            std::getline(ss, token, '\t');
            if (token.size() != length)
                throw std::runtime_error(
                    "Parsed input perfect-match sequence of invalid length (!= 20)"
                );  
            unbind_seq_match = token; 
        } 

        // Parse each subsequent line in the file 
        while (std::getline(infile, line))
        {
            std::stringstream ss;
            ss << line;
            std::getline(ss, token, '\t');
            n_unbind_data++;
            unbind_seqs.conservativeResize(n_unbind_data, length); 
            unbind_data.conservativeResize(n_unbind_data);

            // Parse the sequence, character by character
            if (token.size() != length)
                throw std::runtime_error("Parsed input sequence of invalid length (!= 20)"); 
            for (int j = 0; j < length; ++j)
            {
                if (mode == 1 || mode == 2)
                {
                    if (token[j] == 'A')
                        unbind_seqs(n_unbind_data - 1, j) = 0;
                    else if (token[j] == 'C')
                        unbind_seqs(n_unbind_data - 1, j) = 1;
                    else if (token[j] == 'G')
                        unbind_seqs(n_unbind_data - 1, j) = 2; 
                    else    // token[j] == 'T'
                        unbind_seqs(n_unbind_data - 1, j) = 3;
                }
                else
                {
                    if (token[j] == '0')
                        unbind_seqs(n_unbind_data - 1, j) = 0; 
                    else
                        unbind_seqs(n_unbind_data - 1, j) = 1;
                }
            } 

            // The second entry is the unbinding rate being parsed
            std::getline(ss, token, '\t');
            try
            {
                unbind_data(n_unbind_data - 1) = std::stod(token);
            }
            catch (const std::out_of_range& e)
            {
                unbind_data(n_unbind_data - 1) = 0; 
            }
        }
        infile.close();
    }

    // Exit if no cleavage rates and no unbinding rates were specified 
    if (n_cleave_data == 0 && n_unbind_data == 0)
        throw std::runtime_error("Both cleavage rate and unbinding rate datasets are empty");

    // Define the two perfect-match sequences as integer vectors
    VectorXi cleave_seq_match_arr(length); 
    VectorXi unbind_seq_match_arr(length); 
    if (mode == 0)
    {
        cleave_seq_match_arr = VectorXi::Ones(length);
        unbind_seq_match_arr = VectorXi::Ones(length); 
    } 
    else   // Then we should have cleave_seq_match.size() == unbind_seq_match.size() == 20
    {
        for (int i = 0; i < length; ++i)
        {
            if (cleave_seq_match[i] == 'A')
                cleave_seq_match_arr(i) = 0;
            else if (cleave_seq_match[i] == 'C')
                cleave_seq_match_arr(i) = 1;
            else if (cleave_seq_match[i] == 'G')
                cleave_seq_match_arr(i) = 2;
            else    // cleave_seq_match[i] == 'T'
                cleave_seq_match_arr(i) = 3;

            if (unbind_seq_match[i] == 'A')
                unbind_seq_match_arr(i) = 0;
            else if (unbind_seq_match[i] == 'C')
                unbind_seq_match_arr(i) = 1;
            else if (unbind_seq_match[i] == 'G')
                unbind_seq_match_arr(i) = 2;
            else    // unbind_seq_match[i] == 'T'
                unbind_seq_match_arr(i) = 3;
        }
    }

    // Shuffle the rows of the datasets to ensure that there are no biases 
    // in sequence composition 
    PermutationMatrix<Dynamic, Dynamic> cleave_perm = getPermutation(n_cleave_data, rng);  
    PermutationMatrix<Dynamic, Dynamic> unbind_perm = getPermutation(n_unbind_data, rng);
    cleave_data = cleave_perm * cleave_data; 
    cleave_seqs = cleave_perm * cleave_seqs; 
    unbind_data = unbind_perm * unbind_data; 
    unbind_seqs = unbind_perm * unbind_seqs; 

    // Assume that there exist cleavage and dead unbinding rates for the
    // perfect-match substrate (if any cleavage/unbinding rates have been
    // specified at all) 
    int unbind_match_index = -1;
    for (int i = 0; i < unbind_seqs.rows(); ++i)
    {
        if ((mode == 0 && unbind_seqs.row(i).sum() == length) ||
            ((mode == 1 || mode == 2) && unbind_seqs.row(i).transpose() == unbind_seq_match_arr)
        )
        {
            unbind_match_index = i;
            break; 
        }
    } 
    if (n_unbind_data > 0 && unbind_match_index == -1)
    {
        throw std::runtime_error(
            "Cannot normalize given data without value corresponding to perfect-match substrate"
        );
    }
    int cleave_match_index = -1; 
    for (int i = 0; i < cleave_seqs.rows(); ++i)
    {
        if ((mode == 0 && cleave_seqs.row(i).sum() == length) ||
            ((mode == 1 || mode == 2) && cleave_seqs.row(i).transpose() == cleave_seq_match_arr)
        )
        {
            cleave_match_index = i; 
            break;
        }
    }
    if (n_cleave_data > 0 && cleave_match_index == -1)
    {
        throw std::runtime_error(
            "Cannot normalize given data without value corresponding to perfect-match substrate"
        );
    }

    // Normalize all given composite cleavage rates/times and dead unbinding
    // rates/times (with copies of the data matrices that were passed into
    // this function)
    Matrix<PreciseType, Dynamic, 1> unbind_data_norm(unbind_data.size());
    Matrix<PreciseType, Dynamic, 1> cleave_data_norm(cleave_data.size());
    if (data_specified_as_times)    // Data specified as times 
    {
        for (int i = 0; i < unbind_data_norm.size(); ++i)
            unbind_data_norm(i) = unbind_data(i) / unbind_data(unbind_match_index);   // rate on perfect / rate on mismatched
        for (int i = 0; i < cleave_data_norm.size(); ++i)
            cleave_data_norm(i) = cleave_data(cleave_match_index) / cleave_data(i);   // time on perfect / time on mismatched
    }
    else                            // Data specified as rates 
    { 
        for (int i = 0; i < unbind_data_norm.size(); ++i)
            unbind_data_norm(i) = unbind_data(unbind_match_index) / unbind_data(i);   // rate on perfect / rate on mismatched
        for (int i = 0; i < cleave_data_norm.size(); ++i)
            cleave_data_norm(i) = cleave_data(i) / cleave_data(cleave_match_index);   // time on perfect / time on mismatched
    }

    /** ----------------------------------------------------------------------- //
     *     RUN K-FOLD CROSS VALIDATION AGAINST GIVEN CLEAVAGE/UNBINDING DATA    //
     *  ----------------------------------------------------------------------- */
    std::tuple<Matrix<PreciseType, Dynamic, Dynamic>, 
               Matrix<PreciseType, Dynamic, Dynamic>, 
               Matrix<PreciseType, Dynamic, 1> > results;
    std::stringstream header_ss; 
    if (mode == 0)
    {
        header_ss << "match_forward\tmatch_reverse\tmismatch_forward\tmismatch_reverse\t"
                  << "terminal_cleave_rate\tterminal_bind_rate\terror\t";
    }
    else if (mode == 1)
    {
        header_ss << "match_forward\ttransition_forward\ttransversion_forward\t"
                  << "match_reverse\ttransition_reverse\ttransversion_reverse\t";
    }
    else    // mode == 2
    {
        std::string nucleotides = "ACGT";
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                header_ss << nucleotides[i] << nucleotides[j] << "_forward\t";
            }
        }
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                header_ss << nucleotides[i] << nucleotides[j] << "_reverse\t";
            }
        }
    }
    if (nfolds == 1)
    {
        results = fitLineParamsAgainstMeasuredRates(
            cleave_data_norm, unbind_data_norm, cleave_seqs, unbind_seqs, mode, 
            cleave_seq_match_arr, unbind_seq_match_arr, ninit, bind_conc,
            logscale, rng, tau, delta, beta, max_iter, tol, method, regularize,
            regularize_weight, use_only_armijo, use_strong_wolfe,
            hessian_modify_max_iter, c1, c2, x_tol, verbose
        );
        Matrix<PreciseType, Dynamic, Dynamic> best_fit = std::get<0>(results); 
        Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats = std::get<1>(results); 
        Matrix<PreciseType, Dynamic, 1> errors = std::get<2>(results); 

        // Output the fits to file 
        std::ofstream outfile(outfilename); 
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        outfile << "fit_attempt\t" << header_ss.str()
                << "terminal_cleave_rate\tterminal_bind_rate\terror\t";
        for (int i = 0; i < length; ++i)
        {
            outfile << "mm" << i << "_log_spec\t"
                    << "mm" << i << "_log_rapid\t"
                    << "mm" << i << "_log_deaddissoc\t"
                    << "mm" << i << "_log_composite_rapid";
            if (i == length - 1)
                outfile << std::endl;
            else 
                outfile << '\t';
        }
        for (int i = 0; i < ninit; ++i)
        {
            outfile << i << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fit.cols(); ++j)
                outfile << best_fit(i, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors(i) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            for (int j = 0; j < 4 * length - 1; ++j)
                outfile << fit_single_mismatch_stats(i, j) << '\t';
            outfile << fit_single_mismatch_stats(i, 4 * length - 1) << std::endl;  
        }
        outfile.close();
    }
    else
    {
        // Divide the indices in each dataset into folds
        auto unbind_fold_pairs = getFolds(n_unbind_data, nfolds);
        auto cleave_fold_pairs = getFolds(n_cleave_data, nfolds); 
        
        // For each fold ...
        Matrix<PreciseType, Dynamic, 1> unbind_data_train, unbind_data_test,
                                        cleave_data_train, cleave_data_test;
        MatrixXi unbind_seqs_train, unbind_seqs_test, cleave_seqs_train, cleave_seqs_test;
        Matrix<PreciseType, Dynamic, Dynamic> best_fit_total(nfolds, 0); 
        Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats_total(nfolds, 4 * length); 
        Matrix<PreciseType, Dynamic, 1> errors_against_test(nfolds); 
        for (int fi = 0; fi < nfolds; ++fi)
        {
            unbind_data_train = unbind_data_norm(unbind_fold_pairs[fi].first);
            unbind_data_test = unbind_data_norm(unbind_fold_pairs[fi].second);
            unbind_seqs_train = unbind_seqs(unbind_fold_pairs[fi].first, Eigen::all);
            unbind_seqs_test = unbind_seqs(unbind_fold_pairs[fi].second, Eigen::all); 
            cleave_data_train = cleave_data_norm(cleave_fold_pairs[fi].first); 
            cleave_data_test = cleave_data_norm(cleave_fold_pairs[fi].second); 
            cleave_seqs_train = cleave_seqs(cleave_fold_pairs[fi].first, Eigen::all); 
            cleave_seqs_test = cleave_seqs(cleave_fold_pairs[fi].second, Eigen::all);

            // Optimize model parameters on the training subset 
            results = fitLineParamsAgainstMeasuredRates(
                cleave_data_train, unbind_data_train, cleave_seqs_train,
                unbind_seqs_train, mode, cleave_seq_match_arr, unbind_seq_match_arr,
                ninit, bind_conc, logscale, rng, tau, delta, beta, max_iter, tol,
                method, regularize, regularize_weight, use_only_armijo,
                use_strong_wolfe, hessian_modify_max_iter, c1, c2, x_tol, verbose
            );
            Matrix<PreciseType, Dynamic, Dynamic> best_fit_per_fold = std::get<0>(results); 
            Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats_per_fold = std::get<1>(results); 
            Matrix<PreciseType, Dynamic, 1> errors_per_fold = std::get<2>(results);
            if (fi == 0) 
                best_fit_total.resize(nfolds, best_fit_per_fold.cols()); 

            // Find the parameter vector corresponding to the least error
            Eigen::Index minidx; 
            PreciseType minerror = errors_per_fold.minCoeff(&minidx); 
            Matrix<PreciseType, Dynamic, 1> best_fit = best_fit_per_fold.row(minidx); 
            Matrix<PreciseType, Dynamic, 1> fit_single_mismatch_stats = fit_single_mismatch_stats_per_fold.row(minidx);

            // Evaluate error against the test subset 
            PreciseType error_against_test = errorAgainstData(
                best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                unbind_data_test, bind_conc, logscale
            );
            best_fit_total.row(fi) = best_fit; 
            fit_single_mismatch_stats_total.row(fi) = fit_single_mismatch_stats; 
            errors_against_test(fi) = error_against_test; 
        }

        // Output the fits to file 
        std::ofstream outfile(outfilename); 
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        outfile << "fold\t" << header_ss.str()
                << "terminal_cleave_rate\tterminal_bind_rate\ttest_error\t";
        for (int i = 0; i < length; ++i)
        {
            outfile << "mm" << i << "_log_spec\t"
                    << "mm" << i << "_log_rapid\t"
                    << "mm" << i << "_log_deaddissoc\t"
                    << "mm" << i << "_log_composite_rapid";
            if (i == length - 1)
                outfile << std::endl;
            else 
                outfile << '\t';
        }
        for (int fi = 0; fi < nfolds; ++fi)
        {
            outfile << fi << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fit_total.cols(); ++j)
                outfile << best_fit_total(fi, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors_against_test(fi) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            for (int j = 0; j < 4 * length - 1; ++j)
                outfile << fit_single_mismatch_stats_total(fi, j) << '\t';
            outfile << fit_single_mismatch_stats_total(fi, 4 * length - 1) << std::endl;  
        }
        outfile.close();
    }

    return 0;
} 

