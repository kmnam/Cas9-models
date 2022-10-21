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
 * In this copy of the script, TEST blocks have been added to check that 
 * quantities have been computed correctly. 
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     10/21/2022
 */

#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <tuple>
#include <vector>
#include <boost/random.hpp>
#include <linearConstraints.hpp>
#include <SQP.hpp>             // Includes Eigen/Dense, CGAL/QP_*, Boost.Multiprecision, boostMultiprecisionEigen.hpp, etc.
#include <polytopes.hpp>       // Must be included after SQP.hpp
#include <graphs/line.hpp>
#include "../include/utils.hpp"

using namespace Eigen;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::abs;
using boost::multiprecision::min;
using boost::multiprecision::log10;
using boost::multiprecision::pow;
using boost::multiprecision::sqrt;
constexpr int INTERNAL_PRECISION = 100; 
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;

const unsigned length = 20;
const PreciseType ten("10");

/**
 * Return a division of the range `0, ..., n - 1` to the given number of folds.
 *
 * For instance, if `n == 6` and `nfolds == 3`, then the returned `std::vector`
 * contains the following: 
 *
 * ```
 * {
 *     {{2, 3, 4, 5}, {0, 1}},
 *     {{0, 1, 4, 5}, {2, 3}},
 *     {{0, 1, 2, 3}, {4, 5}}
 * }
 * ```
 *
 * @param n      Number of data points. 
 * @param nfolds Number of folds into which to divide the dataset. 
 * @returns      Collection of `nfolds` pairs of index-sets, each corresponding
 *               to the training and test subset for each fold.
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
 * Return a randomly generated permutation of the range `0, ..., n - 1`,
 * using the Fisher-Yates shuffle.
 *
 * @param n   Size of input range.
 * @param rng Random number generator instance.
 * @returns Permutation of the range `0, ..., n - 1`, as a permutation matrix.
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
 *   form A/A, A/C, A/G, A/T, ..., T/G, T/T;
 * - the second 16 entries are the reverse rates at DNA-RNA (mis)matches of the
 *   form A/A, A/C, A/G, A/T, ..., T/G, T/T;
 * - the last two entries are the terminal cleavage rate and binding rate, 
 *   respectively.
 *
 * @param logrates  Input vector of 34 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (A), 1 (C),
 *                  2 (G), or 3 (T).
 * @param seq_match Input perfect-match sequence, as a vector with entries of 
 *                  0 (A), 1 (C), 2 (G), or 3 (T).
 * @param bind_conc Input binding concentration.
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
 *
 * @param logrates  Input vector of 8 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (match w.r.t.
 *                  perfect-match sequence), 1 (transition w.r.t. perfect-match
 *                  sequence), or 2 (transversion w.r.t. perfect-match sequence).
 * @param seq_match Input perfect-match sequence, as a vector with entries of
 *                  0 (A), 1 (C), 2 (G), or 3 (T).
 * @param bind_conc Input binding concentration.
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
 *
 * Here, `logrates` is assumed to contain 6 entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal cleavage rate, and
 * 6) terminal binding rate.
 *
 * @param logrates  Input vector of 6 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (match w.r.t.
 *                  perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                  sequence).
 * @param bind_conc Input binding concentration.
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
std::pair<PreciseType, PreciseType> errorAgainstData(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                                                     const Ref<const MatrixXi>& cleave_seqs,
                                                     const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
                                                     const Ref<const MatrixXi>& unbind_seqs,
                                                     const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
                                                     const Ref<const VectorXi>& cleave_seq_match, 
                                                     const Ref<const VectorXi>& unbind_seq_match,
                                                     const int mode, const PreciseType bind_conc,
                                                     PreciseType cleave_error_weight = 1.0,
                                                     PreciseType unbind_error_weight = 1.0)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* *inverse* cleavage metrics and corresponding error
    // against data
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
   
    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    PreciseType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
    Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < stats1.rows(); ++i)
            stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
        for (int i = 0; i < stats2.rows(); ++i)
            stats2_transformed(i, j) = pow(ten, stats2(i, j));
    }
    const int n_cleave_data = cleave_data.size(); 
    const int n_unbind_data = unbind_data.size();
    Array<PreciseType, Dynamic, 1> cleave_denom(n_cleave_data);
    Array<PreciseType, Dynamic, 1> unbind_denom(n_unbind_data);
    for (int i = 0; i < n_cleave_data; ++i)
        cleave_denom(i) = (abs(cleave_data(i)) + abs(stats1_transformed(i, 3))) / 2;
    for (int i = 0; i < n_unbind_data; ++i)
        unbind_denom(i) = (abs(unbind_data(i)) + abs(stats2_transformed(i, 2))) / 2;
    PreciseType cleave_error = 0;
    PreciseType unbind_error = 0;
    if (n_cleave_data > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1_transformed.col(3) - cleave_data).array().abs() / cleave_denom).sum() / n_cleave_data
        );
    }
    if (n_unbind_data > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2_transformed.col(2) - unbind_data).array().abs() / unbind_denom).sum() / n_unbind_data
        );
    }

    return std::make_pair(cleave_error, unbind_error);
}

/**
 * Compute the *unregularized* error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the given LGPs and a set of
 * experimentally determined cleavage rates/times and (dead) unbinding
 * rates/times, with respect to a set of binary complementarity patterns.
 *
 * The error computed here is the mean absolute percentage error, defined as
 * error = |(true value - fit value) / true value|.
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates
 * @param cleave_seqs
 * @param cleave_data
 * @param unbind_seqs
 * @param unbind_data
 * @param bind_conc
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @returns Mean absolute percentage error against cleavage rate data and
 *          against unbinding rate data, as two separate values.  
 */
std::pair<PreciseType, PreciseType> meanAbsolutePercentageErrorAgainstData(
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
    const Ref<const MatrixXi>& cleave_seqs,
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
    const Ref<const MatrixXi>& unbind_seqs,
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
    const PreciseType bind_conc, PreciseType cleave_error_weight = 1.0,
    PreciseType unbind_error_weight = 1.0)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc);

    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    PreciseType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
    Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < stats1.rows(); ++i)
            stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
        for (int i = 0; i < stats2.rows(); ++i)
            stats2_transformed(i, j) = pow(ten, stats2(i, j));
    }

    // Compute each error as the mean absolute percentage error:
    // |(true value - fit value) / fit value|
    PreciseType cleave_error = 0;
    PreciseType unbind_error = 0;
    if (cleave_data.size() > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1_transformed.col(3) - cleave_data).array() / cleave_data.array()).abs().mean()
        );
    }
    if (unbind_data.size() > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2_transformed.col(2) - unbind_data).array() / unbind_data.array()).abs().mean()
        );
    }

    return std::make_pair(cleave_error, unbind_error);
}

/**
 * Compute the *unregularized* error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the given LGPs and a set of
 * experimentally determined cleavage rates/times and (dead) unbinding
 * rates/times, with respect to a set of binary complementarity patterns.
 *
 * The error computed here is the symmetric mean absolute percentage error,
 * defined as error = (|true value - fit value|) / ((|true value| + |fit value|) / 2).
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates
 * @param cleave_seqs
 * @param cleave_data
 * @param unbind_seqs
 * @param unbind_data
 * @param bind_conc
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @returns Symmetric mean absolute percentage error against cleavage rate
 *          data and against unbinding rate data, as two separate values.  
 */
std::pair<PreciseType, PreciseType> symmetricMeanAbsolutePercentageErrorAgainstData(
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
    const Ref<const MatrixXi>& cleave_seqs,
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
    const Ref<const MatrixXi>& unbind_seqs,
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
    const PreciseType bind_conc, PreciseType cleave_error_weight = 1.0,
    PreciseType unbind_error_weight = 1.0)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc);

    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    PreciseType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
    Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < stats1.rows(); ++i)
            stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
        for (int i = 0; i < stats2.rows(); ++i)
            stats2_transformed(i, j) = pow(ten, stats2(i, j));
    }

    // Compute each error as the symmetric mean absolute percentage error:
    // (|true value - fit value|) / ((|true value| + |fit value|) / 2)
    const int n_cleave_data = cleave_data.size(); 
    const int n_unbind_data = unbind_data.size();
    Array<PreciseType, Dynamic, 1> cleave_denom(n_cleave_data);
    Array<PreciseType, Dynamic, 1> unbind_denom(n_unbind_data);
    for (int i = 0; i < n_cleave_data; ++i)
        cleave_denom(i) = (abs(cleave_data(i)) + abs(stats1_transformed(i, 3))) / 2;
    for (int i = 0; i < n_unbind_data; ++i)
        unbind_denom(i) = (abs(unbind_data(i)) + abs(stats2_transformed(i, 2))) / 2;
    PreciseType cleave_error = 0; 
    PreciseType unbind_error = 0;
    if (n_cleave_data > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1_transformed.col(3) - cleave_data).array().abs() / cleave_denom).mean()
        );
    }
    if (n_unbind_data > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2_transformed.col(2) - unbind_data).array().abs() / unbind_denom).mean()
        );
    }

    return std::make_pair(cleave_error, unbind_error);
}

/**
 * Compute the *unregularized* error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the given LGPs and a set of
 * experimentally determined cleavage rates/times and (dead) unbinding
 * rates/times, with respect to a set of binary complementarity patterns.
 *
 * The error computed here is the minimum-based mean absolute percentage error,
 * defined as error = (|true value - fit value|) / (min(|true value|, |fit value|))
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates
 * @param cleave_seqs
 * @param cleave_data
 * @param unbind_seqs
 * @param unbind_data
 * @param bind_conc
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @returns Minimum-based mean absolute percentage error against cleavage rate
 *          data and against unbinding rate data, as two separate values.  
 */
std::pair<PreciseType, PreciseType> minBasedMeanAbsolutePercentageErrorAgainstData(
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
    const Ref<const MatrixXi>& cleave_seqs,
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
    const Ref<const MatrixXi>& unbind_seqs,
    const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
    const PreciseType bind_conc, PreciseType cleave_error_weight = 1.0,
    PreciseType unbind_error_weight = 1.0)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc);

    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    PreciseType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
    Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < stats1.rows(); ++i)
            stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
        for (int i = 0; i < stats2.rows(); ++i)
            stats2_transformed(i, j) = pow(ten, stats2(i, j));
    }

    // Compute each error as the minimum-based mean absolute percentage error:
    // (|true value - fit value|) / (min(|true value|, |fit value|))
    const int n_cleave_data = cleave_data.size(); 
    const int n_unbind_data = unbind_data.size();
    Array<PreciseType, Dynamic, 1> cleave_denom(n_cleave_data);
    Array<PreciseType, Dynamic, 1> unbind_denom(n_unbind_data);
    for (int i = 0; i < n_cleave_data; ++i)
        cleave_denom(i) = min(abs(cleave_data(i)), abs(stats1_transformed(i, 3)));
    for (int i = 0; i < n_unbind_data; ++i)
        unbind_denom(i) = min(abs(unbind_data(i)), abs(stats2_transformed(i, 2)));
    PreciseType cleave_error = 0;
    PreciseType unbind_error = 0;
    if (n_cleave_data > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1_transformed.col(3) - cleave_data).array().abs() / cleave_denom).mean()
        );
    }
    if (n_unbind_data > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2_transformed.col(2) - unbind_data).array().abs() / unbind_denom).mean()
        );
    }

    return std::make_pair(cleave_error, unbind_error);
}

/**
 * @param cleave_data
 * @param unbind_data
 * @param cleave_seqs
 * @param unbind_seqs
 * @param mode
 * @param error_mode
 * @param cleave_seq_match
 * @param unbind_seq_match
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @param ninit
 * @param bind_conc
 * @param rng
 * @param delta
 * @param beta
 * @param min_stepsize
 * @param max_iter
 * @param tol
 * @param x_tol
 * @param method
 * @param regularize
 * @param regularize_weight
 * @param hessian_modify_max_iter
 * @param c1
 * @param c2
 * @param line_search_max_iter
 * @param zoom_max_iter
 * @param verbose
 * @param search_verbose
 * @param zoom_verbose
 */
std::tuple<Matrix<PreciseType, Dynamic, Dynamic>, 
           Matrix<PreciseType, Dynamic, Dynamic>,
           Matrix<PreciseType, Dynamic, 1> >
    fitLineParamsAgainstMeasuredRates(const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data, 
                                      const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data, 
                                      const Ref<const MatrixXi>& cleave_seqs, 
                                      const Ref<const MatrixXi>& unbind_seqs,
                                      const int mode, const int error_mode,
                                      const Ref<const VectorXi>& cleave_seq_match,
                                      const Ref<const VectorXi>& unbind_seq_match,
                                      const PreciseType cleave_error_weight,
                                      const PreciseType unbind_error_weight,
                                      const int ninit, const PreciseType bind_conc,
                                      boost::random::mt19937& rng,
                                      const PreciseType delta, const PreciseType beta,
                                      const PreciseType min_stepsize, const int max_iter,
                                      const PreciseType tol, const PreciseType x_tol, 
                                      const QuasiNewtonMethod method, 
                                      const RegularizationMethod regularize,
                                      const PreciseType regularize_weight, 
                                      const int hessian_modify_max_iter, 
                                      const PreciseType c1, const PreciseType c2,
                                      const int line_search_max_iter,
                                      const int zoom_max_iter, const bool verbose,
                                      const bool search_verbose, const bool zoom_verbose) 
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

    // Define matrix of single-mismatch DNA sequences relative to the 
    // perfect-match sequence for cleavage rates
    MatrixXi single_mismatch_seqs;
    if (mode == 0)
    {
        single_mismatch_seqs = MatrixXi::Ones(length, length) - MatrixXi::Identity(length, length); 
    }
    else    // mode == 1 or 2
    {
        // Three possible single-mismatch substrates per mismatch position
        single_mismatch_seqs.resize(3 * length, length);
        for (int j = 0; j < length; ++j)
        {
            for (int k = 0; k < length; ++k)
            {
                single_mismatch_seqs(3 * j, k) = cleave_seq_match(k); 
                single_mismatch_seqs(3 * j + 1, k) = cleave_seq_match(k); 
                single_mismatch_seqs(3 * j + 2, k) = cleave_seq_match(k);
            }
            int seq_j = cleave_seq_match(j); 
            if (seq_j == 0)
            {
                single_mismatch_seqs(3 * j, j) = 1; 
                single_mismatch_seqs(3 * j + 1, j) = 2; 
                single_mismatch_seqs(3 * j + 2, j) = 3;
            }
            else if (seq_j == 1)
            {
                single_mismatch_seqs(3 * j, j) = 0;
                single_mismatch_seqs(3 * j + 1, j) = 2; 
                single_mismatch_seqs(3 * j + 2, j) = 3;
            }
            else if (seq_j == 2)
            {
                single_mismatch_seqs(3 * j, j) = 0;
                single_mismatch_seqs(3 * j + 1, j) = 1; 
                single_mismatch_seqs(3 * j + 2, j) = 3;
            }
            else    // seq_j == 3
            {
                single_mismatch_seqs(3 * j, j) = 0;
                single_mismatch_seqs(3 * j + 1, j) = 1; 
                single_mismatch_seqs(3 * j + 2, j) = 2;
            }
        }
    }

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<PreciseType, Dynamic, Dynamic> best_fit(ninit, D); 
    Matrix<PreciseType, Dynamic, 1> x_init, l_init;
    Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats;
    if (mode == 0)
        fit_single_mismatch_stats.resize(ninit, 4 * length);
    else    // mode == 1 or 2 
        fit_single_mismatch_stats.resize(ninit, 12 * length);
    Matrix<PreciseType, Dynamic, 1> errors(ninit);
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
            if (error_mode == 0)   // Mean absolute percentage error 
            {
                func = [
                    &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
                    &bind_conc, &cleave_error_weight, &unbind_error_weight
                ](const Ref<const Matrix<PreciseType, Dynamic, 1> >& x) -> PreciseType
                {
                    std::pair<PreciseType, PreciseType> error = meanAbsolutePercentageErrorAgainstData(
                        x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                        bind_conc, cleave_error_weight, unbind_error_weight
                    );
                    return error.first + error.second;
                };
            }
            else if (error_mode == 1)   // Symmetric mean absolute percentage error
            {
                func = [
                    &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
                    &bind_conc, &cleave_error_weight, &unbind_error_weight
                ](const Ref<const Matrix<PreciseType, Dynamic, 1> >& x) -> PreciseType
                {
                    std::pair<PreciseType, PreciseType> error = symmetricMeanAbsolutePercentageErrorAgainstData(
                        x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                        bind_conc, cleave_error_weight, unbind_error_weight
                    );
                    return error.first + error.second;
                };
            }
            else if (error_mode == 2)   // Minimum-based mean absolute percentage error
            {
                func = [
                    &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
                    &bind_conc, &cleave_error_weight, &unbind_error_weight
                ](const Ref<const Matrix<PreciseType, Dynamic, 1> >& x) -> PreciseType
                {
                    std::pair<PreciseType, PreciseType> error = minBasedMeanAbsolutePercentageErrorAgainstData(
                        x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                        bind_conc, cleave_error_weight, unbind_error_weight
                    );
                    return error.first + error.second;
                };
            }
        }
        else    // mode == 1 or mode == 2
        {
            func = [
                &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
                &cleave_seq_match, &unbind_seq_match, &mode, &bind_conc,
                &cleave_error_weight, &unbind_error_weight
            ](const Ref<const Matrix<PreciseType, Dynamic, 1> >& x) -> PreciseType
            {
                std::pair<PreciseType, PreciseType> error = errorAgainstData(
                    x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    cleave_seq_match, unbind_seq_match, mode, bind_conc,
                    cleave_error_weight, unbind_error_weight
                );
                return error.first + error.second;
            };
        }
        best_fit.row(i) = opt->run(
            func, x_init, l_init, delta, beta, min_stepsize, max_iter, tol,
            x_tol, method, regularize, regularize_weight, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, verbose, search_verbose,
            zoom_verbose
        );
        if (mode == 0)
        {
            std::pair<PreciseType, PreciseType> error;
            if (error_mode == 0)         // Mean absolute percentage error
            {
                error = meanAbsolutePercentageErrorAgainstData(
                    best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    bind_conc, cleave_error_weight, unbind_error_weight
                );
            }
            else if (error_mode == 1)    // Symmetric mean absolute percentage error
            {
                error = symmetricMeanAbsolutePercentageErrorAgainstData(
                    best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    bind_conc, cleave_error_weight, unbind_error_weight
                );
            }
            else if (error_mode == 2)    // Minimum-based mean absolute percentage error
            {
                error = minBasedMeanAbsolutePercentageErrorAgainstData(
                    best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    bind_conc, cleave_error_weight, unbind_error_weight
                );
            }
            errors(i) = error.first + error.second;
        }
        else if (mode == 1)
        {
            std::pair<PreciseType, PreciseType> error = errorAgainstData(
                best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_seq_match, unbind_seq_match, 1, bind_conc, cleave_error_weight,
                unbind_error_weight
            );
            errors(i) = error.first + error.second; 
        }
        else    // mode == 2
        {
            std::pair<PreciseType, PreciseType> error = errorAgainstData(
                best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_seq_match, unbind_seq_match, 2, bind_conc, cleave_error_weight,
                unbind_error_weight
            );
            errors(i) = error.first + error.second;
        }
        
        // Then compute the normalized cleavage statistics of the best-fit 
        // model against all single-mismatch substrates 
        Matrix<PreciseType, Dynamic, 4> fit_stats;
        if (mode == 0)
        {
            fit_stats = computeCleavageStats(best_fit.row(i), single_mismatch_seqs, bind_conc);
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
            if (mode == 0)
            {
                for (int k = 0; k < 4; ++k)
                {
                    // Invert all returned statistics  
                    fit_single_mismatch_stats(i, 4 * j + k) = -fit_stats(j, k); 
                }
            }
            else    // mode == 1 or 2
            {
                // For each mismatched sequence, write all four output metrics
                // as a group of entries
                for (int m = 0; m < 3; ++m)
                {
                    for (int k = 0; k < 4; ++k)
                    {
                        // Invert all returned statistics
                        fit_single_mismatch_stats(i, 12 * j + 4 * m + k) = -fit_stats(3 * j + m, k);
                    }
                }
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
    
    // Check that a valid error mode was specified
    if (!json_data.if_contains("error_mode"))
        throw std::runtime_error("Error mode must be specified"); 
    else if (json_data["error_mode"].as_int64() < 0 || json_data["error_mode"].as_int64() > 2)
        throw std::invalid_argument("Invalid error mode specified");  
    const int error_mode = json_data["error_mode"].as_int64();

    // Assign default values for parameters that were not specified 
    bool data_specified_as_times = false;
    int ninit = 100; 
    int nfolds = 10;
    PreciseType bind_conc = 1e-9;
    PreciseType cleave_error_weight = 1;
    PreciseType unbind_error_weight = 1;
    PreciseType cleave_pseudocount = 0;
    PreciseType unbind_pseudocount = 0; 
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
    if (json_data.if_contains("cleave_error_weight"))
    {
        cleave_error_weight = static_cast<PreciseType>(json_data["cleave_error_weight"].as_double()); 
        if (cleave_error_weight <= 0)
            throw std::runtime_error("Invalid cleavage rate error weight specified"); 
    }
    if (json_data.if_contains("unbind_error_weight"))
    {
        unbind_error_weight = static_cast<PreciseType>(json_data["unbind_error_weight"].as_double());
        if (unbind_error_weight <= 0)
            throw std::runtime_error("Invalid unbinding rate error weight specified");
    }
    if (json_data.if_contains("cleave_pseudocount"))
    {
        cleave_pseudocount = static_cast<PreciseType>(json_data["cleave_pseudocount"].as_double()); 
        if (cleave_pseudocount < 0)
            throw std::runtime_error("Invalid cleavage rate pseudocount specified");
    }
    if (json_data.if_contains("unbind_pseudocount"))
    {
        unbind_pseudocount = static_cast<PreciseType>(json_data["unbind_pseudocount"].as_double()); 
        if (unbind_pseudocount < 0)
            throw std::runtime_error("Invalid unbinding rate pseudocount specified");
    }
    
    // Parse SQP configurations
    PreciseType delta = 1e-9; 
    PreciseType beta = 1e-4;
    PreciseType min_stepsize = 1e-8;
    int max_iter = 1000; 
    PreciseType tol = 1e-8;       // Set the y-value tolerance to be small
    PreciseType x_tol = 10000;    // Set the x-value tolerance to be large 
    QuasiNewtonMethod method = QuasiNewtonMethod::BFGS;
    RegularizationMethod regularize = RegularizationMethod::L2; 
    PreciseType regularize_weight = 0.1; 
    int hessian_modify_max_iter = 10000;
    PreciseType c1 = 1e-4;            // Default value suggested by Nocedal and Wright
    PreciseType c2 = 0.9;             // Default value suggested by Nocedal and Wright
    int line_search_max_iter = 10;    // Default value in scipy.optimize.line_search()
    int zoom_max_iter = 10;           // Default value in scipy.optimize.line_search()
    bool verbose = true;
    bool search_verbose = false;
    bool zoom_verbose = false;
    if (json_data.if_contains("sqp_config"))
    {
        boost::json::object sqp_data = json_data["sqp_config"].as_object(); 
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
        if (sqp_data.if_contains("min_stepsize"))
        {
            min_stepsize = static_cast<PreciseType>(sqp_data["min_stepsize"].as_double()); 
            if (min_stepsize <= 0 || min_stepsize >= 1)              // Must be less than 1
                throw std::runtime_error("Invalid value for min_stepsize specified");
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
        if (sqp_data.if_contains("x_tol"))
        {
            x_tol = static_cast<PreciseType>(sqp_data["x_tol"].as_double());
            if (x_tol <= 0)
                throw std::runtime_error("Invalid value for x_tol specified"); 
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
        if (sqp_data.if_contains("hessian_modify_max_iter"))
        {
            hessian_modify_max_iter = sqp_data["hessian_modify_max_iter"].as_int64(); 
            if (hessian_modify_max_iter <= 0)
                throw std::runtime_error("Invalid value for hessian_modify_max_iter specified"); 
        }
        if (sqp_data.if_contains("line_search_max_iter"))
        {
            line_search_max_iter = sqp_data["line_search_max_iter"].as_int64(); 
            if (line_search_max_iter <= 0)
                throw std::runtime_error("Invalid value of line_search_max_iter specified");
        }
        if (sqp_data.if_contains("zoom_max_iter"))
        {
            zoom_max_iter = sqp_data["zoom_max_iter"].as_int64();
            if (zoom_max_iter <= 0)
                throw std::runtime_error("Invalid valud of zoom_max_iter specified");
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
        if (sqp_data.if_contains("verbose"))
        {
            verbose = sqp_data["verbose"].as_bool();
        }
        if (sqp_data.if_contains("line_search_verbose"))
        {
            search_verbose = sqp_data["line_search_verbose"].as_bool();
        }
        if (sqp_data.if_contains("zoom_verbose"))
        {
            zoom_verbose = sqp_data["zoom_verbose"].as_bool();
        }
    }

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

    // Add pseudocounts to the cleavage rates and unbinding rates 
    cleave_data += cleave_pseudocount * Matrix<PreciseType, Dynamic, 1>::Ones(n_cleave_data);
    unbind_data += unbind_pseudocount * Matrix<PreciseType, Dynamic, 1>::Ones(n_unbind_data);

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

    // ------------------------------------------------------------------ //
    // TEST that datasets are correctly shuffled
    // ------------------------------------------------------------------ //
    MatrixXi cleave_perm_dense = cleave_perm.toDenseMatrix();
    MatrixXi unbind_perm_dense = unbind_perm.toDenseMatrix();
    assert(cleave_perm_dense.rows() == n_cleave_data); 
    assert(cleave_perm_dense.cols() == n_cleave_data);
    assert(unbind_perm_dense.rows() == n_unbind_data);
    assert(unbind_perm_dense.cols() == n_unbind_data); 
    assert(cleave_perm_dense.transpose() * cleave_perm_dense == MatrixXi::Identity(n_cleave_data, n_cleave_data)); 
    assert(unbind_perm_dense.transpose() * unbind_perm_dense == MatrixXi::Identity(n_unbind_data, n_unbind_data));
    assert((cleave_perm_dense.array() >= 0).all()); 
    assert((unbind_perm_dense.array() >= 0).all());
    assert((cleave_perm_dense.rowwise().sum().array() == 1).all());
    assert((cleave_perm_dense.colwise().sum().array() == 1).all());
    assert((unbind_perm_dense.rowwise().sum().array() == 1).all());
    assert((unbind_perm_dense.colwise().sum().array() == 1).all());
    auto cleave_perm_indices = cleave_perm.indices(); 
    auto unbind_perm_indices = unbind_perm.indices(); 
    Matrix<PreciseType, Dynamic, Dynamic> cleave_data_orig = cleave_perm.transpose() * cleave_data;
    MatrixXi cleave_seqs_orig = cleave_perm.transpose() * cleave_seqs; 
    Matrix<PreciseType, Dynamic, Dynamic> unbind_data_orig = unbind_perm.transpose() * unbind_data;
    MatrixXi unbind_seqs_orig = unbind_perm.transpose() * unbind_seqs;
    int i = 0;
    for (auto it = cleave_perm_indices.begin(); it != cleave_perm_indices.end(); ++it)
    {
        // What is the image of i under the permutation?
        int j = *it;
        assert(cleave_perm_dense(j, i) == 1); 

        // Check that the permutation maps the i-th entry in the dataset to 
        // the correct position in the permuted dataset
        assert(cleave_data_orig(i) == cleave_data(j));
        assert(cleave_seqs_orig.row(i) == cleave_seqs.row(j));

        i++;
    }
    i = 0;
    for (auto it = unbind_perm_indices.begin(); it != unbind_perm_indices.end(); ++it)
    {
        // What is the image of i under the permutation?
        int j = *it;
        assert(unbind_perm_dense(j, i) == 1); 

        // Check that the permutation maps the i-th entry in the dataset to 
        // the correct position in the permuted dataset
        assert(unbind_data_orig(i) == unbind_data(j));
        assert(unbind_seqs_orig.row(i) == unbind_seqs.row(j));

        i++;
    }
    // ------------------------------------------------------------------ //

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

    // ------------------------------------------------------------------ //
    // TEST that perfect-match sequences were correctly identified
    // ------------------------------------------------------------------ //
    assert(cleave_match_index >= 0);
    assert(unbind_match_index >= 0);
    if (mode == 0)
    {
        assert((cleave_seqs.row(cleave_match_index).array() == 1).all());
        assert((unbind_seqs.row(unbind_match_index).array() == 1).all());
    }

    // Normalize all given composite cleavage rates/times and dead unbinding
    // rates/times (with copies of the data matrices that were passed into
    // this function)
    Matrix<PreciseType, Dynamic, 1> unbind_data_norm(unbind_data.size());
    Matrix<PreciseType, Dynamic, 1> cleave_data_norm(cleave_data.size());
    if (data_specified_as_times)    // Data specified as times 
    {
        // The normalized data are *inverse* specific dissociativities and rapidities
        for (int i = 0; i < unbind_data_norm.size(); ++i)
            unbind_data_norm(i) = unbind_data(i) / unbind_data(unbind_match_index);   // rate on perfect / rate on mismatched
        for (int i = 0; i < cleave_data_norm.size(); ++i)
            cleave_data_norm(i) = cleave_data(cleave_match_index) / cleave_data(i);   // time on perfect / time on mismatched
    }
    else                            // Data specified as rates 
    {
        // The normalized data are *inverse* specific dissociativities and rapidities
        for (int i = 0; i < unbind_data_norm.size(); ++i)
            unbind_data_norm(i) = unbind_data(unbind_match_index) / unbind_data(i);   // rate on perfect / rate on mismatched
        for (int i = 0; i < cleave_data_norm.size(); ++i)
            cleave_data_norm(i) = cleave_data(i) / cleave_data(cleave_match_index);   // time on perfect / time on mismatched
    }

    // ------------------------------------------------------------------ //
    // TEST that cleavage rates and unbinding rates were correctly normalized
    // ------------------------------------------------------------------ //
    for (int i = 0; i < cleave_data_norm.size(); ++i)
    {
        // Each inverse rapidity is (rate on mismatched) / (rate on perfect)
        // = (time on perfect) / (time on mismatched)
        if (data_specified_as_times)
            assert(cleave_data_norm(i) == cleave_data(cleave_match_index) / cleave_data(i));
        else 
            assert(cleave_data_norm(i) == cleave_data(i) / cleave_data(cleave_match_index)); 
    }
    for (int i = 0; i < unbind_data_norm.size(); ++i)
    {
        // Each inverse dissociativity is (rate on perfect) / (rate on mismatched)
        // = (time on mismatched) / (time on perfect)
        if (data_specified_as_times)
            assert(unbind_data_norm(i) == unbind_data(i) / unbind_data(unbind_match_index));
        else
            assert(unbind_data_norm(i) == unbind_data(unbind_match_index) / unbind_data(i));
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
        header_ss << "match_forward\tmatch_reverse\tmismatch_forward\tmismatch_reverse\t";
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
            error_mode, cleave_seq_match_arr, unbind_seq_match_arr,
            cleave_error_weight, unbind_error_weight, ninit, bind_conc, rng,
            delta, beta, min_stepsize, max_iter, tol, x_tol, method, regularize,
            regularize_weight, hessian_modify_max_iter, c1, c2,
            line_search_max_iter, zoom_max_iter, verbose, search_verbose,
            zoom_verbose
        );
        Matrix<PreciseType, Dynamic, Dynamic> best_fit = std::get<0>(results); 
        Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats = std::get<1>(results); 
        Matrix<PreciseType, Dynamic, 1> errors = std::get<2>(results); 

        // Output the fits to file 
        std::ofstream outfile(outfilename); 
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        outfile << "fit_attempt\t" << header_ss.str()
                << "terminal_cleave_rate\tterminal_bind_rate\terror\t";
        if (mode == 0)
        {
            for (int i = 0; i < length; ++i)
            {
                outfile << "mm" << i << "_log_spec\t"
                        << "mm" << i << "_log_rapid\t"
                        << "mm" << i << "_log_deaddissoc\t"
                        << "mm" << i << "_log_composite_rapid\t";
            }
        }
        else    // mode == 1 or 2
        {
            for (int i = 0; i < length; ++i)
            {
                char seq_match_i = cleave_seq_match[i];
                std::string mismatch_bases;
                if (seq_match_i == 'A')      mismatch_bases = "CGT";
                else if (seq_match_i == 'C') mismatch_bases = "AGT";
                else if (seq_match_i == 'G') mismatch_bases = "ACT";
                else                         mismatch_bases = "ACG";
                for (int j = 0; j < 3; ++j)
                {
                    outfile << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_spec\t"
                            << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_rapid\t"
                            << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_deaddissoc\t"
                            << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_composite_rapid\t";
                }
            }
        }
        int pos = outfile.tellp();
        outfile.seekp(pos - 1);
        outfile << std::endl;  
        for (int i = 0; i < ninit; ++i)
        {
            outfile << i << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fit.cols(); ++j)
                outfile << best_fit(i, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors(i) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            for (int j = 0; j < fit_single_mismatch_stats.cols() - 1; ++j)
                outfile << fit_single_mismatch_stats(i, j) << '\t';
            outfile << fit_single_mismatch_stats(i, fit_single_mismatch_stats.cols() - 1) << std::endl;  
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
        Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats_total(nfolds, 0);
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
                unbind_seqs_train, mode, error_mode, cleave_seq_match_arr,
                unbind_seq_match_arr, cleave_error_weight, unbind_error_weight,
                ninit, bind_conc, rng, delta, beta, min_stepsize, max_iter, tol,
                x_tol, method, regularize, regularize_weight, hessian_modify_max_iter,
                c1, c2, line_search_max_iter, zoom_max_iter, verbose, search_verbose,
                zoom_verbose
            );
            Matrix<PreciseType, Dynamic, Dynamic> best_fit_per_fold = std::get<0>(results); 
            Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats_per_fold = std::get<1>(results); 
            Matrix<PreciseType, Dynamic, 1> errors_per_fold = std::get<2>(results);
            if (fi == 0)
            {
                best_fit_total.resize(nfolds, best_fit_per_fold.cols());
                fit_single_mismatch_stats_total.resize(nfolds, fit_single_mismatch_stats_per_fold.cols());
            } 

            // Find the parameter vector corresponding to the least error
            Eigen::Index minidx; 
            PreciseType minerror = errors_per_fold.minCoeff(&minidx); 
            Matrix<PreciseType, Dynamic, 1> best_fit = best_fit_per_fold.row(minidx); 
            Matrix<PreciseType, Dynamic, 1> fit_single_mismatch_stats = fit_single_mismatch_stats_per_fold.row(minidx);

            // Evaluate error against the test subset 
            std::pair<PreciseType, PreciseType> error_against_test;
            if (error_mode == 0)         // Mean absolute percentage error
            {
                error_against_test = meanAbsolutePercentageErrorAgainstData(
                    best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                    unbind_data_test, bind_conc, cleave_error_weight, unbind_error_weight
                );
            }
            else if (error_mode == 1)    // Symmetric mean absolute percentage error
            {
                error_against_test = symmetricMeanAbsolutePercentageErrorAgainstData(
                    best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                    unbind_data_test, bind_conc, cleave_error_weight, unbind_error_weight
                );
            }
            else if (error_mode == 2)    // Minimum-based mean absolute percentage error
            {
                error_against_test = minBasedMeanAbsolutePercentageErrorAgainstData(
                    best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                    unbind_data_test, bind_conc, cleave_error_weight, unbind_error_weight
                );
            }
            best_fit_total.row(fi) = best_fit; 
            fit_single_mismatch_stats_total.row(fi) = fit_single_mismatch_stats; 
            errors_against_test(fi) = error_against_test.first + error_against_test.second; 
        }

        // Output the fits to file 
        std::ofstream outfile(outfilename); 
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        outfile << "fold\t" << header_ss.str()
                << "terminal_cleave_rate\tterminal_bind_rate\ttest_error\t";
        if (mode == 0)
        {
            for (int i = 0; i < length; ++i)
            {
                outfile << "mm" << i << "_log_spec\t"
                        << "mm" << i << "_log_rapid\t"
                        << "mm" << i << "_log_deaddissoc\t"
                        << "mm" << i << "_log_composite_rapid\t";
            }
        }
        else    // mode == 1 or 2
        {
            for (int i = 0; i < length; ++i)
            {
                char seq_match_i = cleave_seq_match[i];
                std::string mismatch_bases;
                if (seq_match_i == 'A')      mismatch_bases = "CGT";
                else if (seq_match_i == 'C') mismatch_bases = "AGT";
                else if (seq_match_i == 'G') mismatch_bases = "ACT";
                else                         mismatch_bases = "ACG";
                for (int j = 0; j < 3; ++j)
                {
                    outfile << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_spec\t"
                            << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_rapid\t"
                            << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_deaddissoc\t"
                            << "mm" << i << "_" << seq_match_i << mismatch_bases[j] << "_log_composite_rapid\t";
                }
            }
        }
        int pos = outfile.tellp();
        outfile.seekp(pos - 1);
        outfile << std::endl;  
        for (int fi = 0; fi < nfolds; ++fi)
        {
            outfile << fi << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fit_total.cols(); ++j)
                outfile << best_fit_total(fi, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors_against_test(fi) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            for (int j = 0; j < fit_single_mismatch_stats_total.cols() - 1; ++j)
                outfile << fit_single_mismatch_stats_total(fi, j) << '\t';
            outfile << fit_single_mismatch_stats_total(fi, fit_single_mismatch_stats_total.cols() - 1) << std::endl;
        }
        outfile.close();
    }

    return 0;
} 

