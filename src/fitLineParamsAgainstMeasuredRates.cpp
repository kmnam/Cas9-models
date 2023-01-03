/**
 * Using line-search SQP, identify the set of line graph parameter vectors 
 * that yields each given set of unbinding and cleavage rates in the given
 * data files.
 *
 * Abbreviations in the below comments:
 * - LG:   line graph
 * - LGPs: line graph parameters
 * - QP:   quadratic programming
 * - SQP:  sequential quadratic programming
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     1/3/2023
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <tuple>
#include <vector>
#include <boost/random.hpp>
#include <SQP.hpp>                 // Includes linearConstraints.hpp from convex-polytopes, which 
                                   // includes Eigen/Dense, quadraticProgram.hpp, CGAL/QP_*,
                                   // boost/multiprecision/gmp.hpp, boostMultiprecisionEigen.hpp; 
                                   // additionally includes boost/multiprecision/mpfr.hpp
#include <polytopes.hpp>           // Must be included after SQP.hpp
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
constexpr int MAIN_PRECISION = 30; 
typedef number<mpfr_float_backend<MAIN_PRECISION> > MainType;
typedef number<mpfr_float_backend<100> >            PreciseType;
const int length = 20;
const MainType ten_main(10);
const PreciseType ten_precise(10);

/**
 * Return a division of the range `0, ..., n - 1` to the given number of folds.
 *
 * For instance, if `n == 6` and `nfolds == 3`, then the returned `std::vector<int>`
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
 * Return a subsample of the range `0, ..., n - 1` chosen *with* replacement.
 *
 * @param n   Size of input range.
 * @param k   Size of desired subsample.
 * @param rng Random number generator instance.
 * @returns Subsample of input range. 
 */
std::vector<int> sampleWithReplacement(const int n, const int k, boost::random::mt19937& rng)
{
    boost::random::uniform_int_distribution<> dist(0, n - 1); 
    std::vector<int> sample; 
    for (int i = 0; i < k; ++i)
        sample.push_back(dist(rng)); 

    return sample; 
}

/**
 * Compute the dead unbinding rate and composite cleavage rate on all given
 * mismatched sequences specified in the given matrix of complementarity
 * patterns, for the given LG.
 *
 * Here, `logrates` is assumed to contain 7 entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate, and
 * 7) terminal binding rate (units of inverse (M * sec)).
 *
 * @param logrates  Input vector of 7 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (match w.r.t.
 *                  perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                  sequence).
 * @param bind_conc Concentration of available Cas9. 
 */
Matrix<MainType, Dynamic, 2> computeCleavageStats(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                  const Ref<const MatrixXi>& seqs,
                                                  const MainType bind_conc) 
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten_precise, static_cast<PreciseType>(logrates(0))),
        pow(ten_precise, static_cast<PreciseType>(logrates(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten_precise, static_cast<PreciseType>(logrates(2))),
        pow(ten_precise, static_cast<PreciseType>(logrates(3)))
    );

    // Exit rates from terminal nodes 
    PreciseType terminal_unbind_rate = pow(ten_precise, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten_precise, static_cast<PreciseType>(logrates(5))); 

    // Binding rate entering state 0
    PreciseType bind_rate = pow(ten_precise, static_cast<PreciseType>(logrates(6))) * static_cast<PreciseType>(bind_conc);

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute dead unbinding rate against perfect-match substrate
    PreciseType prob_perfect = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
    PreciseType cleave_rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    PreciseType dead_unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate);
    PreciseType live_unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
    PreciseType term = 1 / prob_perfect;
    PreciseType composite_cleave_time_perfect = (
        (term / bind_rate) + (1 / cleave_rate_perfect) + ((term - 1) / live_unbind_rate_perfect)
    );

    // Compute dead unbinding rate and composite cleavage rate against each
    // given mismatched substrate
    Matrix<PreciseType, Dynamic, 2> stats(seqs.rows(), 2);  
    for (int i = 0; i < seqs.rows(); ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            if (seqs(i, j))
                model->setEdgeLabels(j, match_rates);
            else
                model->setEdgeLabels(j, mismatch_rates);
        }
        PreciseType prob = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        PreciseType cleave_rate = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        PreciseType dead_unbind_rate = model->getLowerExitRate(terminal_unbind_rate); 
        PreciseType live_unbind_rate = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        term = 1 / prob;
        PreciseType composite_cleave_time = (term / bind_rate) + (1 / cleave_rate) + ((term - 1) / live_unbind_rate);

        // Compute *inverse* specific dissociativity: rate on perfect / rate on mismatched 
        stats(i, 0) = log10(dead_unbind_rate_perfect) - log10(dead_unbind_rate);

        // Compute *inverse* composite cleavage rate ratio: rate on mismatched / rate on perfect
        // or time on perfect / time on mismatched
        stats(i, 1) = log10(composite_cleave_time_perfect) - log10(composite_cleave_time);
    }

    delete model;
    return stats.template cast<MainType>();
}

/**
 * Compute the cleavage probability, cleavage rate, dead unbinding rate, 
 * composite cleavage rate, and all associated normalized statistics on all
 * given mismatched sequences specified in the given matrix of complementarity
 * patterns, for the given LG.
 *
 * Here, `logrates` is assumed to contain 7 entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate, and
 * 7) terminal binding rate (units of inverse (M * sec)).
 *
 * @param logrates  Input vector of 7 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (match w.r.t.
 *                  perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                  sequence).
 * @param bind_conc Concentration of available Cas9. 
 */
Matrix<MainType, Dynamic, 8> computeCleavageStatsAll(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                     const Ref<const MatrixXi>& seqs,
                                                     const MainType bind_conc) 
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten_precise, static_cast<PreciseType>(logrates(0))),
        pow(ten_precise, static_cast<PreciseType>(logrates(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten_precise, static_cast<PreciseType>(logrates(2))),
        pow(ten_precise, static_cast<PreciseType>(logrates(3)))
    );

    // Exit rates from terminal nodes 
    PreciseType terminal_unbind_rate = pow(ten_precise, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten_precise, static_cast<PreciseType>(logrates(5))); 

    // Binding rate entering state 0
    PreciseType bind_rate = pow(ten_precise, static_cast<PreciseType>(logrates(6))) * static_cast<PreciseType>(bind_conc);

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute dead unbinding rate against perfect-match substrate
    PreciseType prob_perfect = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
    PreciseType cleave_rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    PreciseType dead_unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate);
    PreciseType live_unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
    PreciseType term = 1 / prob_perfect;
    PreciseType composite_cleave_time_perfect = (
        (term / bind_rate) + (1 / cleave_rate_perfect) + ((term - 1) / live_unbind_rate_perfect)
    );

    // Compute dead unbinding rate and composite cleavage rate against each
    // given mismatched substrate
    Matrix<PreciseType, Dynamic, 8> stats(seqs.rows(), 8);  
    for (int i = 0; i < seqs.rows(); ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            if (seqs(i, j))
                model->setEdgeLabels(j, match_rates);
            else
                model->setEdgeLabels(j, mismatch_rates);
        }
        PreciseType prob = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        PreciseType cleave_rate = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        PreciseType dead_unbind_rate = model->getLowerExitRate(terminal_unbind_rate); 
        PreciseType live_unbind_rate = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        term = 1 / prob;
        PreciseType composite_cleave_time = (term / bind_rate) + (1 / cleave_rate) + ((term - 1) / live_unbind_rate);
        stats(i, 0) = prob; 
        stats(i, 1) = cleave_rate; 
        stats(i, 2) = dead_unbind_rate; 
        stats(i, 3) = live_unbind_rate;

        // Compute cleavage specificity: prob on perfect / prob on mismatched
        stats(i, 4) = log10(prob_perfect) - log10(prob);

        // Compute specific rapidity: rate on perfect / rate on mismatched 
        stats(i, 5) = log10(cleave_rate_perfect) - log10(cleave_rate);  

        // Compute specific dissociativity: rate on mismatched / rate on perfect
        stats(i, 6) = log10(dead_unbind_rate) - log10(dead_unbind_rate_perfect);

        // Compute composite cleavage rate ratio: rate on perfect / rate on mismatched
        // or time on mismatched / time on perfect
        stats(i, 7) = log10(composite_cleave_time) - log10(composite_cleave_time_perfect);
    }

    delete model;
    return stats.template cast<MainType>();
}


/**
 * Compute the *unregularized* error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the given LGPs and a set of
 * experimentally determined cleavage rates and (dead) unbinding rates, with
 * respect to a set of binary complementarity patterns.
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
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @param bind_conc
 * @returns Mean absolute percentage error against cleavage rate data and
 *          against unbinding rate data, as two separate values.  
 */
std::pair<MainType, MainType> meanAbsolutePercentageErrorAgainstData(
    const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
    const Ref<const MatrixXi>& cleave_seqs,
    const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data,
    const Ref<const MatrixXi>& unbind_seqs,
    const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data,
    MainType cleave_error_weight = 1, MainType unbind_error_weight = 1,
    MainType bind_conc = 1e-7)
{
    Matrix<MainType, Dynamic, 1> stats1, stats2; 

    // Compute cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc).col(1);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc).col(0);

    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    MainType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    for (int i = 0; i < stats1.size(); ++i)
        stats1(i) = pow(ten_main, stats1(i)); 
    for (int i = 0; i < stats2.size(); ++i)
        stats2(i) = pow(ten_main, stats2(i));

    // Compute each error as the mean absolute percentage error:
    // |(true value - fit value) / fit value|
    MainType cleave_error = 0;
    MainType unbind_error = 0;
    if (cleave_data.size() > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1 - cleave_data).array() / cleave_data.array()).abs().mean()
        );
    }
    if (unbind_data.size() > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2 - unbind_data).array() / unbind_data.array()).abs().mean()
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
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @param bind_conc
 * @returns Symmetric mean absolute percentage error against cleavage rate
 *          data and against unbinding rate data, as two separate values.  
 */
std::pair<MainType, MainType> symmetricMeanAbsolutePercentageErrorAgainstData(
    const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
    const Ref<const MatrixXi>& cleave_seqs,
    const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data,
    const Ref<const MatrixXi>& unbind_seqs,
    const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data,
    MainType cleave_error_weight = 1, MainType unbind_error_weight = 1,
    MainType bind_conc = 1e-7)
{
    Matrix<MainType, Dynamic, 1> stats1, stats2; 

    // Compute cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc).col(1);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc).col(0);

    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    MainType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    for (int i = 0; i < stats1.size(); ++i)
        stats1(i) = pow(ten_main, stats1(i)); 
    for (int i = 0; i < stats2.size(); ++i)
        stats2(i) = pow(ten_main, stats2(i));

    // Compute each error as the symmetric mean absolute percentage error:
    // (|true value - fit value|) / ((|true value| + |fit value|) / 2)
    const int n_cleave_data = cleave_data.size(); 
    const int n_unbind_data = unbind_data.size();
    Array<MainType, Dynamic, 1> cleave_denom(n_cleave_data);
    Array<MainType, Dynamic, 1> unbind_denom(n_unbind_data);
    for (int i = 0; i < n_cleave_data; ++i)
        cleave_denom(i) = (abs(cleave_data(i)) + abs(stats1(i))) / 2;
    for (int i = 0; i < n_unbind_data; ++i)
        unbind_denom(i) = (abs(unbind_data(i)) + abs(stats2(i))) / 2;
    MainType cleave_error = 0; 
    MainType unbind_error = 0;
    if (n_cleave_data > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1 - cleave_data).array().abs() / cleave_denom).mean()
        );
    }
    if (n_unbind_data > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2 - unbind_data).array().abs() / unbind_denom).mean()
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
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @param bind_conc
 * @returns Minimum-based mean absolute percentage error against cleavage rate
 *          data and against unbinding rate data, as two separate values.  
 */
std::pair<MainType, MainType> minBasedMeanAbsolutePercentageErrorAgainstData(
    const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
    const Ref<const MatrixXi>& cleave_seqs,
    const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data,
    const Ref<const MatrixXi>& unbind_seqs,
    const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data,
    MainType cleave_error_weight = 1, MainType unbind_error_weight = 1,
    MainType bind_conc = 1e-7)
{
    Matrix<MainType, Dynamic, 1> stats1, stats2; 

    // Compute cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc).col(1);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc).col(0);

    // Normalize error weights to sum to *two* (so that both weights equaling 
    // one means that the weights can be effectively ignored)
    MainType weight_mean = (cleave_error_weight + unbind_error_weight) / 2; 
    cleave_error_weight /= weight_mean;
    unbind_error_weight /= weight_mean;
    for (int i = 0; i < stats1.size(); ++i)
        stats1(i) = pow(ten_main, stats1(i)); 
    for (int i = 0; i < stats2.size(); ++i)
        stats2(i) = pow(ten_main, stats2(i));

    // Compute each error as the minimum-based mean absolute percentage error:
    // (|true value - fit value|) / (min(|true value|, |fit value|))
    const int n_cleave_data = cleave_data.size(); 
    const int n_unbind_data = unbind_data.size();
    Array<MainType, Dynamic, 1> cleave_denom(n_cleave_data);
    Array<MainType, Dynamic, 1> unbind_denom(n_unbind_data);
    for (int i = 0; i < n_cleave_data; ++i)
        cleave_denom(i) = min(abs(cleave_data(i)), abs(stats1(i)));
    for (int i = 0; i < n_unbind_data; ++i)
        unbind_denom(i) = min(abs(unbind_data(i)), abs(stats2(i)));
    MainType cleave_error = 0;
    MainType unbind_error = 0;
    if (n_cleave_data > 0)
    {
        cleave_error = cleave_error_weight * (
            ((stats1 - cleave_data).array().abs() / cleave_denom).mean()
        );
    }
    if (n_unbind_data > 0)
    {
        unbind_error = unbind_error_weight * (
            ((stats2 - unbind_data).array().abs() / unbind_denom).mean()
        );
    }

    return std::make_pair(cleave_error, unbind_error);
}

/**
 * @param cleave_data
 * @param unbind_data
 * @param cleave_seqs
 * @param unbind_seqs
 * @param bind_conc
 * @param error_mode
 * @param cleave_error_weight
 * @param unbind_error_weight
 * @param ninit
 * @param rng
 * @param delta
 * @param beta
 * @param sqp_min_stepsize
 * @param max_iter
 * @param tol
 * @param x_tol
 * @param qp_stepsize_tol
 * @param quasi_newton
 * @param regularize
 * @param regularize_weight
 * @param hessian_modify_max_iter
 * @param c1
 * @param c2
 * @param line_search_max_iter
 * @param zoom_max_iter
 * @param qp_max_iter
 * @param verbose
 * @param search_verbose
 * @param zoom_verbose
 */
std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, 1> >
    fitLineParamsAgainstMeasuredRates(const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data, 
                                      const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data, 
                                      const Ref<const MatrixXi>& cleave_seqs, 
                                      const Ref<const MatrixXi>& unbind_seqs,
                                      const MainType bind_conc, const int error_mode,
                                      const MainType cleave_error_weight,
                                      const MainType unbind_error_weight,
                                      const int ninit, boost::random::mt19937& rng,
                                      const MainType delta, const MainType beta,
                                      const MainType sqp_min_stepsize,
                                      const int max_iter, const MainType tol,
                                      const MainType x_tol,
                                      const MainType qp_stepsize_tol, 
                                      const QuasiNewtonMethod quasi_newton,
                                      const RegularizationMethod regularize,
                                      const MainType regularize_weight, 
                                      const int hessian_modify_max_iter, 
                                      const MainType c1, const MainType c2,
                                      const int line_search_max_iter,
                                      const int zoom_max_iter, const int qp_max_iter,
                                      const bool verbose, const bool search_verbose,
                                      const bool zoom_verbose) 
{
    // Set up an SQPOptimizer instance
    std::string poly_filename = "polytopes/line-2-w4-plusbind.poly"; 
    std::string vert_filename = "polytopes/line-2-w4-plusbind.vert";
    Polytopes::LinearConstraints* constraints = new Polytopes::LinearConstraints(
        Polytopes::InequalityType::GreaterThanOrEqualTo 
    );
    constraints->parse(poly_filename);
    const int D = constraints->getD(); 
    const int N = constraints->getN(); 
    SQPOptimizer<MainType>* opt = new SQPOptimizer<MainType>(constraints);

    // Sample a set of initial parameter points from the given polytope 
    Delaunay_triangulation* tri = new Delaunay_triangulation(D); 
    Matrix<MainType, Dynamic, Dynamic> init_points(ninit, D); 
    Polytopes::parseVerticesFile(vert_filename, tri);
    init_points = Polytopes::sampleFromConvexPolytope<MAIN_PRECISION>(tri, ninit, 0, rng);
    delete tri;

    // Get the vertices of the given polytope and extract the min/max bounds 
    // of each parameter
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = Polytopes::parseVertexCoords(vert_filename); 
    Matrix<MainType, Dynamic, 2> bounds(D, 2); 
    for (int i = 0; i < D; ++i)
    {
        mpq_rational min_param = vertices.col(i).minCoeff();
        mpq_rational max_param = vertices.col(i).maxCoeff();
        bounds(i, 0) = static_cast<MainType>(min_param); 
        bounds(i, 1) = static_cast<MainType>(max_param);
    }
    Matrix<MainType, Dynamic, 1> regularize_bases = (bounds.col(0) + bounds.col(1)) / 2;

    // Define objective function and vector of regularization weights
    std::function<MainType(const Ref<const Matrix<MainType, Dynamic, 1> >&)> func;
    Matrix<MainType, Dynamic, 1> regularize_weights(D);
    //regularize_weights.head(D - 1) = regularize_weight * Matrix<MainType, Dynamic, 1>::Ones(D - 1);
    //regularize_weights(D - 1) = 0;   // No regularization for terminal binding rate
    regularize_weights = regularize_weight * Matrix<MainType, Dynamic, 1>::Ones(D); 
    if (error_mode == 0)   // Mean absolute percentage error 
    {
        func = [
            &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
            &cleave_error_weight, &unbind_error_weight, &bind_conc
        ](const Ref<const Matrix<MainType, Dynamic, 1> >& x) -> MainType
        {
            std::pair<MainType, MainType> error = meanAbsolutePercentageErrorAgainstData(
                x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_error_weight, unbind_error_weight, bind_conc
            );
            return error.first + error.second;
        };
    }
    else if (error_mode == 1)   // Symmetric mean absolute percentage error
    {
        func = [
            &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
            &cleave_error_weight, &unbind_error_weight, &bind_conc
        ](const Ref<const Matrix<MainType, Dynamic, 1> >& x) -> MainType
        {
            std::pair<MainType, MainType> error = symmetricMeanAbsolutePercentageErrorAgainstData(
                x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_error_weight, unbind_error_weight, bind_conc
            );
            return error.first + error.second;
        };
    }
    else if (error_mode == 2)   // Minimum-based mean absolute percentage error
    {
        func = [
            &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
            &cleave_error_weight, &unbind_error_weight, &bind_conc
        ](const Ref<const Matrix<MainType, Dynamic, 1> >& x) -> MainType
        {
            std::pair<MainType, MainType> error = minBasedMeanAbsolutePercentageErrorAgainstData(
                x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_error_weight, unbind_error_weight, bind_conc
            );
            return error.first + error.second;
        };
    }

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<MainType, Dynamic, Dynamic> best_fits(ninit, D); 
    Matrix<MainType, Dynamic, 1> x_init, l_init;
    Matrix<MainType, Dynamic, 1> errors(ninit);
    QuadraticProgramSolveMethod qp_solve_method = USE_CUSTOM_SOLVER;
    for (int i = 0; i < ninit; ++i)
    {
        // Assemble initial parameter values
        x_init = init_points.row(i); 
        l_init = (
            Matrix<MainType, Dynamic, 1>::Ones(N)
            - constraints->active(x_init.cast<mpq_rational>()).template cast<MainType>()
        );

        // Obtain best-fit parameter values from the initial parameters
        best_fits.row(i) = opt->run(
            func, quasi_newton, regularize, regularize_bases, regularize_weights,
            qp_solve_method, x_init, l_init, delta, beta, sqp_min_stepsize,
            max_iter, tol, x_tol, qp_stepsize_tol, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, verbose,
            search_verbose, zoom_verbose
        );
        std::pair<MainType, MainType> error;
        if (error_mode == 0)         // Mean absolute percentage error
        {
            error = meanAbsolutePercentageErrorAgainstData(
                best_fits.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_error_weight, unbind_error_weight, bind_conc
            );
        }
        else if (error_mode == 1)    // Symmetric mean absolute percentage error
        {
            error = symmetricMeanAbsolutePercentageErrorAgainstData(
                best_fits.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_error_weight, unbind_error_weight, bind_conc
            );
        }
        else if (error_mode == 2)    // Minimum-based mean absolute percentage error
        {
            error = minBasedMeanAbsolutePercentageErrorAgainstData(
                best_fits.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                cleave_error_weight, unbind_error_weight, bind_conc
            );
        }
        errors(i) = error.first + error.second;
    }
    delete opt;

    return std::make_pair(best_fits, errors);
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
    MainType cleave_error_weight = 1;
    MainType unbind_error_weight = 1;
    MainType cleave_pseudocount = 0;
    MainType unbind_pseudocount = 0;
    MainType bind_conc = 1e-7;    // TODO Write block for parsing user-defined value 
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
    if (json_data.if_contains("cleave_error_weight"))
    {
        cleave_error_weight = static_cast<MainType>(json_data["cleave_error_weight"].as_double()); 
        if (cleave_error_weight <= 0)
            throw std::runtime_error("Invalid cleavage rate error weight specified"); 
    }
    if (json_data.if_contains("unbind_error_weight"))
    {
        unbind_error_weight = static_cast<MainType>(json_data["unbind_error_weight"].as_double());
        if (unbind_error_weight <= 0)
            throw std::runtime_error("Invalid unbinding rate error weight specified");
    }
    if (json_data.if_contains("cleave_pseudocount"))
    {
        cleave_pseudocount = static_cast<MainType>(json_data["cleave_pseudocount"].as_double()); 
        if (cleave_pseudocount < 0)
            throw std::runtime_error("Invalid cleavage rate pseudocount specified");
    }
    if (json_data.if_contains("unbind_pseudocount"))
    {
        unbind_pseudocount = static_cast<MainType>(json_data["unbind_pseudocount"].as_double()); 
        if (unbind_pseudocount < 0)
            throw std::runtime_error("Invalid unbinding rate pseudocount specified");
    }
    
    // Parse SQP configurations
    MainType delta = 1e-12; 
    MainType beta = 1e-4;
    MainType sqp_min_stepsize = 1e-9;
    int max_iter = 1000; 
    MainType tol = 1e-8;
    MainType x_tol = 1e-8;
    MainType qp_stepsize_tol = 1e-10;
    QuasiNewtonMethod quasi_newton = QuasiNewtonMethod::BFGS;
    RegularizationMethod regularize = RegularizationMethod::L2; 
    MainType regularize_weight = 0.1; 
    int hessian_modify_max_iter = 10000;
    MainType c1 = 1e-4;               // Default value suggested by Nocedal and Wright
    MainType c2 = 0.9;                // Default value suggested by Nocedal and Wright
    int line_search_max_iter = 10;    // Default value in scipy.optimize.line_search()
    int zoom_max_iter = 10;           // Default value in scipy.optimize.line_search()
    int qp_max_iter = 100;
    bool verbose = true;
    bool search_verbose = false;
    bool zoom_verbose = false;
    if (json_data.if_contains("sqp_config"))
    {
        boost::json::object sqp_data = json_data["sqp_config"].as_object(); 
        if (sqp_data.if_contains("delta"))
        {
            delta = static_cast<MainType>(sqp_data["delta"].as_double());
            if (delta <= 0)
                throw std::runtime_error("Invalid value for delta specified"); 
        }
        if (sqp_data.if_contains("beta"))
        {
            beta = static_cast<MainType>(sqp_data["beta"].as_double()); 
            if (beta <= 0)
                throw std::runtime_error("Invalid value for beta specified"); 
        }
        if (sqp_data.if_contains("min_stepsize"))
        {
            sqp_min_stepsize = static_cast<MainType>(sqp_data["min_stepsize"].as_double()); 
            if (sqp_min_stepsize <= 0 || sqp_min_stepsize >= 1)    // Must be less than 1
                throw std::runtime_error("Invalid value for sqp_min_stepsize specified");
        }
        if (sqp_data.if_contains("max_iter"))
        {
            max_iter = sqp_data["max_iter"].as_int64(); 
            if (max_iter <= 0)
                throw std::runtime_error("Invalid value for max_iter specified"); 
        }
        if (sqp_data.if_contains("tol"))
        {
            tol = static_cast<MainType>(sqp_data["tol"].as_double());
            if (tol <= 0)
                throw std::runtime_error("Invalid value for tol specified"); 
        }
        if (sqp_data.if_contains("x_tol"))
        {
            x_tol = static_cast<MainType>(sqp_data["x_tol"].as_double());
            if (x_tol <= 0)
                throw std::runtime_error("Invalid value for x_tol specified"); 
        }
        if (sqp_data.if_contains("qp_stepsize_tol"))
        {
            qp_stepsize_tol = static_cast<MainType>(sqp_data["qp_stepsize_tol"].as_double());
            if (qp_stepsize_tol <= 0)
                throw std::runtime_error("Invalid value for qp_stepsize_tol specified"); 
        }
        if (sqp_data.if_contains("quasi_newton_method"))
        {
            // Check that the value is either 0, 1, 2
            int value = sqp_data["quasi_newton_method"].as_int64(); 
            if (value < 0 || value > 2)
                throw std::runtime_error("Invalid value for quasi_newton_method specified"); 
            quasi_newton = static_cast<QuasiNewtonMethod>(value); 
        }
        if (sqp_data.if_contains("regularization_method"))
        {
            // Check that the value is either 0, 1, 2
            int value = sqp_data["regularization_method"].as_int64(); 
            if (value < 0 || value > 2)
                throw std::runtime_error("Invalid value for regularization_method specified"); 
            regularize = static_cast<RegularizationMethod>(value); 
        }
        if (regularize != 0 && sqp_data.if_contains("regularization_weight"))   // Only check if regularize != 0
        {
            regularize_weight = static_cast<MainType>(sqp_data["regularization_weight"].as_double());
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
        if (sqp_data.if_contains("qp_max_iter"))
        {
            qp_max_iter = sqp_data["qp_max_iter"].as_int64(); 
            if (qp_max_iter <= 0)
                throw std::runtime_error("Invalid value for qp_max_iter specified"); 
        }
        if (sqp_data.if_contains("c1"))
        {
            c1 = static_cast<MainType>(sqp_data["c1"].as_double());
            if (c1 <= 0)
                throw std::runtime_error("Invalid value for c1 specified"); 
        }
        if (sqp_data.if_contains("c2"))
        {
            c2 = static_cast<MainType>(sqp_data["c2"].as_double());
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
    Matrix<MainType, Dynamic, 1> cleave_data = Matrix<MainType, Dynamic, 1>::Zero(0); 
    Matrix<MainType, Dynamic, 1> unbind_data = Matrix<MainType, Dynamic, 1>::Zero(0);
    
    // Parse the input file of (composite) cleavage rates, if one is given 
    std::ifstream infile;
    std::string line, token;
    if (cleave_infilename.size() > 0)
    {
        infile.open(cleave_infilename);

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
                if (token[j] == '0')
                    cleave_seqs(n_cleave_data - 1, j) = 0; 
                else
                    cleave_seqs(n_cleave_data - 1, j) = 1;
            }

            // The second entry is the cleavage rate
            std::getline(ss, token, '\t'); 
            try
            {
                cleave_data(n_cleave_data - 1) = MainType(token);
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
                if (token[j] == '0')
                    unbind_seqs(n_unbind_data - 1, j) = 0; 
                else
                    unbind_seqs(n_unbind_data - 1, j) = 1;
            } 

            // The second entry is the ndABA (specific dissociativity) being parsed 
            std::getline(ss, token, '\t');
            try
            {
                unbind_data(n_unbind_data - 1) = MainType(token);
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

    // Assume that the cleavage rate for the perfect-match substrate is
    // specified first ...
    //
    // ... and thus normalize all composite cleavage rates and invert
    if (cleave_data.size() > 0)
    {
        for (int i = 1; i < cleave_data.size(); ++i)
            cleave_data(i) = cleave_data(i) / cleave_data(0);   // inverse composite cleavage rate ratio 
                                                                // = rate on mismatched / rate on perfect
        cleave_data(0) = 1;
    }

    // Also invert all parsed ndABAs (i.e., specific dissociativities)
    if (unbind_data.size() > 0)
        unbind_data = unbind_data.array().pow(-1).matrix();
    
    // Add pseudocounts to the cleavage rates and unbinding rates 
    cleave_data += cleave_pseudocount * Matrix<MainType, Dynamic, 1>::Ones(n_cleave_data);
    unbind_data += unbind_pseudocount * Matrix<MainType, Dynamic, 1>::Ones(n_unbind_data);

    // Define matrix of single-mismatch DNA sequences relative to the 
    // perfect-match sequence for cleavage rates
    MatrixXi single_mismatch_seqs;
    single_mismatch_seqs = MatrixXi::Ones(length, length) - MatrixXi::Identity(length, length); 

    /** ------------------------------------------------------- //
     *              FIT AGAINST GIVEN KINETIC DATA              //
     *  --------------------------------------------------------*/ 
    std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, 1> > results;
    std::stringstream header_ss; 
    header_ss << "match_forward\tmatch_reverse\tmismatch_forward\tmismatch_reverse\t";
    if (nfolds == 1)
    {
        results = fitLineParamsAgainstMeasuredRates(
            cleave_data, unbind_data, cleave_seqs, unbind_seqs, bind_conc,
            error_mode, cleave_error_weight, unbind_error_weight, ninit, rng,
            delta, beta, sqp_min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
            quasi_newton, regularize, regularize_weight, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, verbose,
            search_verbose, zoom_verbose
        );
        Matrix<MainType, Dynamic, Dynamic> best_fits = results.first;
        Matrix<MainType, Dynamic, 1> errors = results.second;

        // Output the fits to file 
        std::ofstream outfile(outfilename); 
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        outfile << "fit_attempt\t" << header_ss.str()
                << "terminal_unbind_rate\tterminal_cleave_rate\t"
                << "terminal_bind_rate\ttest_error\t";
        for (int i = 0; i < length; ++i)
        {
            outfile << "mm" << i << "_prob\t"
                    << "mm" << i << "_cleave_rate\t"
                    << "mm" << i << "_dead_unbind_rate\t"
                    << "mm" << i << "_live_unbind_rate\t"
                    << "mm" << i << "_spec\t"
                    << "mm" << i << "_rapid\t"
                    << "mm" << i << "_deaddissoc\t"
                    << "mm" << i << "_composite_cleave_rate_ratio\t";
        }
        int pos = outfile.tellp();
        outfile.seekp(pos - 1);
        outfile << std::endl;  
        for (int i = 0; i < ninit; ++i)
        {
            outfile << i << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fits.cols(); ++j)
                outfile << best_fits(i, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors(i) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            Matrix<MainType, Dynamic, 8> fit_single_mismatch_stats = computeCleavageStatsAll(
                best_fits.row(i), single_mismatch_seqs, bind_conc
            );
            for (int j = 0; j < fit_single_mismatch_stats.rows(); ++j)
            {
                for (int k = 0; k < 8; ++k)
                    outfile << fit_single_mismatch_stats(j, k) << '\t';
            }
            pos = outfile.tellp();
            outfile.seekp(pos - 1); 
            outfile << std::endl;
        }
        outfile.close();
    }
    else
    {
        // TODO Remove this portion 
        assert(false);
        /*
        // Shuffle the rows of the datasets to ensure that there are no biases 
        // in sequence composition (only for cross-validation purposes) 
        PermutationMatrix<Dynamic, Dynamic> cleave_perm = getPermutation(n_cleave_data, rng);  
        PermutationMatrix<Dynamic, Dynamic> unbind_perm = getPermutation(n_unbind_data, rng);
        cleave_data = cleave_perm * cleave_data; 
        cleave_seqs = cleave_perm * cleave_seqs; 
        unbind_data = unbind_perm * unbind_data; 
        unbind_seqs = unbind_perm * unbind_seqs;

        // Divide the indices in each dataset into folds
        auto unbind_fold_pairs = getFolds(n_unbind_data, nfolds);
        auto cleave_fold_pairs = getFolds(n_cleave_data, nfolds); 
        
        // For each fold ...
        Matrix<MainType, Dynamic, 1> unbind_data_train, unbind_data_test,
                                     cleave_data_train, cleave_data_test;
        MatrixXi unbind_seqs_train, unbind_seqs_test, cleave_seqs_train, cleave_seqs_test;
        Matrix<MainType, Dynamic, Dynamic> best_fit_total(nfolds, 0); 
        Matrix<MainType, Dynamic, Dynamic> fit_single_mismatch_stats_total(nfolds, 0);
        Matrix<MainType, Dynamic, 1> errors_against_test(nfolds); 
        for (int fi = 0; fi < nfolds; ++fi)
        {
            unbind_data_train = unbind_data(unbind_fold_pairs[fi].first);
            unbind_data_test = unbind_data(unbind_fold_pairs[fi].second);
            unbind_seqs_train = unbind_seqs(unbind_fold_pairs[fi].first, Eigen::all);
            unbind_seqs_test = unbind_seqs(unbind_fold_pairs[fi].second, Eigen::all); 
            cleave_data_train = cleave_data(cleave_fold_pairs[fi].first); 
            cleave_data_test = cleave_data(cleave_fold_pairs[fi].second); 
            cleave_seqs_train = cleave_seqs(cleave_fold_pairs[fi].first, Eigen::all); 
            cleave_seqs_test = cleave_seqs(cleave_fold_pairs[fi].second, Eigen::all);

            // Optimize model parameters on the training subset 
            results = fitLineParamsAgainstMeasuredRates(
                cleave_data_train, unbind_data_train, cleave_seqs_train,
                unbind_seqs_train, bind_conc, error_mode, cleave_error_weight,
                unbind_error_weight, ninit, rng, delta, beta, sqp_min_stepsize,
                max_iter, tol, x_tol, qp_stepsize_tol, quasi_newton, regularize,
                regularize_weight, hessian_modify_max_iter, c1, c2,
                line_search_max_iter, zoom_max_iter, qp_max_iter, verbose,
                search_verbose, zoom_verbose
            );
            Matrix<MainType, Dynamic, Dynamic> best_fit_per_fold = results.first;
            Matrix<MainType, Dynamic, Dynamic> fit_single_mismatch_stats_per_fold = computeCleavageStatsAll(
                best_fit_per_fold, std::get<1>(results); 
            Matrix<MainType, Dynamic, 1> errors_per_fold = std::get<2>(results);
            if (fi == 0)
            {
                best_fit_total.resize(nfolds, best_fit_per_fold.cols());
                fit_single_mismatch_stats_total.resize(nfolds, fit_single_mismatch_stats_per_fold.cols());
            } 

            // Find the parameter vector corresponding to the least error
            Eigen::Index minidx; 
            MainType minerror = errors_per_fold.minCoeff(&minidx); 
            Matrix<MainType, Dynamic, 1> best_fit = best_fit_per_fold.row(minidx); 
            Matrix<MainType, Dynamic, 1> fit_single_mismatch_stats = fit_single_mismatch_stats_per_fold.row(minidx);

            // Evaluate error against the test subset 
            std::pair<MainType, MainType> error_against_test;
            if (error_mode == 0)         // Mean absolute percentage error
            {
                error_against_test = meanAbsolutePercentageErrorAgainstData(
                    best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                    unbind_data_test, cleave_error_weight, unbind_error_weight,
                    bind_conc
                );
            }
            else if (error_mode == 1)    // Symmetric mean absolute percentage error
            {
                error_against_test = symmetricMeanAbsolutePercentageErrorAgainstData(
                    best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                    unbind_data_test, cleave_error_weight, unbind_error_weight,
                    bind_conc
                );
            }
            else if (error_mode == 2)    // Minimum-based mean absolute percentage error
            {
                error_against_test = minBasedMeanAbsolutePercentageErrorAgainstData(
                    best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                    unbind_data_test, cleave_error_weight, unbind_error_weight,
                    bind_conc
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
                << "terminal_unbind_rate\tterminal_cleave_rate\t"
                << "terminal_bind_rate\ttest_error\t";
        for (int i = 0; i < length; ++i)
        {
            outfile << "mm" << i << "_deaddissoc\t"
                    << "mm" << i << "_composite_cleave_rate_ratio\t";
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
        */
    }

    return 0;
} 

