/**
 * Using line-search SQP, identify the set of line graph parameter vectors
 * that yields each given set of specific dissociativities and (composite)
 * cleavage rates in the given data files.
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
 *     2/8/2023
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
using boost::multiprecision::ceil;
using boost::multiprecision::floor;
constexpr int MAIN_PRECISION = 30; 
typedef number<mpfr_float_backend<MAIN_PRECISION> > MainType;
typedef number<mpfr_float_backend<100> >            PreciseType;
const int length = 20;
const MainType ten_main(10);
const PreciseType ten_precise(10);

template <typename Derived>
typename Derived::Scalar logsumexp(const MatrixBase<Derived>& logx,
                                   const typename Derived::Scalar base)
{
    typedef typename Derived::Scalar T; 
    T maxlogx = logx.maxCoeff();
    Matrix<T, Dynamic, 1> x(logx.size());
    for (int i = 0; i < logx.size(); ++i)
        x(i) = pow(base, logx(i) - maxlogx); 
    
    return maxlogx + log(x.sum()) / log(base);  
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
 * Compute the specific dissociativity on all given matched/mismatched sequences
 * specified in the given matrix of complementarity patterns, for the given LG.
 *
 * Here, `logrates` is assumed to contain *5* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate.
 *
 * @param logrates  Input vector of 5 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (match w.r.t.
 *                  perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                  sequence).
 */
Matrix<MainType, Dynamic, 1> computeDissociativity(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                   const Ref<const MatrixXi>& seqs)
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

    // Terminal unbinding rate 
    PreciseType terminal_unbind_rate = pow(ten_precise, static_cast<PreciseType>(logrates(4)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute dead unbinding rate against perfect-match substrate
    PreciseType unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate);

    // Compute dead unbinding rate against each given mismatched substrate
    Matrix<PreciseType, Dynamic, 1> stats(seqs.rows());  
    for (int i = 0; i < seqs.rows(); ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            if (seqs(i, j))
                model->setEdgeLabels(j, match_rates);
            else
                model->setEdgeLabels(j, mismatch_rates);
        }
        
        // Compute *inverse* specific dissociativity: rate on perfect / rate on mismatched
        PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate);
        stats(i) = log10(unbind_rate_perfect) - log10(unbind_rate);
    }

    delete model;
    return stats.template cast<MainType>();
}

/**
 * Compute the composite cleavage rate ratios on all given matched/mismatched
 * sequences specified in the given matrix of complementarity patterns, for
 * the given LG.
 *
 * Here, `logrates` is assumed to contain *7* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate,
 * 7) terminal binding rate (units of inverse (M * sec)).
 *
 * @param logrates  Input vector of 7 LGPs.
 * @param seqs      Matrix of input sequences, with entries of 0 (match w.r.t.
 *                  perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                  sequence).
 * @param bind_conc Concentration of available Cas9. 
 */
Matrix<MainType, Dynamic, 1> computeCleavageRateRatios(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
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
  
    // Compute composite cleavage timescale against perfect-match substrate
    PreciseType composite_cleave_time_perfect = model->getEntryToUpperExitTime(
        bind_rate, terminal_unbind_rate, terminal_cleave_rate
    );

    // Compute composite cleavage timescale against each given mismatched substrate
    Matrix<PreciseType, Dynamic, 1> stats(seqs.rows());  
    for (int i = 0; i < seqs.rows(); ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            if (seqs(i, j))
                model->setEdgeLabels(j, match_rates);
            else
                model->setEdgeLabels(j, mismatch_rates);
        }
        PreciseType composite_cleave_time = model->getEntryToUpperExitTime(
            bind_rate, terminal_unbind_rate, terminal_cleave_rate
        );
        
        // Compute *inverse* composite cleavage rate ratio: rate on mismatched / rate on perfect,
        // or time on perfect / time on mismatched
        stats(i) = log10(composite_cleave_time_perfect) - log10(composite_cleave_time);
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
Matrix<MainType, Dynamic, 8> computeCleavageStats(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
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
    PreciseType composite_cleave_time_perfect = model->getEntryToUpperExitTime(
        bind_rate, terminal_unbind_rate, terminal_cleave_rate
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
        PreciseType composite_cleave_time = model->getEntryToUpperExitTime(
            bind_rate, terminal_unbind_rate, terminal_cleave_rate
        );
        stats(i, 0) = prob; 
        stats(i, 1) = cleave_rate; 
        stats(i, 2) = dead_unbind_rate; 
        stats(i, 3) = composite_cleave_time;

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
 * Compute the *unregularized* sum-of-squares error between a set of specific
 * dissociativities inferred from the given LGPs and a set of experimentally
 * determined dissociativities. 
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates    Input vector of 5 LGPs.
 * @param unbind_seqs Matrix of input sequences, with entries of 0 (match w.r.t.
 *                    perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                    sequence).
 * @param unbind_data Matrix of measured specific dissociativities for the given
 *                    input sequences.
 * @returns Vector of residuals yielding the sum-of-squares errors. 
 */
Matrix<MainType, Dynamic, 1> dissocErrorAgainstData(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                    const Ref<const MatrixXi>& unbind_seqs,
                                                    const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data)
{
    // Compute dissociativities for the given complementarity patterns
    Matrix<MainType, Dynamic, 1> stats = computeDissociativity(logrates, unbind_seqs);
    for (int i = 0; i < stats.size(); ++i)    // Convert to linear scale 
        stats(i) = pow(ten_main, stats(i)); 

    // Compute each residual contributing to overall sum-of-squares error
    Matrix<MainType, Dynamic, 1> residuals = (stats - unbind_data).array().pow(2); 

    return residuals;
}

/**
 * Compute the *unregularized* sum-of-squares error between a set of composite
 * cleavage rate ratios inferred from the given LGPs and a set of experimentally
 * determined composite cleavage rates. 
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates    Input vector of 7 LGPs.
 * @param cleave_seqs Matrix of input sequences, with entries of 0 (match w.r.t.
 *                    perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                    sequence).
 * @param cleave_data Matrix of measured composite cleavage rate ratios for 
 *                    the given input sequences.
 * @param bind_conc   Concentration of available Cas9.
 * @returns Vector of residuals yielding the sum-of-squares errors. 
 */
Matrix<MainType, Dynamic, 1> cleaveErrorAgainstData(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                    const Ref<const MatrixXi>& cleave_seqs,
                                                    const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data,
                                                    const MainType bind_conc)
{
    // Compute dissociativities for the given complementarity patterns
    Matrix<MainType, Dynamic, 1> stats = computeCleavageRateRatios(logrates, cleave_seqs, bind_conc);
    for (int i = 0; i < stats.size(); ++i)    // Convert to linear scale 
        stats(i) = pow(ten_main, stats(i)); 

    // Compute each residual contributing to overall sum-of-squares error
    Matrix<MainType, Dynamic, 1> residuals = (stats - cleave_data).array().pow(2); 

    return residuals;
}

/**
 * @param constraints             Constraints defining input polytope.
 * @param vertices                Vertex coordinates of input polytope.
 * @param unbind_data             Matrix of measured specific dissociativities.
 * @param unbind_seqs             Matrix of input sequences. 
 * @param ninit                   Number of fitting attempts. 
 * @param rng                     Random number generator. 
 * @param delta                   Increment for finite-differences 
 *                                approximation during each SQP iteration.
 * @param beta                    Increment for Hessian matrix modification
 *                                (for ensuring positive semi-definiteness).
 * @param sqp_min_stepsize        Minimum allowed stepsize during each
 *                                SQP iteration. 
 * @param max_iter                Maximum number of iterations for SQP. 
 * @param tol                     Tolerance for assessing convergence in 
 *                                output value in SQP. 
 * @param x_tol                   Tolerance for assessing convergence in 
 *                                input value in SQP.
 * @param qp_stepsize_tol         Tolerance for assessing whether a
 *                                stepsize during each QP is zero (during 
 *                                each SQP iteration).
 * @param quasi_newton
 * @param regularize              Regularization method: `NOREG`, `L1`,
 *                                or `L2`.
 * @param regularize_weight       Regularization weight. If `regularize`
 *                                is `NOREG`, then this value is ignored.
 * @param hessian_modify_max_iter Maximum number of Hessian matrix
 *                                modification iterations (for ensuring
 *                                positive semi-definiteness).  
 * @param c1                      Pre-factor for testing Armijo's 
 *                                condition during each SQP iteration.
 * @param c2                      Pre-factor for testing the curvature 
 *                                condition during each SQP iteration.
 * @param line_search_max_iter    Maximum number of line search iterations
 *                                during each SQP iteration.
 * @param zoom_max_iter           Maximum number of zoom iterations
 *                                during each SQP iteration.
 * @param qp_max_iter             Maximum number of iterations during 
 *                                each QP (during each SQP iteration). 
 * @param verbose                 If true, output intermittent messages
 *                                during SQP to `stdout`.
 * @param search_verbose          If true, output intermittent messages
 *                                in `lineSearch()` during SQP to `stdout`.
 * @param zoom_verbose            If true, output intermittent messages
 *                                in `zoom()` during SQP to `stdout`.
 */
std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, Dynamic> >
    fitLineParamsAgainstMeasuredRatesDissoc(Polytopes::LinearConstraints* constraints,
                                            const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices,  
                                            const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data, 
                                            const Ref<const MatrixXi>& unbind_seqs,
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
    const int N = constraints->getN();
    const int D = 4;    // b, d, b', d'
    SQPOptimizer<MainType>* opt = new SQPOptimizer<MainType>(constraints);

    // Sample a set of initial parameter points from the given polytope 
    Delaunay_triangulation* tri = new Delaunay_triangulation(D); 
    Matrix<MainType, Dynamic, Dynamic> init_points(ninit, D); 
    Polytopes::triangulate(vertices, tri);
    init_points = Polytopes::sampleFromConvexPolytope<MAIN_PRECISION>(tri, ninit, 0, rng);
    delete tri;

    // Get the vertices of the given polytope and extract the min/max bounds 
    // of each parameter
    Matrix<MainType, Dynamic, 2> bounds(D, 2); 
    for (int i = 0; i < D; ++i)
    {
        mpq_rational min_param = vertices.col(i).minCoeff();
        mpq_rational max_param = vertices.col(i).maxCoeff();
        bounds(i, 0) = static_cast<MainType>(min_param); 
        bounds(i, 1) = static_cast<MainType>(max_param);
    }
    Matrix<MainType, Dynamic, 1> regularize_bases = (bounds.col(0) + bounds.col(1)) / 2;

    // Define vector of regularization weights
    Matrix<MainType, Dynamic, 1> regularize_weights(D);
    regularize_weights = regularize_weight * Matrix<MainType, Dynamic, 1>::Ones(D);

    // Define error function to be minimized 
    std::function<MainType(const Ref<const Matrix<MainType, Dynamic, 1> >&)> func = 
        [&unbind_seqs, &unbind_data](const Ref<const Matrix<MainType, Dynamic, 1> >& x) -> MainType
        {
            Matrix<MainType, Dynamic, 1> y(7); 
            y.head(4) = x; 
            y.tail(3) = Matrix<MainType, Dynamic, 1>::Zero(3);
            Matrix<MainType, Dynamic, 1> error = dissocErrorAgainstData(y, unbind_seqs, unbind_data);
            return error.sum();
        };

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<MainType, Dynamic, Dynamic> best_fits(ninit, D);
    Matrix<MainType, Dynamic, Dynamic> residuals(ninit, unbind_seqs.rows()); 
    Matrix<MainType, Dynamic, 1> x_init, l_init;
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
        Matrix<MainType, Dynamic, 1> y(7); 
        y.head(4) = best_fits.row(i); 
        y.tail(3) = Matrix<MainType, Dynamic, 1>::Zero(3); 
        residuals.row(i) = dissocErrorAgainstData(y, unbind_seqs, unbind_data);
    }
    delete opt;

    return std::make_pair(best_fits, residuals);
}

/**
 * @param fit_logrates
 * @param constraints
 * @param bounds
 * @param cleave_data
 * @param cleave_seqs
 * @param bind_conc
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
std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, Dynamic> >
    fitLineParamsAgainstMeasuredRatesCleave(const Ref<const Matrix<MainType, Dynamic, 1> >& fit_logrates,
                                            Polytopes::LinearConstraints* constraints,
                                            const Ref<const Matrix<mpq_rational, Dynamic, 2> >& bounds, 
                                            const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data, 
                                            const Ref<const MatrixXi>& cleave_seqs, 
                                            const MainType bind_conc, const int ninit,
                                            boost::random::mt19937& rng,
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
    const int N = constraints->getN();
    const int D = constraints->getD();
    SQPOptimizer<MainType>* opt = new SQPOptimizer<MainType>(constraints);

    // Sample a set of initial parameter points from the given polytope
    Matrix<MainType, Dynamic, Dynamic> init_points(ninit, D);
    boost::random::uniform_01<double> dist; 
    int nsample = 0;
    while (nsample < ninit)
    {
        // Start with a parameter point satisfying the given parametric bounds
        // and with the four weights set to zero 
        Matrix<mpq_rational, Dynamic, 1> p = Matrix<mpq_rational, Dynamic, 1>::Zero(D);
        for (int j = 4; j < D; ++j)
        {
            mpq_rational min = bounds(j, 0); 
            mpq_rational max = bounds(j, 1);
            p(j) = min + (max - min) * static_cast<mpq_rational>(dist(rng));
        }
        if (constraints->query(p))
        {
            init_points.row(nsample) = p.cast<MainType>();
            nsample++;
        }
    }

    // Define vector of regularization weights
    Matrix<MainType, Dynamic, 1> regularize_bases = (bounds.col(0).cast<MainType>() + bounds.col(1).cast<MainType>()) / 2;
    Matrix<MainType, Dynamic, 1> regularize_weights(D);
    regularize_weights = regularize_weight * Matrix<MainType, Dynamic, 1>::Ones(D);

    // Define error function to be minimized 
    std::function<MainType(const Ref<const Matrix<MainType, Dynamic, 1> >&)> func = 
        [&fit_logrates, &cleave_seqs, &cleave_data, &bind_conc](
            const Ref<const Matrix<MainType, Dynamic, 1> >& x
        ) -> MainType
        {
            // b, d, b', d' are assumed to have been already fit
            Matrix<MainType, Dynamic, 1> y(7); 
            //y.head(4) = fit_logrates + x.head(4);
            y.head(2) = fit_logrates.head(2) + x(0) * Matrix<MainType, Dynamic, 1>::Ones(2);
            y(Eigen::seqN(2, 2)) = fit_logrates.tail(2) + x(1) * Matrix<MainType, Dynamic, 1>::Ones(2);
            y.tail(3) = x.tail(3);
            Matrix<MainType, Dynamic, 1> error = cleaveErrorAgainstData(
                y, cleave_seqs, cleave_data, bind_conc
            );
            return error.sum();
        };

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<MainType, Dynamic, Dynamic> best_fits(ninit, D);
    Matrix<MainType, Dynamic, Dynamic> residuals(ninit, cleave_seqs.rows()); 
    Matrix<MainType, Dynamic, 1> x_init, l_init;
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

        // Re-obtain residuals for the best-fit parameter values
        Matrix<MainType, Dynamic, 1> y(7);
        //y.head(4) = fit_logrates + best_fits.row(i).head(4);
        y.head(2) = fit_logrates.head(2) + best_fits(i, 0) * Matrix<MainType, Dynamic, 1>::Ones(2);
        y(Eigen::seqN(2, 2)) = fit_logrates.tail(2) + best_fits(i, 1) * Matrix<MainType, Dynamic, 1>::Ones(2);
        y.tail(3) = best_fits.row(i).tail(3); 
        residuals.row(i) = cleaveErrorAgainstData(y, cleave_seqs, cleave_data, bind_conc);
    }
    delete opt;

    return std::make_pair(best_fits, residuals);
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);

    /** ------------------------------------------------------- //
     *       DEFINE POLYTOPE FOR DETERMINING b, d', b', d'      //
     *  ------------------------------------------------------- */
    std::string poly_filename = "polytopes/line_3_Rloop.poly";
    std::string vert_filename = "polytopes/line_3_Rloop.vert";
    Polytopes::LinearConstraints* constraints_1 = new Polytopes::LinearConstraints(Polytopes::InequalityType::GreaterThanOrEqualTo);
    constraints_1->parse(poly_filename);
    Matrix<mpq_rational, Dynamic, Dynamic> vertices_1 = Polytopes::parseVertexCoords(vert_filename);   

    /** ------------------------------------------------------- //
     *                    PARSE CONFIGURATIONS                  //
     *  ------------------------------------------------------- */ 
    boost::json::object json_data = parseConfigFile(argv[1]).as_object();

    // Check that input/output file paths were specified 
    if (!json_data.if_contains("cleave_data_filename") && !json_data.if_contains("unbind_data_filename"))
        throw std::runtime_error("At least one dataset must be specified");
    else if (!json_data.if_contains("cleave_data_filename"))
        json_data["cleave_data_filename"] = ""; 
    else if (!json_data.if_contains("unbind_data_filename"))
        json_data["unbind_data_filename"] = "";
    if (!json_data.if_contains("output_prefix"))
        throw std::runtime_error("Output file prefix must be specified");
    std::string cleave_infilename = json_data["cleave_data_filename"].as_string().c_str();
    std::string unbind_infilename = json_data["unbind_data_filename"].as_string().c_str();
    std::string outprefix = json_data["output_prefix"].as_string().c_str();
    std::string outfilename = outprefix + "-main.tsv"; 
    std::string residuals_filename = outprefix + "-residuals.tsv";

    // Assign default values for parameters that were not specified 
    bool data_specified_as_times = false;
    int ninit = 100; 
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

    /** -------------------------------------------------------------- //
     *              FIT AGAINST GIVEN DISSOCIATIVITY DATA              //
     *  -------------------------------------------------------------- */ 
    std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, Dynamic> > results_dissoc;
    results_dissoc = fitLineParamsAgainstMeasuredRatesDissoc(
        constraints_1, vertices_1, unbind_data, unbind_seqs, ninit, rng,
        delta, beta, sqp_min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        quasi_newton, regularize, regularize_weight, hessian_modify_max_iter,
        c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, verbose,
        search_verbose, zoom_verbose
    );
    Matrix<MainType, Dynamic, Dynamic> best_fits_dissoc = results_dissoc.first;
    Matrix<MainType, Dynamic, Dynamic> residuals_dissoc = results_dissoc.second;
    Matrix<MainType, Dynamic, 1> errors_dissoc = residuals_dissoc.rowwise().sum();

    // Find the parameter vector with the least sum-of-squares error 
    Eigen::Index min_idx; 
    errors_dissoc.minCoeff(&min_idx); 
    Matrix<MainType, Dynamic, 1> fit_logrates = best_fits_dissoc.row(min_idx);
    Matrix<MainType, Dynamic, 1> fit_residuals = residuals_dissoc.row(min_idx);
    MainType fit_error = errors_dissoc(min_idx);
    std::cout << "------------------------------------------------------\n";
    std::cout << "R-loop rates after dissoc fit: "
              << fit_logrates(0) << " "
              << fit_logrates(1) << " "
              << fit_logrates(2) << " "
              << fit_logrates(3) << std::endl;
    std::cout << "------------------------------------------------------\n";

    /** ------------------------------------------------------- //
     *       DEFINE POLYTOPE FOR DETERMINING TERMINAL RATES     //
     *  ------------------------------------------------------- */
    //const int D = 7;
    const int D = 5; 

    // Weight parameters have pre-determined range 
    MainType weight_min = -3;
    MainType weight_max = 3;

    // Terminal cleavage and binding rates have pre-determined ranges
    MainType terminal_cleave_lograte_min = -4;
    MainType terminal_cleave_lograte_max = 2;
    MainType terminal_bind_lograte_min = 3;
    MainType terminal_bind_lograte_max = 9;

    // Get log(b/d)
    MainType log_b_by_d = fit_logrates(0) - fit_logrates(1); 

    // The terminal unbinding rate should roughly range between:
    //
    // 1e-10 * (10^terminal_bind_lograte_min) * (1 + (b/d) + ... + (b/d)^length) and 
    //
    // 1e-7 * (10^terminal_bind_lograte_max) * (1 + (b/d) + ... + (b/d)^length)
    //
    // for a micro/nanomolar dissociation constant on the order of 1e-10 to 1e-7 M
    Matrix<MainType, Dynamic, 1> arr(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        arr(i) = i * log_b_by_d;
    MainType factor = logsumexp(arr, ten_main);
    MainType terminal_unbind_lograte_min = -10 + terminal_bind_lograte_min + factor;
    MainType terminal_unbind_lograte_max = -7 + terminal_bind_lograte_max + factor;

    // Round to nearest integers
    terminal_unbind_lograte_min = floor(terminal_unbind_lograte_min);
    terminal_unbind_lograte_max = ceil(terminal_unbind_lograte_max);

    std::cout << "Terminal parameter bounds: "
              << weight_min << " " << weight_max << " "
              << weight_min << " " << weight_max << " "
              << terminal_unbind_lograte_min << " "
              << terminal_unbind_lograte_max << " "
              << terminal_cleave_lograte_min << " "
              << terminal_cleave_lograte_max << " "
              << terminal_bind_lograte_min << " "
              << terminal_bind_lograte_max << std::endl; 
    std::cout << "------------------------------------------------------\n";

    Matrix<mpq_rational, Dynamic, 2> param_bounds(D, 2); 
    param_bounds << static_cast<mpq_rational>(weight_min),
                    static_cast<mpq_rational>(weight_max),
                    static_cast<mpq_rational>(weight_min),
                    static_cast<mpq_rational>(weight_max),
                    static_cast<mpq_rational>(terminal_unbind_lograte_min),
                    static_cast<mpq_rational>(terminal_unbind_lograte_max),
                    static_cast<mpq_rational>(terminal_cleave_lograte_min),
                    static_cast<mpq_rational>(terminal_cleave_lograte_max),
                    static_cast<mpq_rational>(terminal_bind_lograte_min),
                    static_cast<mpq_rational>(terminal_bind_lograte_max);
    Matrix<mpq_rational, Dynamic, Dynamic> A = Matrix<mpq_rational, Dynamic, Dynamic>::Zero(2 * D + 2, D); 
    Matrix<mpq_rational, Dynamic, 1> b(2 * D + 2);
    for (int i = 0; i < 2 * D; ++i)
    {
        int j = static_cast<int>(std::floor(i / 2)); 
        if (i % 2 == 0)
        {
            A(i, j) = 1;
            b(i) = param_bounds(j, 0);
        }
        else 
        {
            A(i, j) = -1;
            b(i) = -param_bounds(j, 1);
        }
    }
    A(2 * D, 0) = 1;         // weight for b/d - weight for b'/d' >= fit_logrates(2) - fit_logrates(0)
    A(2 * D, 1) = -1;
    A(2 * D + 1, 0) = -1;    // weight for b'/d' - weight for b/d >= fit_logrates(1) - fit_logrates(3)
    A(2 * D + 1, 1) = 1;
    b(2 * D) = static_cast<mpq_rational>(fit_logrates(2)) - static_cast<mpq_rational>(fit_logrates(0));
    b(2 * D + 1) = static_cast<mpq_rational>(fit_logrates(1)) - static_cast<mpq_rational>(fit_logrates(3));
    Polytopes::LinearConstraints* constraints_2 = new Polytopes::LinearConstraints(
        Polytopes::InequalityType::GreaterThanOrEqualTo, A, b
    );

    /** -------------------------------------------------------------- //
     *              FIT AGAINST GIVEN CLEAVAGE RATE DATA               //
     *  -------------------------------------------------------------- */ 
    std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, Dynamic> > results_cleave;
    results_cleave = fitLineParamsAgainstMeasuredRatesCleave(
        fit_logrates, constraints_2, param_bounds, cleave_data, cleave_seqs,
        bind_conc, ninit, rng, delta, beta, sqp_min_stepsize, max_iter, tol,
        x_tol, qp_stepsize_tol, quasi_newton, regularize, regularize_weight,
        hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
        qp_max_iter, verbose, search_verbose, zoom_verbose
    );
    Matrix<MainType, Dynamic, Dynamic> best_fits_cleave = results_cleave.first;
    Matrix<MainType, Dynamic, Dynamic> residuals_cleave = results_cleave.second;
    Matrix<MainType, Dynamic, 1> errors_cleave = residuals_cleave.rowwise().sum();

    /** -------------------------------------------------------------- //
     *                        OUTPUT FITS TO FILE                      //
     *  -------------------------------------------------------------- */ 
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    outfile << "fit_attempt\tmatch_forward\tmatch_reverse\tmismatch_forward\t"
            << "mismatch_reverse\tterminal_unbind_rate\tterminal_cleave_rate\t"
            << "terminal_bind_rate\tdissoc_error\tcleave_error\t";
    for (int i = 0; i < length; ++i)
    {
        outfile << "mm" << i << "_prob\t"
                << "mm" << i << "_cleave\t"
                << "mm" << i << "_unbind\t"
                << "mm" << i << "_compcleave\t"
                << "mm" << i << "_spec\t"
                << "mm" << i << "_rapid\t"
                << "mm" << i << "_deaddissoc\t"
                << "mm" << i << "_ccratio\t";
    }
    int pos = outfile.tellp();
    outfile.seekp(pos - 1);
    outfile << std::endl;  
    for (int i = 0; i < ninit; ++i)
    {
        outfile << i << '\t'; 

        // Write each best-fit parameter vector ...
        for (int j = 0; j < 2; ++j)
            outfile << fit_logrates(j) + best_fits_cleave(i, 0) << '\t';
        for (int j = 2; j < 4; ++j)
            outfile << fit_logrates(j) + best_fits_cleave(i, 1) << '\t';
        for (int j = 2; j < 5; ++j)
            outfile << best_fits_cleave(i, j) << '\t';

        // ... along with the associated error against the corresponding data ... 
        outfile << errors_dissoc(i) << '\t' << errors_cleave(i) << '\t';

        // ... along with the associated single-mismatch cleavage statistics
        Matrix<MainType, Dynamic, 1> y(7);
        y.head(2) = fit_logrates.head(2) + best_fits_cleave(i, 0) * Matrix<MainType, Dynamic, 1>::Ones(2);
        y(Eigen::seqN(2, 2)) = fit_logrates.tail(2) + best_fits_cleave(i, 1) * Matrix<MainType, Dynamic, 1>::Ones(2);
        y.tail(3) = best_fits_cleave.row(i).tail(3);
        Matrix<MainType, Dynamic, 8> fit_single_mismatch_stats = computeCleavageStats(
            y, single_mismatch_seqs, bind_conc
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

    // Output the optimal dissociativity residuals and cleavage rate ratio 
    // residuals to file 
    std::ofstream residuals_outfile(residuals_filename);
    residuals_outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    residuals_outfile << "seq\tfit_dissoc\t";
    for (int i = 0; i < ninit - 1; ++i)
        residuals_outfile << "fit_" << i << '\t'; 
    residuals_outfile << "fit_" << ninit - 1 << std::endl;
    residuals_outfile << "total\t" << fit_error << '\t';
    for (int i = 0; i < ninit - 1; ++i)
        residuals_outfile << residuals_cleave.row(i).sum() << '\t';
    residuals_outfile << residuals_cleave.row(ninit - 1).sum() << std::endl;
    for (int i = 0; i < residuals_cleave.cols(); ++i)
    {
        // Output the sequence ... 
        for (int j = 0; j < length; ++j)
            residuals_outfile << (cleave_seqs(i, j) ? '1' : '0');
        residuals_outfile << '\t';

        // ... and each corresponding dissociativity residual ...
        residuals_outfile << fit_residuals(i) << '\t';

        // ... and each vector of cleavage rate residuals 
        for (int j = 0; j < residuals_cleave.rows() - 1; ++j)
            residuals_outfile << residuals_cleave(j, i) << '\t';
        residuals_outfile << residuals_cleave(residuals_cleave.rows() - 1, i) << std::endl;
    }
    residuals_outfile.close();

    delete constraints_1;
    delete constraints_2;
    return 0;
} 

