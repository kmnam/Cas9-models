/**
 * Using line-search SQP, identify the set of line graph parameter vectors
 * that yields each given set of overall cleavage rates in the given data
 * files.  
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
 *     3/29/2023
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
#include <SQP1D.hpp>
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
 * Compute the overall cleavage rate determined by the given LGPs on the 
 * perfect-match substrate.
 *
 * @param logrates    Input vector of 7 LGPs.
 * @param bind_conc   Concentration of available Cas9.
 * @returns Overall cleavage rate.
 */
MainType computePerfectOverallCleaveRate(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
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
  
    // Compute overall cleavage timescale against perfect-match substrate
    PreciseType overall_cleave_time_perfect = model->getEntryToUpperExitTime(
        bind_rate, terminal_unbind_rate, terminal_cleave_rate
    );

    delete model;
    return static_cast<MainType>(pow(overall_cleave_time_perfect, -1));
}

/**
 * Compute the overall cleavage rate ratios on all matched/mismatched
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
 * @returns Vector of overall cleavage rate ratios. 
 */
Matrix<MainType, Dynamic, 1> computeOverallCleavageRateRatios(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
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
  
    // Compute overall cleavage timescale against perfect-match substrate
    PreciseType overall_cleave_time_perfect = model->getEntryToUpperExitTime(
        bind_rate, terminal_unbind_rate, terminal_cleave_rate
    );

    // Compute overall cleavage timescale against each given mismatched substrate
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
        PreciseType overall_cleave_time = model->getEntryToUpperExitTime(
            bind_rate, terminal_unbind_rate, terminal_cleave_rate
        );
        
        // Compute *inverse* overall cleavage rate ratio: rate on mismatched / rate on perfect,
        // or time on perfect / time on mismatched
        stats(i) = log10(overall_cleave_time_perfect) - log10(overall_cleave_time);
    }

    delete model;
    return stats.template cast<MainType>();
}

/**
 * Compute the cleavage probability, cleavage rate, dead unbinding rate, 
 * overall cleavage rate, and all associated normalized statistics on all
 * specified complementarity patterns for the given LG.
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
 * @returns Matrix of cleavage probabilities, cleavage rates, dead unbinding
 *          rates, overall cleavage rates, and all associated normalized
 *          statistics on the given complementarity patterns.  
 */
Matrix<MainType, Dynamic, 7> computeCleavageStats(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
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
  
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and 
    // overall cleavage timescale against perfect-match substrate 
    PreciseType prob_perfect = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
    PreciseType cleave_rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    PreciseType dead_unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate);
    PreciseType overall_cleave_time_perfect = model->getEntryToUpperExitTime(
        bind_rate, terminal_unbind_rate, terminal_cleave_rate
    );

    // Compute cleavage probability, cleavage rate, dead unbinding rate, and
    // overall cleavage timescale against each given mismatched substrate
    Matrix<PreciseType, Dynamic, 7> stats(seqs.rows(), 7);  
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
        PreciseType overall_cleave_time = model->getEntryToUpperExitTime(
            bind_rate, terminal_unbind_rate, terminal_cleave_rate
        );
        stats(i, 0) = prob; 
        stats(i, 1) = cleave_rate; 
        stats(i, 2) = dead_unbind_rate; 
        stats(i, 3) = overall_cleave_time;

        // Compute cleavage specificity: prob on perfect / prob on mismatched
        stats(i, 4) = log10(prob_perfect) - log10(prob);

        // Compute specific rapidity: rate on perfect / rate on mismatched 
        stats(i, 5) = log10(cleave_rate_perfect) - log10(cleave_rate);  

        // Compute overall cleavage rate ratio: rate on perfect / rate on mismatched
        // or time on mismatched / time on perfect
        stats(i, 6) = log10(overall_cleave_time) - log10(overall_cleave_time_perfect);
    }

    delete model;
    return stats.template cast<MainType>();
}

/**
 * Compute the *unregularized* sum-of-squares error between a set of overall
 * cleavage rate ratios inferred from the given LGPs and a set of experimentally
 * determined overall cleavage rates. 
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates    Input vector of 7 LGPs.
 * @param cleave_seqs Matrix of input sequences, with entries of 0 (match w.r.t.
 *                    perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                    sequence).
 * @param cleave_data Matrix of measured overall cleavage rate ratios for 
 *                    the given input sequences.
 * @param bind_conc   Concentration of available Cas9.
 * @returns Vector of residuals yielding the sum-of-squares errors. 
 */
Matrix<MainType, Dynamic, 1> cleaveErrorAgainstData(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                    const Ref<const MatrixXi>& cleave_seqs,
                                                    const Ref<const Matrix<MainType, Dynamic, 1> >& cleave_data,
                                                    const MainType bind_conc)
{
    // Compute overall cleavage rate ratios for the given complementarity patterns
    Matrix<MainType, Dynamic, 1> stats = computeOverallCleavageRateRatios(logrates, cleave_seqs, bind_conc);
    for (int i = 0; i < stats.size(); ++i)    // Convert to linear scale 
        stats(i) = pow(ten_main, stats(i));

    // Compute each residual contributing to overall sum-of-squares error
    Matrix<MainType, Dynamic, 1> residuals = (stats - cleave_data).array().pow(2); 

    return residuals;
}

/**
 * Compute the *unregularized* sum-of-squares error between the overall 
 * cleavage rate inferred from the given LGPs *on the perfect-match substrate*
 * and an experimentally determined overall cleavage rate. 
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates    Input vector of 7 LGPs.
 * @param cleave_rate Measured overall cleavage rate on perfect-match substrate.
 * @param bind_conc   Concentration of available Cas9.
 * @returns Squared error between the LG-derived and measured overall cleavage
 *          rate values.
 */
MainType cleaveErrorAgainstPerfectOverallRate(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                              const MainType cleave_rate,
                                              const MainType bind_conc)
{
    // Compute squared error between model-derived and measured values
    return pow(computePerfectOverallCleaveRate(logrates, bind_conc) - cleave_rate, 2);
}

/**
 * @param constraints             Constraints defining input polytope.
 * @param vertices                Vertex coordinates of input polytope.
 * @param cleave_data             Matrix of measured overall cleavage
 *                                rate ratios.
 * @param cleave_seqs             Matrix of input sequences.
 * @param bind_conc               Concentration of available Cas9.
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
    fitLineParamsAgainstMeasuredRatesCleave(Polytopes::LinearConstraints* constraints,
                                            const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices,  
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
        [&cleave_seqs, &cleave_data, &bind_conc](
            const Ref<const Matrix<MainType, Dynamic, 1> >& x
        ) -> MainType
        {
            Matrix<MainType, Dynamic, 1> error = cleaveErrorAgainstData(
                x, cleave_seqs, cleave_data, bind_conc
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
        residuals.row(i) = cleaveErrorAgainstData(best_fits.row(i), cleave_seqs, cleave_data, bind_conc);
    }
    delete opt;

    return std::make_pair(best_fits, residuals);
}

/**
 * @param logrates                Input vector of LGPs.
 * @param constraints             Constraints defining input polytope.
 * @param cleave_rate             Measured overall cleavage rate on perfect-
 *                                match substrate.
 * @param bind_conc               Concentration of available Cas9.
 * @param ninit                   Number of fitting attempts. 
 * @param rng                     Random number generator. 
 * @param delta                   Increment for finite-differences 
 *                                approximation during each SQP iteration.
 * @param beta                    Increment for Hessian matrix modification
 *                                (for ensuring positive semi-definiteness).
 * @param sqp_min_stepsize        Minimum allowed stepsize during each
 *                                SQP iteration. 
 * @param scan_max_iter           Maximum number of iterations for SQP. 
 * @param tol                     Tolerance for assessing convergence in 
 *                                output value in SQP. 
 * @param x_tol                   Tolerance for assessing convergence in 
 *                                input value in SQP.
 * @param qp_stepsize_tol         Tolerance for assessing whether a
 *                                stepsize during each QP is zero (during 
 *                                each SQP iteration).
 * @param quasi_newton
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
std::pair<MainType, MainType> scanMainChord(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                            Polytopes::LinearConstraints* constraints,
                                            const MainType cleave_rate,
                                            const MainType bind_conc, const int ninit,
                                            boost::random::mt19937& rng,
                                            const MainType delta, const MainType beta,
                                            const MainType sqp_min_stepsize,
                                            const int scan_max_iter,
                                            const MainType tol, const MainType x_tol,
                                            const MainType qp_stepsize_tol, 
                                            const QuasiNewtonMethod quasi_newton,
                                            const int hessian_modify_max_iter, 
                                            const MainType c1, const MainType c2,
                                            const int line_search_max_iter,
                                            const int zoom_max_iter, const int qp_max_iter)
{
    const int N = constraints->getN();
    const int D = constraints->getD();
    Matrix<mpq_rational, Dynamic, Dynamic> A = constraints->getA();
    Matrix<mpq_rational, Dynamic, 1> b = constraints->getb();

    // Ensure that the given parameter vector lies inside the given polytope 
    Matrix<mpq_rational, Dynamic, 1> logrates_ = logrates.template cast<mpq_rational>();
    if (!constraints->query(logrates_))
        logrates_ = constraints->approxNearestL2<mpq_rational>(logrates_).eval();
   
    // Find the endpoints of the main chord containing the given parameter vector
    /*
    const int endpoint_scan_max_iter = 20;    // TODO Customize?
    const mpq_rational a = 1;
    mpq_rational x_lower_min = 0;
    mpq_rational x_lower_max = 0;
    mpq_rational x_upper_min = 0;
    mpq_rational x_upper_max = 0;
    Matrix<mpq_rational, Dynamic, 1> logrates_ = logrates.template cast<mpq_rational>();
    while (constraints->query(logrates_ - x_lower_max * Matrix<mpq_rational, Dynamic, 1>::Ones(D)))
        x_lower_max += a;
    while (constraints->query(logrates_ + x_upper_max * Matrix<mpq_rational, Dynamic, 1>::Ones(D)))
        x_upper_max += a;
    std::cout << "initial bounds: (lower) " << x_lower_min << " " << x_lower_max 
                             << " (upper) " << x_upper_min << " " << x_upper_max << std::endl << std::flush;
    for (int i = 0; i < endpoint_scan_max_iter; ++i)
    {
        mpq_rational x_lower_mid = (x_lower_min + x_lower_max) / 2;
        Matrix<mpq_rational, Dynamic, 1> logrates_lower = logrates_ - x_lower_mid * Matrix<mpq_rational, Dynamic, 1>::Ones(D);
        if (!constraints->query(logrates_lower))
            x_lower_max = x_lower_mid;
        else
            x_lower_min = x_lower_mid;
        std::cout << "changing lower bounds to " << x_lower_min << " " << x_lower_max << std::endl << std::flush;
    }
    for (int i = 0; i < endpoint_scan_max_iter; ++i)
    {
        mpq_rational x_upper_mid = (x_upper_min + x_upper_max) / 2;
        Matrix<mpq_rational, Dynamic, 1> logrates_upper = logrates_ + x_upper_mid * Matrix<mpq_rational, Dynamic, 1>::Ones(D);
        if (!constraints->query(logrates_upper))
            x_upper_max = x_upper_mid;
        else
            x_upper_min = x_upper_mid;
        std::cout << "changing upper bounds to " << x_upper_min << " " << x_upper_max << std::endl << std::flush;
    }
    MainType x_lower = static_cast<MainType>(-x_lower_min);
    MainType x_upper = static_cast<MainType>(x_upper_min);
    */
    Matrix<mpq_rational, Dynamic, 1> c = b - A * logrates_;
    Matrix<mpq_rational, Dynamic, 1> d = A * Matrix<mpq_rational, Dynamic, 1>::Ones(D);
    mpq_rational x_lower_ = std::numeric_limits<mpq_rational>::infinity(); 
    mpq_rational x_upper_ = -std::numeric_limits<mpq_rational>::infinity();
    for (int i = 0; i < N; ++i)
    {
        if (d(i) != 0)
        {
            mpq_rational bound = c(i) / d(i);
            if (constraints->query(logrates_ + bound * Matrix<mpq_rational, Dynamic, 1>::Ones(D)))
            {
                if (bound < x_lower_)
                    x_lower_ = bound;
                else if (bound > x_upper_)
                    x_upper_ = bound;
            }
        }
    }
    MainType x_lower = static_cast<MainType>(x_lower_);
    MainType x_upper = static_cast<MainType>(x_upper_);
    std::cout << constraints->query(logrates_) << std::endl << std::flush; 
    std::cout << logrates.transpose() << std::endl << std::flush;
    std::cout << x_lower << " " << x_upper << std::endl << std::flush;
    
    // Set up a 1-D SQPOptimizer instance
    SQPOptimizer1D<MainType>* opt = new SQPOptimizer1D<MainType>(x_lower, x_upper);

    // Define error function to be minimized 
    std::function<MainType(MainType)> func = [&logrates_, &cleave_rate, &bind_conc](MainType x) -> MainType
        {
            Matrix<MainType, Dynamic, 1> p = Matrix<MainType, Dynamic, 1>::Ones(logrates_.size());
            Matrix<MainType, Dynamic, 1> logrates_new = logrates_.template cast<MainType>() + x * p;
            return cleaveErrorAgainstPerfectOverallRate(logrates_new, cleave_rate, bind_conc);
        };

    // For each optimization attempt ... 
    boost::random::uniform_real_distribution<double> scan_dist(
        static_cast<double>(x_lower), static_cast<double>(x_upper)
    );
    QuadraticProgramSolveMethod qp_solve_method = USE_CUSTOM_SOLVER;
    MainType best_fit = 0;
    MainType best_error = std::numeric_limits<MainType>::infinity();
    for (int i = 0; i < ninit; ++i)
    {
        MainType x_init = static_cast<MainType>(scan_dist(rng));
        Matrix<MainType, Dynamic, 1> l_init = Matrix<MainType, Dynamic, 1>::Ones(2);

        // Obtain the parameter vector along the main chord that yields the 
        // overall cleavage rate closest to the given value
        //
        // Use no regularization for this optimization
        MainType fit = opt->run(
            func, quasi_newton, RegularizationMethod::NOREG, 0, 0,
            qp_solve_method, x_init, l_init, delta, beta, sqp_min_stepsize,
            scan_max_iter, tol, x_tol, qp_stepsize_tol, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, false,
            false, false
        );

        // Re-obtain error for the chosen parameter vector
        Matrix<MainType, Dynamic, 1> logrates_new = logrates_.template cast<MainType>() + fit * Matrix<MainType, Dynamic, 1>::Ones(D);
        MainType error = cleaveErrorAgainstPerfectOverallRate(logrates_new, cleave_rate, bind_conc);
        
        // Store the parameter vector with the least error
        if (error < best_error)
        {
            best_fit = fit;
            best_error = error;
        }
        std::cout << "- fit number " << i << ": " << fit << std::endl << std::flush;
    }
    delete opt;

    return std::make_pair(best_fit, best_error);
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);

    /** ------------------------------------------------------- //
     *       DEFINE POLYTOPE FOR DETERMINING b, d, b', d'       //
     *  ------------------------------------------------------- */
    std::string poly_filename = "polytopes/line_3_plusbind.poly";
    std::string vert_filename = "polytopes/line_3_plusbind.vert";
    Polytopes::LinearConstraints* constraints = new Polytopes::LinearConstraints(Polytopes::InequalityType::GreaterThanOrEqualTo);
    constraints->parse(poly_filename);
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = Polytopes::parseVertexCoords(vert_filename);   

    /** ------------------------------------------------------- //
     *                    PARSE CONFIGURATIONS                  //
     *  ------------------------------------------------------- */ 
    boost::json::object json_data = parseConfigFile(argv[1]).as_object();

    // Check that input/output file paths were specified 
    if (!json_data.if_contains("cleave_data_filename"))
        throw std::runtime_error("Cleavage rate dataset must be specified");
    if (!json_data.if_contains("output_prefix"))
        throw std::runtime_error("Output file prefix must be specified");
    std::string cleave_infilename = json_data["cleave_data_filename"].as_string().c_str();
    std::string outprefix = json_data["output_prefix"].as_string().c_str();
    std::string outfilename = outprefix + "-main.tsv"; 
    std::string residuals_filename = outprefix + "-residuals.tsv";

    // Assign default values for parameters that were not specified 
    bool data_specified_as_times = false;
    int ninit = 100; 
    MainType cleave_pseudocount = 0;
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

    // Parse measured cleavage rates, along with the mismatched sequences on
    // which they were measured 
    int n_cleave_data = 0; 
    MatrixXi cleave_seqs = MatrixXi::Zero(0, length); 
    Matrix<MainType, Dynamic, 1> cleave_data = Matrix<MainType, Dynamic, 1>::Zero(0); 
    
    // Parse the input file of overall cleavage rates
    std::ifstream infile;
    std::string line, token;
    infile.open(cleave_infilename);
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

    // Exit if no overall cleavage rates were specified 
    if (n_cleave_data == 0)
        throw std::runtime_error("Cleavage rate dataset is empty");

    // Assume that the cleavage rate for the perfect-match substrate is
    // specified first ...
    //
    // ... and thus normalize all overall cleavage rates and invert
    MainType perfect_cleave_rate = cleave_data(0);
    for (int i = 1; i < cleave_data.size(); ++i)
        cleave_data(i) = cleave_data(i) / cleave_data(0);   // inverse overall cleavage rate ratio 
                                                            // = rate on mismatched / rate on perfect
    cleave_data(0) = 1;

    // Add pseudocounts
    cleave_data += cleave_pseudocount * Matrix<MainType, Dynamic, 1>::Ones(n_cleave_data);

    // Define matrix of single-mismatch DNA sequences relative to the 
    // perfect-match sequence for cleavage rates
    MatrixXi single_mismatch_seqs;
    single_mismatch_seqs = MatrixXi::Ones(length, length) - MatrixXi::Identity(length, length); 

    /** -------------------------------------------------------------- //
     *               FIT AGAINST GIVEN CLEAVAGE RATE DATA              //
     *  -------------------------------------------------------------- */ 
    std::pair<Matrix<MainType, Dynamic, Dynamic>, Matrix<MainType, Dynamic, Dynamic> > results;
    results = fitLineParamsAgainstMeasuredRatesCleave(
        constraints, vertices, cleave_data, cleave_seqs, bind_conc, ninit, rng,
        delta, beta, sqp_min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
        quasi_newton, regularize, regularize_weight, hessian_modify_max_iter,
        c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, verbose,
        search_verbose, zoom_verbose
    );
    Matrix<MainType, Dynamic, Dynamic> best_fits = results.first;
    Matrix<MainType, Dynamic, Dynamic> residuals = results.second;
    Matrix<MainType, Dynamic, 1> errors = residuals.rowwise().sum();
    std::cout << "done with fitting\n" << std::flush;

    /** -------------------------------------------------------------- //
     *            SCAN MAIN CHORD CORRESPONDING TO EACH FIT            //
     *  -------------------------------------------------------------- */
    // Scan main chord corresponding to each fit to optimize against the 
    // perfect-match overall cleavage rate 
    std::pair<MainType, MainType> scan_results;
    const int n_scan_init = 50;      // TODO Customize?
    const int scan_max_iter = 100;   // TODO Customize?
    for (int i = 0; i < ninit; ++i)
    {
        std::cout << "scanning for " << i << std::endl << std::flush;
        scan_results = scanMainChord(
            best_fits.row(i), constraints, perfect_cleave_rate, bind_conc,
            n_scan_init, rng, delta, beta, sqp_min_stepsize, scan_max_iter,
            tol, x_tol, qp_stepsize_tol, quasi_newton, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter
        );
        best_fits.row(i) += scan_results.first * Matrix<MainType, Dynamic, 1>::Ones(best_fits.cols());
    }
    std::cout << "done with scanning\n" << std::flush;

    // Output the fits to file
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    outfile << "fit_attempt\tmatch_forward\tmatch_reverse\tmismatch_forward\t"
            << "mismatch_reverse\tterminal_unbind_rate\tterminal_cleave_rate\t"
            << "terminal_bind_rate\tcleave_error\t";
    for (int i = 0; i < length; ++i)
    {
        outfile << "mm" << i << "_prob\t"
                << "mm" << i << "_cleave\t"
                << "mm" << i << "_unbind\t"
                << "mm" << i << "_oct\t"
                << "mm" << i << "_spec\t"
                << "mm" << i << "_rapid\t"
                << "mm" << i << "_octratio\t";
    }
    int pos = outfile.tellp();
    outfile.seekp(pos - 1);
    outfile << std::endl;  
    for (int i = 0; i < ninit; ++i)
    {
        outfile << i << '\t'; 

        // Write each best-fit parameter vector ...
        for (int j = 0; j < 7; ++j)
            outfile << best_fits(i, j) << '\t';

        // ... along with the associated error against the corresponding data ... 
        outfile << errors(i) << '\t';

        // ... along with the associated single-mismatch cleavage statistics
        Matrix<MainType, Dynamic, 7> fit_single_mismatch_stats = computeCleavageStats(
            best_fits.row(i), single_mismatch_seqs, bind_conc
        );
        for (int j = 0; j < fit_single_mismatch_stats.rows(); ++j)
        {
            for (int k = 0; k < 7; ++k)
                outfile << fit_single_mismatch_stats(j, k) << '\t';
        }
        pos = outfile.tellp();
        outfile.seekp(pos - 1); 
        outfile << std::endl;
    }
    outfile.close();

    // Output the optimal cleavage rate ratio residuals to file 
    std::ofstream residuals_outfile(residuals_filename);
    residuals_outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    residuals_outfile << "seq\t";
    for (int i = 0; i < ninit - 1; ++i)
        residuals_outfile << "fit_" << i << '\t'; 
    residuals_outfile << "fit_" << ninit - 1 << std::endl;
    residuals_outfile << "total\t";
    for (int i = 0; i < ninit - 1; ++i)
        residuals_outfile << residuals.row(i).sum() << '\t';
    residuals_outfile << residuals.row(ninit - 1).sum() << std::endl;
    for (int i = 0; i < residuals.cols(); ++i)
    {
        // Output the sequence ... 
        for (int j = 0; j < length; ++j)
            residuals_outfile << (cleave_seqs(i, j) ? '1' : '0');
        residuals_outfile << '\t';

        // ... and each vector of cleavage rate residuals 
        for (int j = 0; j < residuals.rows() - 1; ++j)
            residuals_outfile << residuals(j, i) << '\t';
        residuals_outfile << residuals(residuals.rows() - 1, i) << std::endl;
    }
    residuals_outfile.close();

    delete constraints;
    return 0;
} 

