/**
 * Using line-search SQP, identify the set of line graph parameter vectors
 * that yields each given set of apparent binding affinities in the given data
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
 *     3/3/2023
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
 * Compute the apparent binding affinity on all matched/mismatched sequences
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
Matrix<MainType, Dynamic, 1> computeApparentBindingAffinity(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
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
        
        // Compute *inverse* apparent binding affinity: rate on perfect / rate on mismatched
        PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate);
        stats(i) = log10(unbind_rate_perfect) - log10(unbind_rate);
    }

    delete model;
    return stats.template cast<MainType>();
}

/**
 * Compute the *unregularized* sum-of-squares error between a set of apparent
 * binding affinities inferred from the given LGPs and a set of experimentally
 * determined apparent binding affinities.
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates    Input vector of 5 LGPs.
 * @param unbind_seqs Matrix of input sequences, with entries of 0 (match w.r.t.
 *                    perfect-match sequence) or 1 (mismatch w.r.t. perfect-match
 *                    sequence).
 * @param unbind_data Matrix of measured apparent binding affinities for the 
 *                    given input sequences. 
 * @returns Vector of residuals yielding the sum-of-squares errors. 
 */
Matrix<MainType, Dynamic, 1> dissocErrorAgainstData(const Ref<const Matrix<MainType, Dynamic, 1> >& logrates,
                                                    const Ref<const MatrixXi>& unbind_seqs,
                                                    const Ref<const Matrix<MainType, Dynamic, 1> >& unbind_data)
{
    // Compute apparent binding affinities for the given complementarity patterns
    Matrix<MainType, Dynamic, 1> stats = computeApparentBindingAffinity(logrates, unbind_seqs);
    for (int i = 0; i < stats.size(); ++i)    // Convert to linear scale 
        stats(i) = pow(ten_main, stats(i)); 

    // Compute each residual contributing to overall sum-of-squares error
    Matrix<MainType, Dynamic, 1> residuals = (stats - unbind_data).array().pow(2); 

    return residuals;
}

/**
 * @param constraints             Constraints defining input polytope.
 * @param vertices                Vertex coordinates of input polytope.
 * @param unbind_data             Matrix of measured apparent binding affinities.
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

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);

    /** ------------------------------------------------------- //
     *       DEFINE POLYTOPE FOR DETERMINING b, d, b', d'       //
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
    if (!json_data.if_contains("unbind_data_filename"))
        throw std::runtime_error("Apparent binding affinity dataset must be specified");
    if (!json_data.if_contains("output_prefix"))
        throw std::runtime_error("Output file prefix must be specified");
    std::string unbind_infilename = json_data["unbind_data_filename"].as_string().c_str();
    std::string outprefix = json_data["output_prefix"].as_string().c_str();
    std::string outfilename = outprefix + "-main.tsv"; 
    std::string residuals_filename = outprefix + "-residuals.tsv";

    // Assign default values for parameters that were not specified 
    bool data_specified_as_times = false;
    int ninit = 100; 
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
    if (json_data.if_contains("unbind_pseudocount"))
    {
        unbind_pseudocount = static_cast<MainType>(json_data["unbind_pseudocount"].as_double()); 
        if (unbind_pseudocount < 0)
            throw std::runtime_error("Invalid apparent binding affinity pseudocount specified");
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

    // Parse measured apparent binding affinities, along with the mismatched
    // sequences on which they were measured 
    int n_unbind_data = 0;
    MatrixXi unbind_seqs = MatrixXi::Zero(0, length); 
    Matrix<MainType, Dynamic, 1> unbind_data = Matrix<MainType, Dynamic, 1>::Zero(0);
    
    // Parse the input file of apparent binding affinities 
    std::ifstream infile;
    std::string line, token;
    infile.open(unbind_infilename);
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

        // The second entry is the apparent binding affinity being parsed
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

    // Exit if no apparent binding affinities were specified 
    if (n_unbind_data == 0)
        throw std::runtime_error("Apparent binding affinity dataset is empty");

    // Invert all parsed apparent binding affinities 
    if (unbind_data.size() > 0)
        unbind_data = unbind_data.array().pow(-1).matrix();
    
    // Add pseudocounts
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
    std::cout << "R-loop rate log-ratios after fit: "
              << fit_logrates(0) - fit_logrates(1) << " "
              << fit_logrates(2) - fit_logrates(3) << std::endl;
    std::cout << "------------------------------------------------------\n";

    delete constraints_1;
    return 0;
} 

