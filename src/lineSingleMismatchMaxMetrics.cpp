/**
 * Using line-search SQP, identify the set of line graph parameter vectors
 * that maximize activity, speed, specificity, specific rapidity, and 
 * specific dissociativity with respect to single-mismatch substrates.
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
 *     2/10/2023
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
constexpr int INTERNAL_PRECISION = 100;
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;
const int length = 20;
const PreciseType ten(10);

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
 * Compute the activity (cleavage probability against perfect-match substrate)
 * for the given LG.
 *
 * Here, `logrates` is assumed to contain *6* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate. 
 *
 * @param logrates Input vector of 6 LGPs.
 */
PreciseType computeActivity(const Ref<const VectorXd>& logrates)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1)))
    );

    // Terminal unbinding and cleavage rates 
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute and return cleavage probability against perfect-match substrate
    PreciseType prob_perfect = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
    
    delete model;
    return prob_perfect;
}

/**
 * Compute the specificity against the given complementarity pattern for the
 * given LG. 
 *
 * Here, `logrates` is assumed to contain *6* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate. 
 *
 * @param logrates Input vector of 6 LGPs.
 * @param seq      Input sequence, with entries of 0 (match w.r.t. perfect-match
 *                 sequence) or 1 (mismatch w.r.t. perfect-match sequence).
 */
PreciseType computeSpecificity(const Ref<const VectorXd>& logrates, const Ref<const VectorXi>& seq)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(2))),
        pow(ten, static_cast<PreciseType>(logrates(3)))
    );

    // Terminal unbinding and cleavage rates 
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute cleavage probability against perfect-match substrate
    PreciseType prob_perfect = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);

    // Compute cleavage probability against given substrate 
    for (int j = 0; j < length; ++j)
    {
        if (seq(j))
            model->setEdgeLabels(j, match_rates);
        else
            model->setEdgeLabels(j, mismatch_rates);
    }
    PreciseType prob = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);

    delete model;
    return log10(prob_perfect) - log10(prob);
}

/**
 * Compute the speed (conditional cleavage rate on perfect-match substrate)
 * for the given LG. 
 *
 * Here, `logrates` is assumed to contain *6* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate. 
 *
 * @param logrates Input vector of 6 LGPs.
 */
PreciseType computeSpeed(const Ref<const VectorXd>& logrates)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1)))
    );

    // Terminal unbinding and cleavage rates 
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute and return conditional cleavage rate against perfect-match substrate 
    PreciseType cleave_rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate);

    delete model;
    return cleave_rate_perfect;
}

/**
 * Compute the specific rapidity on the given complementarity pattern for the
 * given LG. 
 *
 * Here, `logrates` is assumed to contain *6* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate. 
 *
 * @param logrates Input vector of 6 LGPs.
 * @param seq      Input sequence, with entries of 0 (match w.r.t. perfect-match
 *                 sequence) or 1 (mismatch w.r.t. perfect-match sequence).
 */
PreciseType computeRapidity(const Ref<const VectorXd>& logrates, const Ref<const VectorXi>& seq)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(2))),
        pow(ten, static_cast<PreciseType>(logrates(3)))
    );

    // Terminal unbinding and cleavage rates 
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute conditional cleavage rate against perfect-match substrate 
    PreciseType cleave_rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate);

    // Compute conditional cleavage rate against given substrate
    for (int j = 0; j < length; ++j)
    {
        if (seq(j))
            model->setEdgeLabels(j, match_rates);
        else
            model->setEdgeLabels(j, mismatch_rates);
    }
    PreciseType cleave_rate = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate);

    delete model;
    return log10(cleave_rate_perfect) - log10(cleave_rate);
}

/**
 * Compute the specific dissociativity on the given complementarity pattern 
 * for the given LG. 
 *
 * Here, `logrates` is assumed to contain *6* entries:
 * 1) forward rate at DNA-RNA matches,
 * 2) reverse rate at DNA-RNA matches,
 * 3) forward rate at DNA-RNA mismatches,
 * 4) reverse rate at DNA-RNA mismatches,
 * 5) terminal unbinding rate,
 * 6) terminal cleavage rate. 
 *
 * @param logrates Input vector of 6 LGPs.
 * @param seq      Input sequence, with entries of 0 (match w.r.t. perfect-match
 *                 sequence) or 1 (mismatch w.r.t. perfect-match sequence).
 */
PreciseType computeDissociativity(const Ref<const VectorXd>& logrates, const Ref<const VectorXi>& seq)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(2))),
        pow(ten, static_cast<PreciseType>(logrates(3)))
    );

    // Terminal unbinding rate 
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int i = 0; i < length; ++i)
        model->setEdgeLabels(i, match_rates); 
  
    // Compute dead unbinding rate against perfect-match substrate
    PreciseType unbind_rate_perfect = model->getLowerExitRate(terminal_unbind_rate);

    // Compute dead unbinding rate against given substrate
    for (int j = 0; j < length; ++j)
    {
        if (seq(j))
            model->setEdgeLabels(j, match_rates);
        else
            model->setEdgeLabels(j, mismatch_rates);
    }
    PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate);

    delete model;
    return log10(unbind_rate) - log10(unbind_rate_perfect);
}

/**
 * @param constraints
 * @param vertices
 * @param ninit
 * @param rng
 * @param delta
 * @param beta
 * @param sqp_min_stepsize
 * @param max_iter
 * @param tol
 * @param x_tol
 * @param qp_stepsize_tol
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
MatrixXd maximizeMetric(std::function<double(const Ref<const VectorXd>&)> func,
                        Polytopes::LinearConstraints* constraints,
                        const Ref<const Matrix<mpq_rational, Dynamic, Dynamic> >& vertices,  
                        const int ninit, boost::random::mt19937& rng,
                        const double delta, const double beta,
                        const double sqp_min_stepsize, const int max_iter,
                        const double tol, const double x_tol,
                        const double qp_stepsize_tol, 
                        const int hessian_modify_max_iter, 
                        const double c1, const double c2,
                        const int line_search_max_iter,
                        const int zoom_max_iter, const int qp_max_iter,
                        const bool verbose, const bool search_verbose,
                        const bool zoom_verbose) 
{
    const int N = constraints->getN();
    const int D = constraints->getD();
    SQPOptimizer<double>* opt = new SQPOptimizer<double>(constraints);

    // Sample a set of initial parameter points from the given polytope 
    Delaunay_triangulation* tri = new Delaunay_triangulation(D); 
    MatrixXd init_points(ninit, D); 
    Polytopes::triangulate(vertices, tri);
    init_points = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, ninit, 0, rng).cast<double>();
    delete tri;

    // Get the vertices of the given polytope and extract the min/max bounds 
    // of each parameter
    Matrix<double, Dynamic, 2> bounds(D, 2); 
    for (int i = 0; i < D; ++i)
    {
        mpq_rational min_param = vertices.col(i).minCoeff();
        mpq_rational max_param = vertices.col(i).maxCoeff();
        bounds(i, 0) = static_cast<double>(min_param); 
        bounds(i, 1) = static_cast<double>(max_param);
    }
    VectorXd regularize_bases = (bounds.col(0) + bounds.col(1)) / 2;

    // Define vector of regularization weights (all zero)
    VectorXd regularize_weights = VectorXd::Zero(D);

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    MatrixXd maxima(ninit, D);
    VectorXd x_init, l_init;
    QuadraticProgramSolveMethod qp_solve_method = USE_CUSTOM_SOLVER;
    for (int i = 0; i < ninit; ++i)
    {
        // Assemble initial parameter values
        x_init = init_points.row(i); 
        l_init = (
            VectorXd::Ones(N)
            - constraints->active(x_init.cast<mpq_rational>()).template cast<double>()
        );

        // Obtain best-fit parameter values from the initial parameters
        maxima.row(i) = opt->run(
            func, QuasiNewtonMethod::BFGS, RegularizationMethod::NOREG, regularize_bases,
            regularize_weights, qp_solve_method, x_init, l_init, delta, beta,
            sqp_min_stepsize, max_iter, tol, x_tol, qp_stepsize_tol,
            hessian_modify_max_iter, c1, c2, line_search_max_iter, zoom_max_iter,
            qp_max_iter, verbose, search_verbose, zoom_verbose
        );
    }
    delete opt;

    return maxima; 
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);

    /** ------------------------------------------------------- //
     *                    PARSE CONFIGURATIONS                  //
     *  --------------------------------------------------------*/ 
    boost::json::object json_data = parseConfigFile(argv[1]).as_object();

    // Check that input/output file paths were specified 
    if (!json_data.if_contains("poly_filename"))
        throw std::runtime_error("Polytope constraints file (.poly) must be specified");
    if (!json_data.if_contains("vert_filename"))
        throw std::runtime_error("Polytope vertices file (.vert) must be specified");
    if (!json_data.if_contains("output_prefix"))
        throw std::runtime_error("Output file path prefix must be specified"); 
    std::string poly_filename = json_data["poly_filename"].as_string().c_str();
    std::string vert_filename = json_data["vert_filename"].as_string().c_str();
    std::string outprefix = json_data["output_prefix"].as_string().c_str();
    int n_init = 100;
    if (json_data.if_contains("n_init"))
    {
        n_init = json_data["n_init"].as_int64();
        if (n_init <= 0)
            throw std::runtime_error("Invalid value for n_init specified");
    }

    // Parse SQP configurations
    int sqp_max_iter = 1000;
    double delta = 1e-8; 
    double beta = 1e-4;
    double sqp_min_stepsize = 1e-8; 
    double sqp_tol = 1e-7;
    double qp_stepsize_tol = 1e-8;
    int hessian_modify_max_iter = 10000;
    double c1 = 1e-4;
    double c2 = 0.9;
    int line_search_max_iter = 5;
    int zoom_max_iter = 5;
    int qp_max_iter = 100;
    bool sqp_verbose = false;
    bool sqp_line_search_verbose = false;
    bool sqp_zoom_verbose = false;
    if (json_data.if_contains("sqp_config"))
    {
        boost::json::object sqp_data = json_data["sqp_config"].as_object(); 
        if (sqp_data.if_contains("delta"))
        {
            delta = sqp_data["delta"].as_double();
            if (delta <= 0)
                throw std::runtime_error("Invalid value for delta specified"); 
        }
        if (sqp_data.if_contains("beta"))
        {
            beta = sqp_data["beta"].as_double(); 
            if (beta <= 0)
                throw std::runtime_error("Invalid value for beta specified"); 
        }
        if (sqp_data.if_contains("min_stepsize"))
        {
            sqp_min_stepsize = sqp_data["min_stepsize"].as_double();
            if (sqp_min_stepsize <= 0 || sqp_min_stepsize >= 1)
                throw std::runtime_error("Invalid value for minimum stepsize (min_stepsize) specified");
        }
        if (sqp_data.if_contains("max_iter"))
        {
            sqp_max_iter = sqp_data["max_iter"].as_int64(); 
            if (sqp_max_iter <= 0)
                throw std::runtime_error("Invalid value for maximum number of SQP iterations (max_iter) specified"); 
        }
        if (sqp_data.if_contains("tol"))
        {
            sqp_tol = sqp_data["tol"].as_double();
            if (sqp_tol <= 0)
                throw std::runtime_error("Invalid value for SQP tolerance (tol) specified"); 
        }
        if (sqp_data.if_contains("qp_stepsize_tol"))
        {
            qp_stepsize_tol = sqp_data["qp_stepsize_tol"].as_double();
            if (qp_stepsize_tol <= 0)
            {
                std::stringstream ss_err;
                ss_err << "Invalid value for QP stepsize tolerance (qp_stepsize_tol) "
                      "specified";
                throw std::runtime_error(ss_err.str());
            }
        }
        if (sqp_data.if_contains("hessian_modify_max_iter"))
        {
            hessian_modify_max_iter = sqp_data["hessian_modify_max_iter"].as_int64(); 
            if (hessian_modify_max_iter <= 0)
            {
                std::stringstream ss_err; 
                ss_err << "Invalid value for maximum number of SQP Hessian modification "
                       << "iterations (hessian_modify_max_iter) specified";
                throw std::runtime_error(ss_err.str()); 
            } 
        }
        if (sqp_data.if_contains("c1"))
        {
            c1 = sqp_data["c1"].as_double();
            if (c1 <= 0)
                throw std::runtime_error("Invalid value for c1 specified"); 
        }
        if (sqp_data.if_contains("c2"))
        {
            c2 = sqp_data["c2"].as_double();
            if (c2 <= 0)
                throw std::runtime_error("Invalid value for c2 specified"); 
        }
        if (sqp_data.if_contains("line_search_max_iter"))
        {
            line_search_max_iter = sqp_data["line_search_max_iter"].as_int64();
            if (line_search_max_iter <= 0)
            {
                std::stringstream ss_err;
                ss_err << "Invalid value for maximum number of SQP line search "
                       << "iterations (line_search_max_iter) specified";
                throw std::runtime_error(ss_err.str());
            }
        }
        if (sqp_data.if_contains("zoom_max_iter"))
        {
            zoom_max_iter = sqp_data["zoom_max_iter"].as_int64();
            if (zoom_max_iter <= 0)
            {
                std::stringstream ss_err;
                ss_err << "Invalid value for maximum number of SQP zoom iterations "
                       << "(zoom_max_iter) specified";
                throw std::runtime_error(ss_err.str());
            }
        }
        if (sqp_data.if_contains("qp_max_iter"))
        {
            qp_max_iter = sqp_data["qp_max_iter"].as_int64();
            if (qp_max_iter <= 0)
            {
                std::stringstream ss_err;
                ss_err << "Invalid value for maximum number of QP solver iterations "
                       << "(qp_max_iter) specified";
                throw std::runtime_error(ss_err.str());
            }
        }
        if (sqp_data.if_contains("verbose"))
        {
            sqp_verbose = sqp_data["verbose"].as_bool();
        }
        if (sqp_data.if_contains("line_search_verbose"))
        {
            sqp_line_search_verbose = sqp_data["line_search_verbose"].as_bool();
        }
        if (sqp_data.if_contains("zoom_verbose"))
        {
            sqp_zoom_verbose = sqp_data["zoom_verbose"].as_bool();
        }
    }

    /** -------------------------------------------------------------- //
     *               DEFINE POLYTOPE FOR DETERMINING LGPs              //
     *  -------------------------------------------------------------- */
    Polytopes::LinearConstraints* constraints = new Polytopes::LinearConstraints(Polytopes::InequalityType::GreaterThanOrEqualTo);
    constraints->parse(poly_filename);
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = Polytopes::parseVertexCoords(vert_filename);

    /** -------------------------------------------------------------- //
     *                       MAXIMIZE EACH METRIC                      //
     *  -------------------------------------------------------------- */
    VectorXd max_activity_params(6); 
    VectorXd max_speed_params(6);
    VectorXd max_spec_params(length, 6);
    MatrixXd max_rapid_params(length, 6); 
    MatrixXd min_rapid_params(length, 6);
    MatrixXd max_dissoc_params(length, 6);

    // Maximize activity 
    MatrixXd optima = maximizeMetric(
        [](const Ref<const VectorXd>& x) -> double
        {
            return static_cast<double>(-computeActivity(x));
        },
        constraints, vertices, n_init, rng, delta, beta, sqp_min_stepsize,
        sqp_max_iter, sqp_tol, sqp_tol, qp_stepsize_tol, hessian_modify_max_iter,
        c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, sqp_verbose,
        sqp_line_search_verbose, sqp_zoom_verbose
    );
    VectorXd max_activities(n_init);
    for (int i = 0; i < n_init; ++i)
        max_activities(i) = static_cast<double>(computeActivity(optima.row(i)));
    Eigen::Index max_idx; 
    max_activities.maxCoeff(&max_idx);
    max_activity_params = optima.row(max_idx);

    // Maximize speed 
    optima = maximizeMetric(
        [](const Ref<const VectorXd>& x) -> double
        {
            return static_cast<double>(-computeSpeed(x));
        },
        constraints, vertices, n_init, rng, delta, beta, sqp_min_stepsize,
        sqp_max_iter, sqp_tol, sqp_tol, qp_stepsize_tol, hessian_modify_max_iter,
        c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, sqp_verbose,
        sqp_line_search_verbose, sqp_zoom_verbose
    );
    VectorXd max_speeds(n_init);
    for (int i = 0; i < n_init; ++i)
        max_speeds(i) = static_cast<double>(computeSpeed(optima.row(i)));
    max_speeds.maxCoeff(&max_idx);
    max_speed_params = optima.row(max_idx);

    for (int i = 0; i < length; ++i)
    {
        VectorXi seq = VectorXi::Ones(length);
        seq(i) = 0;

        // Maximize specificity with respect to single-mismatch substrate with 
        // mismatch at position i 
        optima = maximizeMetric(
            [&seq](const Ref<const VectorXd>& x) -> double
            {
                return static_cast<double>(-computeSpecificity(x, seq)); 
            },
            constraints, vertices, n_init, rng, delta, beta, sqp_min_stepsize,
            sqp_max_iter, sqp_tol, sqp_tol, qp_stepsize_tol, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, sqp_verbose,
            sqp_line_search_verbose, sqp_zoom_verbose
        );
        VectorXd max_spec(n_init); 
        for (int j = 0; j < n_init; ++j)
            max_spec(j) = static_cast<double>(computeSpecificity(optima.row(j), seq));
        max_spec.maxCoeff(&max_idx);
        max_spec_params.row(i) = optima.row(max_idx);

        // Maximize rapidity with respect to single-mismatch substrate with
        // mismatch at position i
        optima = maximizeMetric(
            [&seq](const Ref<const VectorXd>& x) -> double
            {
                return static_cast<double>(-computeRapidity(x, seq));
            },
            constraints, vertices, n_init, rng, delta, beta, sqp_min_stepsize,
            sqp_max_iter, sqp_tol, sqp_tol, qp_stepsize_tol, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, sqp_verbose,
            sqp_line_search_verbose, sqp_zoom_verbose
        );
        VectorXd max_rapid(n_init);
        for (int j = 0; j < n_init; ++j)
            max_rapid(j) = static_cast<double>(computeRapidity(optima.row(j), seq));
        max_rapid.maxCoeff(&max_idx);
        max_rapid_params.row(i) = optima.row(max_idx);

        // Minimize rapidity with respect to single-mismatch substrate with
        // mismatch at position i
        optima = maximizeMetric(
            [&seq](const Ref<const VectorXd>& x) -> double
            {
                return static_cast<double>(computeRapidity(x, seq));
            },
            constraints, vertices, n_init, rng, delta, beta, sqp_min_stepsize,
            sqp_max_iter, sqp_tol, sqp_tol, qp_stepsize_tol, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, sqp_verbose,
            sqp_line_search_verbose, sqp_zoom_verbose
        );
        VectorXd min_rapid(n_init);
        for (int j = 0; j < n_init; ++j)
            min_rapid(j) = static_cast<double>(computeRapidity(optima.row(j), seq));
        min_rapid.minCoeff(&max_idx);
        min_rapid_params.row(i) = optima.row(max_idx);

        // Maximize rapidity with respect to single-mismatch substrate with
        // mismatch at position i
        optima = maximizeMetric(
            [&seq](const Ref<const VectorXd>& x) -> double
            {
                return static_cast<double>(-computeDissociativity(x, seq));
            },
            constraints, vertices, n_init, rng, delta, beta, sqp_min_stepsize,
            sqp_max_iter, sqp_tol, sqp_tol, qp_stepsize_tol, hessian_modify_max_iter,
            c1, c2, line_search_max_iter, zoom_max_iter, qp_max_iter, sqp_verbose,
            sqp_line_search_verbose, sqp_zoom_verbose
        );
        VectorXd max_dissoc(n_init);
        for (int j = 0; j < n_init; ++j)
            max_dissoc(j) = static_cast<double>(computeDissociativity(optima.row(j), seq));
        max_dissoc.maxCoeff(&max_idx);
        max_dissoc_params.row(i) = optima.row(max_idx);
    }

    // Output maximized values to file
    std::stringstream ss; 
    ss << outprefix << "-maximized-metrics.tsv"; 
    std::ofstream outfile(ss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    outfile << "activity\t";
    for (int i = 0; i < 5; ++i)
        outfile << max_activity_params(i) << '\t';
    outfile << max_activity_params(5) << std::endl;
    outfile << "speed\t"; 
    for (int i = 0; i < 5; ++i)
        outfile << max_speed_params(i) << '\t'; 
    outfile << max_speed_params(5) << std::endl; 
    for (int i = 0; i < length; ++i)
    {
        outfile << "spec_mm" << i << '\t';
        for (int j = 0; j < 5; ++j)
            outfile << max_spec_params(i, j) << '\t';
        outfile << max_spec_params(i, 5) << std::endl;
    }
    for (int i = 0; i < length; ++i)
    {
        outfile << "maxrapid_mm" << i << '\t';
        for (int j = 0; j < 5; ++j)
            outfile << max_rapid_params(i, j) << '\t';
        outfile << max_rapid_params(i, 5) << std::endl; 
    }
    for (int i = 0; i < length; ++i)
    {
        outfile << "minrapid_mm" << i << '\t';
        for (int j = 0; j < 5; ++j)
            outfile << min_rapid_params(i, j) << '\t';
        outfile << min_rapid_params(i, 5) << std::endl; 
    }
    for (int i = 0; i < length; ++i)
    {
        outfile << "deaddissoc_mm" << i << '\t';
        for (int j = 0; j < 5; ++j)
            outfile << max_dissoc_params(i, j) << '\t';
        outfile << max_dissoc_params(i, 5) << std::endl;
    }
    outfile.close();

    delete constraints;
    return 0;
} 

