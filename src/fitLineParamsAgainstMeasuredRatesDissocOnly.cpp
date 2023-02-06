/**
 * Using line-search SQP, identify the set of *position-specific* line graph
 * parameter vectors that yields each given set of specific dissociativities
 * in the given data file.
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
 *     2/6/2023
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
 * Compute the *unregularized* sum-of-squares error between a set of specific
 * dissociativities inferred from the given LGPs and a set of experimentally
 * determined dissociativities. 
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.
 *
 * @param logrates
 * @param unbind_seqs
 * @param unbind_data
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
 * @param constraints
 * @param vertices
 * @param unbind_data
 * @param unbind_seqs
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
     *       DEFINE POLYTOPE FOR DETERMINING b, d', b', d'      //
     *  ------------------------------------------------------- */
    std::string poly_filename = "polytopes/line_4_Rloop.poly";
    std::string vert_filename = "polytopes/line_4_Rloop.vert";
    Polytopes::LinearConstraints* constraints_1 = new Polytopes::LinearConstraints(Polytopes::InequalityType::GreaterThanOrEqualTo);
    constraints_1->parse(poly_filename);
    Matrix<mpq_rational, Dynamic, Dynamic> vertices_1 = Polytopes::parseVertexCoords(vert_filename);   

    /** ------------------------------------------------------- //
     *                    PARSE CONFIGURATIONS                  //
     *  ------------------------------------------------------- */ 
    boost::json::object json_data = parseConfigFile(argv[1]).as_object();

    // Check that input/output file paths were specified 
    if (!json_data.if_contains("unbind_data_filename"))
        throw std::runtime_error("Dataset must be specified");
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

    // Parse measured dead unbinding rates, along with the mismatched sequences
    // on which they were measured 
    int n_unbind_data = 0;
    MatrixXi unbind_seqs = MatrixXi::Zero(0, length); 
    Matrix<MainType, Dynamic, 1> unbind_data = Matrix<MainType, Dynamic, 1>::Zero(0);
    
    // Parse the input file of unbinding rates
    std::ifstream infile;
    std::string line, token;
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

    // Exit if no unbinding rates were specified 
    if (n_unbind_data == 0)
        throw std::runtime_error("Unbinding rate dataset is empty");

    // Also invert all parsed ndABAs (i.e., specific dissociativities)
    if (unbind_data.size() > 0)
        unbind_data = unbind_data.array().pow(-1).matrix();
    
    // Add pseudocounts to the unbinding rates 
    unbind_data += unbind_pseudocount * Matrix<MainType, Dynamic, 1>::Ones(n_unbind_data);

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

    // Output the fits to file
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    outfile << "fit_attempt\tmatch_forward\tmatch_reverse\tmismatch_forward\t"
            << "mismatch_reverse\tdissoc_error\n";
    int pos = outfile.tellp();
    outfile.seekp(pos - 1);
    outfile << std::endl;  
    for (int i = 0; i < ninit; ++i)
    {
        outfile << i << '\t'; 

        // Write each best-fit parameter vector ...
        for (int j = 0; j < 4; ++j)
            outfile << best_fits_dissoc(i, j) << '\t';

        // ... along with the associated error against the corresponding data 
        outfile << errors_dissoc(i) << std::endl;
    }
    outfile.close();

    // Output the dissociativity residuals to file 
    std::ofstream residuals_outfile(residuals_filename);
    residuals_outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    for (int i = 0; i < ninit - 1; ++i)
        residuals_outfile << "fit_" << i << '\t'; 
    residuals_outfile << "fit_" << ninit - 1 << std::endl;
    for (int i = 0; i < residuals_dissoc.rows(); ++i)
    {
        // Output the sequence ... 
        for (int j = 0; j < length; ++j)
            residuals_outfile << (unbind_seqs(i, j) ? '1' : '0');
        residuals_outfile << '\t';

        // ... and each vector of dissociativity residuals
        for (int j = 0; j < residuals_dissoc.cols() - 1; ++j)
            residuals_outfile << residuals_dissoc(i, j) << '\t';
        residuals_outfile << residuals_dissoc(i, residuals_dissoc.cols() - 1) << std::endl;
    }
    residuals_outfile.close();

    delete constraints_1;
    delete constraints_2;
    return 0;
} 

