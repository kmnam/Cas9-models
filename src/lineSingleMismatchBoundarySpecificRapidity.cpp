/**
 * Estimates the boundary of the cleavage specificity vs. specific rapidity
 * region in the line graph.
 *
 * Abbreviations in the below comments:
 * - LG:   line graph
 * - LGPs: line graph parameters
 * - SQP:  sequential quadratic programming
 *
 * **Author:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     10/3/2022
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <boostMultiprecisionEigen.hpp>
#include <boundaryFinder.hpp>
#include <graphs/line.hpp>

using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
typedef number<mpfr_float_backend<100> > PreciseType; 

const int length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng;

/**
 * Get the maximum distance between any pair of vertices in the given matrix.
 *
 * @param vertices Matrix of vertex coordinates.
 * @returns        Maximum distance between any pair of vertices. 
 */
template <typename T>
T getMaxDist(const Ref<const Matrix<T, Dynamic, Dynamic> >& vertices)
{
    T maxdist = 0;
    for (int i = 0; i < vertices.rows() - 1; ++i)
    {
        for (int j = i + 1; j < vertices.rows(); ++j)
        {
            T dist = (vertices.row(i) - vertices.row(j)).norm(); 
            if (maxdist < dist)
                maxdist = dist; 
        }
    }
    
    return maxdist; 
}

/**
 * Compute the following quantities for the given set of LGPs:
 *
 * - cleavage specificity w.r.t the single-mismatch substrate for the given
 *   mismatch position and
 * - specific rapidity w.r.t the single-mismatch substrate for the given
 *   mismatch position
 *
 * with the terminal unbinding rate set to one.
 *
 * @param input Input vector of LGPs.
 * @returns Cleavage specificity and specific rapidity w.r.t the single-mismatch
 *          substrate for the given mismatch position. 
 */
template <typename T, int position>
VectorXd computeCleavageStats(const Ref<const VectorXd>& input)
{
    // Array of DNA/RNA match parameters
    std::pair<T, T> match;
    match.first = static_cast<T>(std::pow(10.0, input(0)));
    match.second = static_cast<T>(std::pow(10.0, input(1)));

    // Array of DNA/RNA mismatch parameters
    std::pair<T, T> mismatch;
    mismatch.first = static_cast<T>(std::pow(10.0, input(2)));
    mismatch.second = static_cast<T>(std::pow(10.0, input(3)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match);
    
    // Compute cleavage probability and cleavage rate on the perfect-match
    // substrate
    T terminal_unbind_rate = 1;
    T terminal_cleave_rate = static_cast<T>(std::pow(10.0, input(4))); 
    T prob_perfect = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
    T rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 

    // Introduce one mismatch at the specified position and re-compute
    // cleavage probability and cleavage rate 
    model->setEdgeLabels(position, mismatch); 
    T prob_mismatched = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
    T rate_mismatched = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate);  

    // Compile results and return 
    VectorXd output(2);
    output << static_cast<double>(log10(prob_perfect) - log10(prob_mismatched)),
              static_cast<double>(log10(rate_perfect) - log10(rate_mismatched)); 

    delete model;
    return output;
}

/**
 * Return the template specialization of `computeCleavageStats()` corresponding
 * to the given mismatch position.
 *
 * @param position Mismatch position.
 * @returns Template specialization of `computeCleavageStats()`.
 */ 
template <typename T>
std::function<VectorXd(const Ref<const VectorXd>&)> getCleavageFunc(int position, int dim)
{
    switch (position)
    {
        case 0:
            return computeCleavageStats<PreciseType, 0>;
        case 1: 
            return computeCleavageStats<PreciseType, 1>;
        case 2:
            return computeCleavageStats<PreciseType, 2>; 
        case 3: 
            return computeCleavageStats<PreciseType, 3>; 
        case 4: 
            return computeCleavageStats<PreciseType, 4>; 
        case 5: 
            return computeCleavageStats<PreciseType, 5>; 
        case 6: 
            return computeCleavageStats<PreciseType, 6>; 
        case 7: 
            return computeCleavageStats<PreciseType, 7>; 
        case 8: 
            return computeCleavageStats<PreciseType, 8>; 
        case 9: 
            return computeCleavageStats<PreciseType, 9>; 
        case 10: 
            return computeCleavageStats<PreciseType, 10>; 
        case 11:
            return computeCleavageStats<PreciseType, 11>; 
        case 12: 
            return computeCleavageStats<PreciseType, 12>; 
        case 13: 
            return computeCleavageStats<PreciseType, 13>; 
        case 14: 
            return computeCleavageStats<PreciseType, 14>; 
        case 15:
            return computeCleavageStats<PreciseType, 15>; 
        case 16: 
            return computeCleavageStats<PreciseType, 16>; 
        case 17: 
            return computeCleavageStats<PreciseType, 17>; 
        case 18:
            return computeCleavageStats<PreciseType, 18>; 
        case 19:
            return computeCleavageStats<PreciseType, 19>; 
        default:
            throw std::invalid_argument("Invalid mismatch position");
    }
}

int main(int argc, char** argv)
{
    // Define trivial filtering function
    std::function<bool(const Ref<const VectorXd>&)> filter
        = [](const Ref<const VectorXd>& x)
        {
            return false;
        };

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

    // Check that required boundary-finding algorithm settings were specified
    if (!json_data.if_contains("n_init"))
        throw std::runtime_error("Initial sample size (n_init) must be specified");
    if (!json_data.if_contains("max_step_iter"))
        throw std::runtime_error("Maximum number of step iterations (max_step_iter) must be specified"); 
    if (!json_data.if_contains("max_pull_iter"))
        throw std::runtime_error("Maximum number of pull iterations (max_pull_iter) must be specified");
    if (!json_data.if_contains("tol"))
        throw std::runtime_error("Boundary-finding tolerance (tol) must be specified");
    if (!json_data.if_contains("mismatch_position"))
        throw std::runtime_error("Mismatch position (mismatch_position) must be specified");

    // Parse boundary-finding algorithm settings 
    int n_init, max_step_iter, max_pull_iter, mismatch_pos;  
    double tol;
    int min_step_iter = 0;
    int min_pull_iter = 0;
    int max_edges = 2000;
    int n_keep_interior = 10000;
    int n_keep_origbound = 10000;
    int n_mutate_origbound = 400;
    int n_pull_origbound = 400;
    bool verbose = true;
    bool traversal_verbose = true;  
    bool write_pulled_points = true;
    n_init = json_data["n_init"].as_int64(); 
    if (n_init <= 0)
    {
        throw std::runtime_error("Invalid initial sample size (n_init) specified");
    }
    max_step_iter = json_data["max_step_iter"].as_int64(); 
    if (max_step_iter <= 0)
    {
        throw std::runtime_error("Invalid maximum number of step iterations (max_step_iter) specified");
    } 
    max_pull_iter = json_data["max_pull_iter"].as_int64(); 
    if (max_pull_iter <= 0)
    {
        throw std::runtime_error("Invalid maximum number of pull iterations (max_pull_iter) specified");
    }
    tol = json_data["tol"].as_int64();
    if (tol <= 0)
    {
        throw std::runtime_error("Invalid boundary-finding tolerance (tol) specified");
    }
    mismatch_pos = json_data["mismatch_position"].as_int64(); 
    if (mismatch_position < 0 || mismatch_position > 19)
    {
        throw std::runtime_error("Invalid mismatch position (mismatch_position) specified");
    }
    if (json_data.if_contains("min_step_iter"))
    {
        min_step_iter = json_data["min_step_iter"].as_int64(); 
        if (min_step_iter <= 0)
            throw std::runtime_error("Invalid minimum number of step iterations (min_step_iter) specified"); 
    }
    if (json_data.if_contains("min_pull_iter"))
    {
        min_pull_iter = json_data["min_pull_iter"].as_int64(); 
        if (min_pull_iter <= 0)
            throw std::runtime_error("Invalid minimum number of pull iterations (min_pull_iter) specified"); 
    }
    if (json_data.if_contains("max_edges"))
    {
        max_edges = json_data["max_edges"].as_int64(); 
        if (max_edges <= 0)
            throw std::runtime_error("Invalid maximum number of edges (max_edges) specified"); 
    }
    if (json_data.if_contains("n_keep_interior"))
    {
        n_keep_interior = json_data["n_keep_interior"].as_int64(); 
        if (n_keep_interior <= 0)
            throw std::runtime_error(
                "Invalid number of interior points to keep per iteration (n_keep_interior) specified"
            );
    }
    if (json_data.if_contains("n_keep_origbound"))
    {
        n_keep_origbound = json_data["n_keep_origbound"].as_int64(); 
        if (n_keep_origbound <= 0)
            throw std::runtime_error(
                "Invalid number of unsimplified boundary vertices to keep per iteration (n_keep_origbound) specified"
            );
    }
    if (json_data.if_contains("n_mutate_origbound"))
    {
        n_mutate_origbound = json_data["n_mutate_origbound"].as_int64(); 
        if (n_mutate_origbound <= 0)
            throw std::runtime_error(
                "Invalid number of unsimplified boundary vertices to mutate per step iteration (n_mutate_origbound) specified"
            );
    }
    if (json_data.if_contains("n_pull_origbound"))
    {
        n_pull_origbound = json_data["n_pull_origbound"].as_int64(); 
        if (n_pull_origbound <= 0)
            throw std::runtime_error(
                "Invalid number of unsimplified boundary vertices to pull per pull iteration (n_pull_origbound) specified"
            );
    }
    if (json_data.if_contains("verbose"))
    {
        verbose = json_data["verbose"].as_bool(); 
    }
    if (json_data.if_contains("traversal_verbose"))
    {
        traversal_verbose = json_data["traversal_verbose"].as_bool(); 
    }
    if (json_data.if_contains("write_pulled_points"))
    {
        write_pulled_points = json_data["write_pulled_points"].as_bool();
    }

    // Parse SQP configurations
    int sqp_max_iter = 1000;   // 100? 
    double tau = 0.5;
    double delta = 1e-8; 
    double beta = 1e-4; 
    double sqp_tol = 1e-8;     // 1e-6?
    bool use_only_armijo = false;
    bool use_strong_wolfe = false;
    int hessian_modify_max_iter = 10000;
    double c1 = 1e-4;
    double c2 = 0.9;
    bool sqp_verbose = false;
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
            sqp_max_iter = sqp_data["max_iter"].as_int64(); 
            if (sqp_max_iter <= 0)
                throw std::runtime_error("Invalid value for maximum number of SQP iterations (max_iter) specified"); 
        }
        if (sqp_data.if_contains("tol"))
        {
            sqp_tol = static_cast<PreciseType>(sqp_data["tol"].as_double());
            if (sqp_tol <= 0)
                throw std::runtime_error("Invalid value for SQP tolerance (tol) specified"); 
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
            {
                std::stringstream ss_err; 
                ss_err << "Invalid value for maximum number of SQP Hessian modification "
                       << "iterations (hessian_modify_max_iter) specified";
                throw std::runtime_error(ss_err.str()); 
            } 
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
            sqp_verbose = sqp_data["verbose"].as_bool();
        }
    }
    std::stringstream ss;
    ss << outprefix << "-rapid-mm" << mismatch_pos << "-boundary";

    // Parse the given .poly file and store its contents as a string 
    std::ifstream infile(poly_filename);
    std::stringstream ss2;
    ss2 << infile.rdbuf(); 
    infile.close();

    // Initialize the boundary-finding algorithm
    std::size_t seed = 0; 
    boost::hash_combine(seed, 1234567890); 
    boost::hash_combine(seed, mismatch_pos);
    boost::hash_combine(seed, ss2.str()); 
    rng.seed(seed); 
    BoundaryFinder* finder = new BoundaryFinder(
        tol, rng, poly_filename, vert_filename,
        Polytopes::InequalityType::GreaterThanOrEqualTo
    );
    std::function<VectorXd(const Ref<const VectorXd>&)> func = getCleavageFunc<PreciseType>(mismatch_pos);
    finder->setFunc(func);  
    double mutate_delta = 0.1 * getMaxDist<double>(finder->getVertices());

    // Obtain the initial set of input points
    MatrixXd init_input = finder->sampleInput(n_init);

    // Run the boundary-finding algorithm  
    finder->run(
        mutate_delta, filter, init_input, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, sqp_max_iter, sqp_tol, max_edges, n_keep_interior,
        n_keep_origbound, n_mutate_origbound, n_pull_origbound, tau, delta,
        beta, use_only_armijo, use_strong_wolfe, hessian_modify_max_iter,
        ss.str(), RegularizationMethod::NOREG, 0, c1, c2, verbose, sqp_verbose,
        traversal_verbose, write_pulled_points 
    );

    delete finder;    
    return 0;
}
