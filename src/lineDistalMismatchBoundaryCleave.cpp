#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <boostMultiprecisionEigen.hpp>
#include <boundaryFinder.hpp>
#include <graphs/line.hpp>

/**
 * Estimates the boundary of the cleavage rate vs. *normalized* cleavage rate
 * region in the line-graph Cas9 model. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     7/23/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
typedef number<mpfr_float_backend<100> > PreciseType; 

const int length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

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
 * Compute the following quantities for the given set of parameter values:
 *
 * - cleavage rate on the perfect-match substrate and
 * - normalized cleavage rate with respect to the distal-mismatch substrate 
 *   with the given number of mismatches for the line-graph Cas9 model. 
 */
template <typename T, int num_mismatches>
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
    
    // Compute cleavage rate on the perfect-match substrate 
    T terminal_unbind_rate = static_cast<T>(std::pow(10.0, input(4)));
    T terminal_cleave_rate = static_cast<T>(std::pow(10.0, input(5)));
    T rate_perfect = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate);  

    // Introduce the specified number of distal mismatches and re-compute 
    // cleavage rate 
    for (int j = 0; j < num_mismatches; ++j)
        model->setEdgeLabels(19 - j, mismatch); 
    T rate_mismatched = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 

    // Compile results and return 
    VectorXd output(2);
    output << static_cast<double>(log10(rate_perfect)),
              static_cast<double>(log10(rate_perfect) - log10(rate_mismatched)); 

    delete model;
    return output;
}

/**
 * Return the template specialization of `computeCleavageStats()` corresponding
 * to the given number of mismatches. 
 */ 
template <typename T>
std::function<VectorXd(const Ref<const VectorXd>&)> getCleavageFunc(int num_mismatches)
{
    switch (num_mismatches)
    {
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
        case 20:
            return computeCleavageStats<PreciseType, 20>;  
        default:
            throw std::invalid_argument("Invalid number of mismatches"); 
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

    // Boundary-finding algorithm settings
    const int n_init = 20000; 
    const double tol = 1e-6;
    const int min_step_iter = 10;
    const int max_step_iter = 100;
    const int min_pull_iter = 10;
    const int max_pull_iter = 100;
    const int sqp_max_iter = 100;
    const double sqp_tol = 1e-6;
    const int max_edges = 500;
    const int psi = 100;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const bool use_only_armijo = false; 
    const bool use_strong_wolfe = false;
    const int hessian_modify_max_iter = 10000; 
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const bool verbose = true;
    const bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-cleave-mm" << argv[4] << "-boundary";

    // Initialize the boundary-finding algorithm
    const int num_mismatches = std::stoi(argv[4]); 
    std::function<VectorXd(const Ref<const VectorXd>&)> func = getCleavageFunc<PreciseType>(num_mismatches);
    BoundaryFinder* finder = new BoundaryFinder(
        tol, rng, argv[1], argv[2],
        Polytopes::InequalityType::GreaterThanOrEqualTo, func
    );
    double mutate_delta = 0.1 * getMaxDist<double>(finder->getVertices());

    // Obtain the initial set of input points
    MatrixXd init_input = finder->sampleInput(n_init);

    // Run the boundary-finding algorithm  
    finder->run(
        mutate_delta, filter, init_input, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, sqp_max_iter, sqp_tol, max_edges, psi, tau, delta, beta,
        use_only_armijo, use_strong_wolfe, hessian_modify_max_iter, ss.str(),
        RegularizationMethod::NOREG, 0, c1, c2, verbose, sqp_verbose 
    );

    delete finder;    
    return 0;
}
