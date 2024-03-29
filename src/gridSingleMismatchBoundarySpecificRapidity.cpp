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
#include <graphs/grid.hpp>

/*
 * Estimates the boundary of the cleavage specificity vs. normalized cleavage 
 * rate region in the grid-graph Cas9 model. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     5/16/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10; 
typedef number<mpfr_float_backend<100> > PreciseType; 

const unsigned length = 20;

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
 * - cleavage specificity with respect to the single-mismatch substrate 
 *   with the given mismatch position and
 * - normalized cleavage rate with respect to the single-mismatch substrate 
 *   with the given mismatch position for the grid-graph Cas9 model.
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

    // Array of conformational change parameters
    std::pair<T, T> conf_switch = std::make_pair(
        static_cast<T>(std::pow(10.0, input(4))), 
        static_cast<T>(std::pow(10.0, input(5)))
    ); 

    // Populate each rung with DNA/RNA match parameters
    GridGraph<T, T>* model = new GridGraph<T, T>(length);
    std::array<T, 6> labels; 
    model->setZerothLabels(conf_switch.first, conf_switch.second);
    for (unsigned j = 0; j < length; ++j)
    {
        labels[0] = match.first; 
        labels[1] = match.second; 
        labels[2] = match.first; 
        labels[3] = match.second; 
        labels[4] = conf_switch.first; 
        labels[5] = conf_switch.second; 
        model->setRungLabels(j, labels); 
    } 
    
    // Compute cleavage probability and cleavage rate on the perfect-match
    // substrate
    T unbind_rate = 1;
    T cleave_rate = 1; 
    auto result = model->getExitStats(unbind_rate, cleave_rate);
    T prob_perfect = std::get<0>(result); 
    T rate_perfect = std::get<2>(result); 

    // Introduce one mismatch at the specified position and re-compute
    // cleavage probability and cleavage rate
    labels[0] = mismatch.first; 
    labels[1] = mismatch.second; 
    labels[2] = mismatch.first; 
    labels[3] = mismatch.second; 
    model->setRungLabels(position, labels); 
    result = model->getExitStats(unbind_rate, cleave_rate);
    T prob_mismatched = std::get<0>(result); 
    T rate_mismatched = std::get<2>(result); 

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
 */ 
template <typename T>
std::function<VectorXd(const Ref<const VectorXd>&)> getCleavageFunc(int position)
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
    std::function<bool(const Ref<const VectorXd>& x)> filter
        = [](const Ref<const VectorXd>& x)
        {
            return false;
        };

    // Boundary-finding algorithm settings
    const unsigned n_init = 1000; 
    const double tol = 1e-6;
    const unsigned min_step_iter = 10;
    const unsigned max_step_iter = 100;
    const unsigned min_pull_iter = 10;
    const unsigned max_pull_iter = 100;
    const unsigned sqp_max_iter = 100; 
    const double sqp_tol = 1e-6; 
    const unsigned max_edges = 300;
    const double tau = 0.5;
    const double delta = 1e-8; 
    const double beta = 1e-4;
    const bool use_only_armijo = false;
    const bool use_strong_wolfe = false; 
    const unsigned hessian_modify_max_iter = 10000;
    const double c1 = 1e-4;
    const double c2 = 0.9;
    const bool verbose = true; 
    const bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-spec-rapidity-mm" << argv[4] << "-boundary";

    // Initialize the boundary-finding algorithm
    const int position = std::stoi(argv[4]);
    std::function<VectorXd(const Ref<const VectorXd>&)> func = getCleavageFunc<PreciseType>(position); 
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
        max_pull_iter, sqp_max_iter, sqp_tol, max_edges, tau, delta, beta,
        use_only_armijo, use_strong_wolfe, hessian_modify_max_iter, ss.str(),
        c1, c2, verbose, sqp_verbose
    );

    delete finder;    
    return 0;
}
