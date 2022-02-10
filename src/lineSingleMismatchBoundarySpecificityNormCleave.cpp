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

/*
 * Estimates the boundary of the cleavage specificity vs. normalized cleavage
 * rate region in the line-graph Cas9 model. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     2/9/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
typedef number<mpfr_float_backend<100> > PreciseType; 

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

std::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(boost::random::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

/**
 * Compute the following quantities for the given set of parameter values:
 *
 * - cleavage specificity with respect to the single-mismatch substrate
 *   with the given mismatch position for the line graph
 * - normalized cleavage rate with respect to the single-mismatch substrate
 *   with the given mismatch position for the line graph for the line-graph
 *   Cas9 model. 
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
    for (unsigned j = 0; j < length; ++j)
        model->setEdgeLabels(j, match);
    
    // Compute cleavage probability and cleavage rate on the perfect-match
    // substrate
    T unbind_rate = 1;
    T cleave_rate = 1; 
    T prob_perfect = model->getUpperExitProb(unbind_rate, cleave_rate);
    T rate_perfect = model->getUpperExitRate(unbind_rate, cleave_rate); 

    // Introduce one mismatch at the specified position and re-compute
    // cleavage probability and cleavage rate 
    model->setEdgeLabels(position, mismatch); 
    T prob_mismatched = model->getUpperExitProb(unbind_rate, cleave_rate);
    T rate_mismatched = model->getUpperExitRate(unbind_rate, cleave_rate);  

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

/**
 * Mutate the given parameter values by delta = 0.1. 
 */
template <typename T>
Matrix<T, Dynamic, 1> mutateByDelta(const Ref<const Matrix<T, Dynamic, 1> >& input,
                                    boost::random::mt19937& rng)
{
    Matrix<T, Dynamic, 1> mutated(input);
    const T delta = 0.1;
    for (unsigned i = 0; i < mutated.size(); ++i)
    {
        int toss = coin_toss(rng);
        if (!toss) mutated(i) += delta;
        else       mutated(i) -= delta;
    }
    return mutated;
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
    const unsigned n_init = 5000; 
    const double tol = 1e-6;
    const unsigned min_step_iter = 100;
    const unsigned max_step_iter = 1000;
    const unsigned min_pull_iter = 10;
    const unsigned max_pull_iter = 50;
    const unsigned max_edges = 500;
    const bool verbose = true;
    const unsigned sqp_max_iter = 50;
    const double sqp_tol = 1e-3;
    const bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-spec-vs-cleave-mm" << argv[4] << "-boundary";

    // Initialize the boundary-finding algorithm
    const int position = std::stoi(argv[4]);
    std::function<VectorXd(const Ref<const VectorXd>&)> func = getCleavageFunc<PreciseType>(position); 
    BoundaryFinder<4> finder(tol, rng, argv[1], argv[2], func);
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate = mutateByDelta<double>;

    // Obtain the initial set of input points 
    MatrixXd init_input = finder.sampleInput(n_init); 

    // Run the boundary-finding algorithm 
    finder.run(
        mutate, filter, init_input, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, max_edges, verbose, sqp_max_iter, sqp_tol, sqp_verbose,
        ss.str()
    );
    MatrixXd final_input = finder.getInput(); 

    // Write final set of input points to file 
    std::ostringstream oss;
    oss << argv[3] << "-spec-vs-cleave-mm" << argv[4] << "-boundary-input.tsv";
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (samplefile.is_open())
    {
        for (unsigned i = 0; i < final_input.rows(); i++)
        {
            for (unsigned j = 0; j < final_input.cols() - 1; j++)
            {
                samplefile << final_input(i, j) << "\t";
            }
            samplefile << final_input(i, final_input.cols() - 1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());
    
    return 0;
}
