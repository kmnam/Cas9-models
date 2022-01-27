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
#include <boundaryFinder.hpp>
#include <graphs/line.hpp>
#include <boostMultiprecisionEigen.hpp>

/*
 * Estimates the boundary of the cleavage activity vs. cleavage specificity
 * region in the line Cas9 model. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     1/27/2022
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
 * Compute:
 * - cleavage activity on the perfect-match substrate and
 * - cleavage specificity with respect to the single-mismatch substrate
 *   with the given mismatch position for the line graph with the given 
 *   set of parameter values. 
 */
template <typename T, int position>
VectorXd computeCleavageStats(const Ref<const VectorXd>& params)
{
    // Array of DNA/RNA match parameters
    std::pair<T, T> match_params;
    match_params.first = static_cast<T>(std::pow(10.0, params(0)));
    match_params.second = static_cast<T>(std::pow(10.0, params(1)));

    // Array of DNA/RNA mismatch parameters
    std::pair<T, T> mismatch_params;
    mismatch_params.first = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params.second = static_cast<T>(std::pow(10.0, params(3)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_params);
    
    // Compute cleavage probability on the perfect-match substrate
    T unbind_rate = 1;
    T cleave_rate = 1; 
    T prob_perfect = model->getUpperExitProb(unbind_rate, cleave_rate);

    // Introduce one mismatch at the specified position and re-compute
    // cleavage probability
    model->setEdgeLabels(position, mismatch_params); 
    T prob_mismatched = model->getUpperExitProb(unbind_rate, cleave_rate); 

    // Compile results and return 
    VectorXd output(2);
    output << static_cast<double>(prob_perfect),
              static_cast<double>(log10(prob_perfect) - log10(prob_mismatched)); 

    delete model;
    return output;
}

/**
 * Mutate the given parameter values by delta = 0.1. 
 */
template <typename T>
Matrix<T, Dynamic, 1> mutateByDelta(const Ref<const Matrix<T, Dynamic, 1> >& params, boost::random::mt19937& rng)
{
    Matrix<T, Dynamic, 1> mutated(params);
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
    // Define filtering function
    std::function<bool(const Ref<const VectorXd>& x)> filter
        = [](const Ref<const VectorXd>& x)
        {
            return false;
        };

    // Boundary-finding algorithm settings
    double tol = 1e-6;
    unsigned n_within = 100;
    unsigned n_bound = 0;
    unsigned min_step_iter = 100;
    unsigned max_step_iter = 200;
    unsigned min_pull_iter = 10;
    unsigned max_pull_iter = 50;
    unsigned max_edges = 500;
    bool verbose = true;
    unsigned sqp_max_iter = 50;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-activity-spec" << argv[4] << "-boundary";

    // Run the boundary-finding algorithm
    BoundaryFinder finder(tol, rng, argv[1], argv[2]);
    int position = std::stoi(argv[4]); 
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate = mutateByDelta<double>;
    std::function<VectorXd(const Ref<const VectorXd>&)> func; 
    switch (position)
    {
        case 0:
            func = computeCleavageStats<PreciseType, 0>;
            break;
        case 1: 
            func = computeCleavageStats<PreciseType, 1>;
            break; 
        case 2:
            func = computeCleavageStats<PreciseType, 2>; 
            break; 
        case 3: 
            func = computeCleavageStats<PreciseType, 3>; 
            break; 
        case 4: 
            func = computeCleavageStats<PreciseType, 4>; 
            break; 
        case 5: 
            func = computeCleavageStats<PreciseType, 5>; 
            break; 
        case 6: 
            func = computeCleavageStats<PreciseType, 6>; 
            break; 
        case 7: 
            func = computeCleavageStats<PreciseType, 7>; 
            break; 
        case 8: 
            func = computeCleavageStats<PreciseType, 8>; 
            break; 
        case 9: 
            func = computeCleavageStats<PreciseType, 9>; 
            break; 
        case 10: 
            func = computeCleavageStats<PreciseType, 10>; 
            break; 
        case 11:
            func = computeCleavageStats<PreciseType, 11>; 
            break; 
        case 12: 
            func = computeCleavageStats<PreciseType, 12>; 
            break; 
        case 13: 
            func = computeCleavageStats<PreciseType, 13>; 
            break; 
        case 14: 
            func = computeCleavageStats<PreciseType, 14>; 
            break; 
        case 15:
            func = computeCleavageStats<PreciseType, 15>; 
            break; 
        case 16: 
            func = computeCleavageStats<PreciseType, 16>; 
            break;
        case 17: 
            func = computeCleavageStats<PreciseType, 17>; 
            break; 
        case 18:
            func = computeCleavageStats<PreciseType, 18>; 
            break; 
        case 19:
            func = computeCleavageStats<PreciseType, 19>; 
            break; 
        default:
            break; 
    }
    finder.run(
        func, mutate, filter, n_within, n_bound, min_step_iter, max_step_iter,
        min_pull_iter, max_pull_iter, max_edges, verbose, sqp_max_iter,
        sqp_tol, sqp_verbose, ss.str()
    );
    MatrixXd params = finder.getParams();

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[3] << "-activity-spec" << argv[4] << "-boundary-params.tsv";
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (samplefile.is_open())
    {
        for (unsigned i = 0; i < params.rows(); i++)
        {
            for (unsigned j = 0; j < params.cols() - 1; j++)
            {
                samplefile << params(i, j) << "\t";
            }
            samplefile << params(i, params.cols()-1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());
    
    return 0;
}
