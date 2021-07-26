#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/random.hpp>
#include <boundaryFinder.hpp>
#include "../include/graphs/line.hpp"

/*
 * Estimates the boundary of the cleavage activity vs. cleavage specificity
 * region in the line Cas9 model.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     5/3/2021
 */
using namespace Eigen;
using boost::math::constants::ln_ten; 
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

std::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(boost::random::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

template <typename T, unsigned n_mismatches>
VectorXd computeCleavageStats(const Ref<const VectorXd>& params)
{
    /*
     * Compute the specificity and speed ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values.
     */
    // Array of DNA/RNA match parameters
    std::array<T, 2> match_params;
    match_params[0] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[1] = static_cast<T>(std::pow(10.0, params(1)));

    // Array of DNA/RNA mismatch parameters
    std::array<T, 2> mismatch_params;
    mismatch_params[0] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[1] = static_cast<T>(std::pow(10.0, params(3)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T>* model = new LineGraph<T>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setLabels(j, match_params);
    
    // Compute cleavage probability without any mismatches present
    T prob_perfect = model->computeUpperExitProb(1, 1);

    // Introduce distal mismatches and re-compute cleavage probability
    for (unsigned j = 1; j <= n_mismatches; ++j)
        model->setLabels(length - j, mismatch_params);
    T prob = model->computeUpperExitProb(1, 1); 

    // Compile results and return 
    VectorXd output(2);
    output << static_cast<double>(prob_perfect / ln_ten<T>()),
              static_cast<double>((prob_perfect - prob) / ln_ten<T>());

    delete model;
    return output;
}

template <typename T>
std::function<VectorXd(const Ref<const VectorXd>&)> cleavageFunc(unsigned n_mismatches)
{
    /*
     * Return the function instance with the given number of mismatches.
     */
    std::function<VectorXd(const Ref<const VectorXd>&)> func;
    switch (n_mismatches)
    {
        case 1:
            func = computeCleavageStats<T, 1>;
            break;
        case 2:
            func = computeCleavageStats<T, 2>;
            break;
        case 3:
            func = computeCleavageStats<T, 3>;
            break;
        case 4:
            func = computeCleavageStats<T, 4>;
            break;
        case 5:
            func = computeCleavageStats<T, 5>;
            break;
        case 6:
            func = computeCleavageStats<T, 6>;
            break;
        case 7:
            func = computeCleavageStats<T, 7>;
            break;
        case 8:
            func = computeCleavageStats<T, 8>;
            break;
        case 9:
            func = computeCleavageStats<T, 9>;
            break;
        case 10:
            func = computeCleavageStats<T, 10>;
            break;
        case 11:
            func = computeCleavageStats<T, 11>;
            break;
        case 12:
            func = computeCleavageStats<T, 12>;
            break;
        case 13:
            func = computeCleavageStats<T, 13>;
            break;
        case 14:
            func = computeCleavageStats<T, 14>;
            break;
        case 15:
            func = computeCleavageStats<T, 15>;
            break;
        case 16:
            func = computeCleavageStats<T, 16>;
            break;
        case 17:
            func = computeCleavageStats<T, 17>;
            break;
        case 18:
            func = computeCleavageStats<T, 18>;
            break;
        case 19:
            func = computeCleavageStats<T, 19>;
            break;
        case 20:
            func = computeCleavageStats<T, 20>;
            break;
        default:
            break;
    }
    return func;
}

template <typename T>
Matrix<T, Dynamic, 1> mutateByDelta(const Ref<const Matrix<T, Dynamic, 1> >& params, boost::random::mt19937& rng)
{
    /*
     * Mutate the given parameter values by delta = 0.1. 
     */
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
    // Number of mismatches 
    unsigned m;
    sscanf(argv[4], "%u", &m);

    // Define filtering function (no filtering performed at all!)
    std::function<bool(const Ref<const VectorXd>& x)> filter
        = [](const Ref<const VectorXd>& x)
        {
            return false;
        };

    // Boundary-finding algorithm settings
    double tol = 1e-8;
    unsigned n_within = 5000;
    unsigned n_bound = 0;
    unsigned min_step_iter = 100;
    unsigned max_step_iter = 500;
    unsigned min_pull_iter = 10;
    unsigned max_pull_iter = 100;
    unsigned max_edges = 200;
    bool verbose = true;
    unsigned sqp_max_iter = 100;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-boundary";

    // Run the boundary-finding algorithm
    BoundaryFinder finder(tol, rng, argv[1], argv[2]);
    std::function<VectorXd(const Ref<const VectorXd>&)> func
        = cleavageFunc<number<mpfr_float_backend<100> > >(m);
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate
        = mutateByDelta<double>;
    finder.run(
        func, mutate, filter, n_within, n_bound, min_step_iter, max_step_iter,
        min_pull_iter, max_pull_iter, max_edges, verbose, sqp_max_iter,
        sqp_tol, sqp_verbose, ss.str()
    );
    MatrixXd params = finder.getParams();

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[3] << "-boundary-params.tsv";
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (samplefile.is_open())
    {
        for (unsigned i = 0; i < params.rows(); i++)
        {
            for (unsigned j = 0; j < params.cols() - 1; j++)
            {
                samplefile << params(i,j) << "\t";
            }
            samplefile << params(i,params.cols()-1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());
    
    return 0;
}
