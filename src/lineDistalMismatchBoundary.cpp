#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/random.hpp>
#include <boundaryFinder.hpp>
#include "../include/graphs/line.hpp"

/*
 * Estimates the boundary of the cleavage/unbinding specificity region in 
 * the line Cas9 model.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     4/13/2021
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

std::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(boost::random::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

template <typename T, unsigned n_mismatches>
Matrix<double, Dynamic, 1> computeCleavageStats(const Ref<const Matrix<double, Dynamic, 1> >& params)
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
    
    // Compute cleavage probability and mean first passage time 
    double prob_perfect = static_cast<double>(log10(model->computeUpperExitProb(1, 1)));
    double rate_perfect = static_cast<double>(log10(model->computeLowerExitRate(1, 0)));
    
    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= n_mismatches; ++j)
        model->setLabels(length - j, mismatch_params);
    double prob = static_cast<double>(log10(model->computeUpperExitProb(1, 1)));
    double rate = static_cast<double>(log10(model->computeLowerExitRate(1, 0)));

    // Compute the specificity and speed ratio
    double cleave_spec = prob_perfect - prob;
    double unbind_spec = rate - rate_perfect;
    Matrix<double, Dynamic, 1> output(2);
    output << cleave_spec, unbind_spec;

    delete model;
    return output;
}

template <typename T>
std::function<Matrix<double, Dynamic, 1>(const Ref<const Matrix<double, Dynamic, 1> >&)> cleavageFunc(unsigned n_mismatches)
{
    /*
     * Return the function instance with the given number of mismatches.
     */
    std::function<Matrix<double, Dynamic, 1>(const Ref<const Matrix<double, Dynamic, 1> >&)> func;
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

    // Define filtering function
    std::function<bool(const Ref<const Matrix<double, Dynamic, 1> >& x)> filter
        = [](const Ref<const Matrix<double, Dynamic, 1> >& x)
        {
            return x(0) < 0.01;
        };

    // Boundary-finding algorithm settings
    double tol = 1e-8;
    unsigned n_within = 1000;
    unsigned n_bound = 0;
    unsigned min_step_iter = 100;
    unsigned max_step_iter = 200;
    unsigned min_pull_iter = 10;
    unsigned max_pull_iter = 50;
    unsigned max_edges = 100;
    bool verbose = true;
    unsigned sqp_max_iter = 50;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-boundary";

    // Run the boundary-finding algorithm
    BoundaryFinder finder(tol, rng, argv[1], argv[2]);
    std::function<Matrix<double, Dynamic, 1>(const Ref<const Matrix<double, Dynamic, 1> >&)> func
        = cleavageFunc<number<mpfr_float_backend<50> > >(m);
    std::function<Matrix<double, Dynamic, 1>(const Ref<const Matrix<double, Dynamic, 1> >&, boost::random::mt19937&)> mutate
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
