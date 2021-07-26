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
#include "../include/graphs/gridMatchMismatch.hpp"

/*
 * Estimates the boundary of the cleavage specificity vs. conditional unbinding 
 * specificity region in the grid Cas9 model. 
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
    // Array of DNA/RNA match/mismatch parameters
    std::array<T, 6> exp_params;
    for (unsigned i = 0; i < 6; ++i)
        exp_params[i] = static_cast<T>(std::pow(10.0, params(i)));

    // Populate each rung with DNA/RNA match parameters
    GridMatchMismatchGraph<T>* model = new GridMatchMismatchGraph<T>(length);
    model->setStartLabels(exp_params[4], exp_params[5]);
    model->setMatchForwardLabel(exp_params[0]);
    model->setMatchReverseLabel(exp_params[1]);
    model->setMismatchForwardLabel(exp_params[2]);
    model->setMismatchReverseLabel(exp_params[3]);
    for (unsigned i = 0; i < length; ++i)    // Start with perfectly matched substrate
        model->setRungLabels(i, true);
    
    // Compute conditional mean first passage times to either exit without
    // any mismatches present
    std::tuple<T, T, T> data = model->computeExitStats(1, 1, 1, 1);
    T condrate_lower_perfect = std::get<1>(data);
    T condrate_upper_perfect = std::get<2>(data);

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned i = 1; i <= n_mismatches; ++i)
        model->setRungLabels(length - i, false);
    data = model->computeExitStats(1, 1, 1, 1);
    T condrate_lower = std::get<1>(data);
    T condrate_upper = std::get<2>(data);
    
    // Compile results and return 
    VectorXd output(2);
    output << static_cast<double>((condrate_upper_perfect - condrate_upper) / ln_ten<T>()), 
              static_cast<double>((condrate_lower_perfect - condrate_lower) / ln_ten<T>());

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

    // Define filtering function
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
    std::function<VectorXd(const Ref<const VectorXd>&)> func = cleavageFunc<number<mpfr_float_backend<100> > >(m);
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate = mutateByDelta<double>;
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
