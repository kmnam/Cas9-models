#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <stdexcept>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <boost/random.hpp>
#include <boundaryFinder.hpp>
#include "../include/graphs/triangularPrism.hpp"

/*
 * Estimates the boundary of the cleavage specificity vs. unbinding
 * specificity region in the triangular-prism Cas9 model.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     5/8/2021
 */
using namespace Eigen;
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
VectorXd computeStats(const Ref<const VectorXd>& params)
{
    /*
     * Compute the specificity and speed ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    // Array of DNA/RNA match parameters
    std::array<T, 12> match_params;
    match_params[0] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[1] = static_cast<T>(std::pow(10.0, params(1)));
    match_params[2] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[3] = static_cast<T>(std::pow(10.0, params(1)));
    match_params[4] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[5] = static_cast<T>(std::pow(10.0, params(1)));
    match_params[6] = static_cast<T>(std::pow(10.0, params(4)));
    match_params[7] = static_cast<T>(std::pow(10.0, params(5)));
    match_params[8] = static_cast<T>(std::pow(10.0, params(6)));
    match_params[9] = static_cast<T>(std::pow(10.0, params(7)));
    match_params[10] = static_cast<T>(std::pow(10.0, params(8)));
    match_params[11] = static_cast<T>(std::pow(10.0, params(9)));

    // Array of DNA/RNA mismatch parameters
    std::array<T, 12> mismatch_params;
    mismatch_params[0] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[1] = static_cast<T>(std::pow(10.0, params(3)));
    mismatch_params[2] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[3] = static_cast<T>(std::pow(10.0, params(3)));
    mismatch_params[4] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[5] = static_cast<T>(std::pow(10.0, params(3)));
    mismatch_params[6] = static_cast<T>(std::pow(10.0, params(4)));
    mismatch_params[7] = static_cast<T>(std::pow(10.0, params(5)));
    mismatch_params[8] = static_cast<T>(std::pow(10.0, params(6)));
    mismatch_params[9] = static_cast<T>(std::pow(10.0, params(7)));
    mismatch_params[10] = static_cast<T>(std::pow(10.0, params(8)));
    mismatch_params[11] = static_cast<T>(std::pow(10.0, params(9)));

    // Populate each rung with DNA/RNA match parameters
    TriangularPrismGraph<T>* model = new TriangularPrismGraph<T>(length);
    model->setStartLabels(
        match_params[6], match_params[7], match_params[8], match_params[9],
        match_params[10], match_params[11]
    );
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    std::tuple<T, T, T> data = model->computeExitStats(1, 1, 1, 0);
    T prob_perfect = std::get<0>(data);
    T rate_perfect = std::get<1>(data);
   
    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= n_mismatches; ++j)
        model->setRungLabels(length - j, mismatch_params);
    data = model->computeExitStats(1, 1, 1, 0);
    T prob = std::get<0>(data);
    T rate = std::get<1>(data);

    // Compute the specificity and speed ratio
    double cleave_spec = static_cast<double>(prob_perfect - prob);
    double unbind_spec = static_cast<double>(rate_perfect - rate);
    VectorXd output(2);
    output << cleave_spec, unbind_spec;

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
            func = computeStats<T, 1>;
            break;
        case 2:
            func = computeStats<T, 2>;
            break;
        case 3:
            func = computeStats<T, 3>;
            break;
        case 4:
            func = computeStats<T, 4>;
            break;
        case 5:
            func = computeStats<T, 5>;
            break;
        case 6:
            func = computeStats<T, 6>;
            break;
        case 7:
            func = computeStats<T, 7>;
            break;
        case 8:
            func = computeStats<T, 8>;
            break;
        case 9:
            func = computeStats<T, 9>;
            break;
        case 10:
            func = computeStats<T, 10>;
            break;
        case 11:
            func = computeStats<T, 11>;
            break;
        case 12:
            func = computeStats<T, 12>;
            break;
        case 13:
            func = computeStats<T, 13>;
            break;
        case 14:
            func = computeStats<T, 14>;
            break;
        case 15:
            func = computeStats<T, 15>;
            break;
        case 16:
            func = computeStats<T, 16>;
            break;
        case 17:
            func = computeStats<T, 17>;
            break;
        case 18:
            func = computeStats<T, 18>;
            break;
        case 19:
            func = computeStats<T, 19>;
            break;
        case 20:
            func = computeStats<T, 20>;
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
            return x(0) < 0.01;
        };

    // Boundary-finding algorithm settings
    double tol = 1e-8;
    unsigned n_sample = 1000;
    unsigned min_step_iter = 100;
    unsigned max_step_iter = 200;
    unsigned min_pull_iter = 10;
    unsigned max_pull_iter = 50;
    unsigned max_edges = 100;
    bool verbose = true;
    unsigned sqp_max_iter = 50;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    unsigned nchains = 5;
    double warmup = 0.5;
    unsigned ntrials = 50;
    std::stringstream ss;
    ss << argv[3] << "-boundary";

    // Run the boundary-finding algorithm
    BoundaryFinder finder(tol, rng, argv[1]);
    std::function<VectorXd(const Ref<const VectorXd>&)> func = cleavageFunc<number<mpfr_float_backend<200> > >(m);
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate
        = mutateByDelta<double>;
    finder.runFromRandomWalk(
        func, mutate, filter, n_sample, min_step_iter, max_step_iter,
        min_pull_iter, max_pull_iter, max_edges, verbose, sqp_max_iter,
        sqp_tol, sqp_verbose, ss.str(), nchains, tol, warmup, ntrials
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
