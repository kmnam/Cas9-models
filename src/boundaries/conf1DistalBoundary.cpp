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
#include "../../include/graphs/line.hpp"

/*
 * Estimates the boundary of the specificity vs. speed ratio region in 
 * the single-conformation Cas9 model (line graph).  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     1/15/2020
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::et_off;
typedef number<mpfr_float_backend<30>, et_off> mpfr_30_noet;
typedef Matrix<mpfr_30_noet, Dynamic, Dynamic> MatrixX30; 
typedef Matrix<mpfr_30_noet, Dynamic, 1> VectorX30;

const unsigned length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

boost::random::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(boost::random::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

template <unsigned n_mismatches = 1>
VectorX30 computeCleavageStats(const Ref<const VectorX30>& params)
{
    /*
     * Compute the specificity and speed ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    using boost::multiprecision::pow;

    // Array of DNA/RNA match parameters
    std::array<mpfr_30_noet, 2> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(1));

    // Array of DNA/RNA mismatch parameters
    std::array<mpfr_30_noet, 2> mismatch_params;
    mismatch_params[0] = pow(10.0, params(2));
    mismatch_params[1] = pow(10.0, params(3));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<mpfr_30_noet>* model = new LineGraph<mpfr_30_noet>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<mpfr_30_noet, 2, 1> match_data = model->computeCleavageStats(1, 1000).array().log10().matrix();

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= n_mismatches; ++j)
        model->setLabels(length - j, mismatch_params);
    Matrix<mpfr_30_noet, 2, 1> mismatch_data = model->computeCleavageStats(1, 1000).array().log10().matrix();

    // Compute the specificity and speed ratio
    mpfr_30_noet specificity = match_data(0) - mismatch_data(0);
    mpfr_30_noet speed_ratio = mismatch_data(1) - match_data(1);
    VectorX30 stats(2);
    stats << specificity, speed_ratio;

    delete model;
    return stats;
}

std::function<VectorX30(const Ref<const VectorX30>&)> cleavageFunc(unsigned n_mismatches)
{
    /*
     * Return the function instance with the given number of mismatches.
     */
    std::function<VectorX30(const Ref<const VectorX30>&)> func;
    switch (n_mismatches)
    {
        case 0:
            func = computeCleavageStats<0>;
            break;
        case 1:
            func = computeCleavageStats<1>;
            break;
        case 2:
            func = computeCleavageStats<2>;
            break;
        case 3:
            func = computeCleavageStats<3>;
            break;
        case 4:
            func = computeCleavageStats<4>;
            break;
        case 5:
            func = computeCleavageStats<5>;
            break;
        case 6:
            func = computeCleavageStats<6>;
            break;
        case 7:
            func = computeCleavageStats<7>;
            break;
        case 8:
            func = computeCleavageStats<8>;
            break;
        case 9:
            func = computeCleavageStats<9>;
            break;
        case 10:
            func = computeCleavageStats<10>;
            break;
        case 11:
            func = computeCleavageStats<11>;
            break;
        case 12:
            func = computeCleavageStats<12>;
            break;
        case 13:
            func = computeCleavageStats<13>;
            break;
        case 14:
            func = computeCleavageStats<14>;
            break;
        case 15:
            func = computeCleavageStats<15>;
            break;
        case 16:
            func = computeCleavageStats<16>;
            break;
        case 17:
            func = computeCleavageStats<17>;
            break;
        case 18:
            func = computeCleavageStats<18>;
            break;
        case 19:
            func = computeCleavageStats<19>;
            break;
        case 20:
            func = computeCleavageStats<20>;
            break;
        default:
            break;
    }
    return func;
}

VectorX30 mutate_by_delta(const Ref<const VectorX30>& params, boost::random::mt19937& rng)
{
    /*
     * Mutate the given parameter values by delta = 0.1. 
     */
    VectorX30 mutated(params);
    const mpfr_30_noet delta = 0.1;
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
    // Sample model parameter combinations
    unsigned m;
    sscanf(argv[4], "%u", &m);

    // Define trivial filtering function
    std::function<bool(const Ref<const VectorX30>& x)> filter
        = [](const Ref<const VectorX30>& x){ return false; };

    // Boundary-finding algorithm settings
    double tol = 1e-8;
    unsigned n_within = 200;
    unsigned n_bound = 200;
    unsigned min_step_iter = 100;
    unsigned max_step_iter = 500;
    unsigned min_pull_iter = 10;
    unsigned max_pull_iter = 50;
    unsigned max_edges = 50;
    bool verbose = true;
    unsigned sqp_max_iter = 50;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-boundary";

    // Run the boundary-finding algorithm
    BoundaryFinder<mpfr_30_noet> finder(tol, rng, argv[1], argv[2]);
    std::function<VectorX30(const Ref<const VectorX30>&)> func = cleavageFunc(m);
    std::function<VectorX30(const Ref<const VectorX30>&, boost::random::mt19937&)> mutate = mutate_by_delta;
    finder.run(
        func, mutate, filter, n_within, n_bound, min_step_iter, max_step_iter, min_pull_iter,
        max_pull_iter, max_edges, verbose, sqp_max_iter, sqp_tol, sqp_verbose, ss.str()
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
