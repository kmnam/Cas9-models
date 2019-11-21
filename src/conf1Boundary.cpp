#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <random>
#include <Eigen/Dense>
#include <boundaryFinder.hpp>
#include <duals/duals.hpp>
#include <duals/eigen.hpp>
#include "../include/graphs/line.hpp"
#include "../include/sample.hpp"

/*
 * Estimates the boundary of the specificity vs. speed ratio region in 
 * the single-conformation Cas9 model (line graph).  
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/21/2019
 */
using namespace Eigen;
using Duals::DualNumber;

const unsigned length = 20;

// Instantiate random number generator 
std::mt19937 rng(1234567890);

std::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(std::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

template <unsigned n_mismatches = 1>
VectorXDual computeCleavageStats(const Ref<const VectorXDual>& params)
{
    /*
     * Compute the specificity and speed ratio with respect to the 
     * given number of mismatches, with the given set of parameter
     * values. 
     */
    using Duals::pow;

    // Array of DNA/RNA match parameters
    std::array<DualNumber, 2> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(1));

    // Array of DNA/RNA mismatch parameters
    std::array<DualNumber, 2> mismatch_params;
    mismatch_params[0] = pow(10.0, params(2));
    mismatch_params[1] = pow(10.0, params(3));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<DualNumber>* model = new LineGraph<DualNumber>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<DualNumber, 2, 1> match_data = model->computeCleavageStats(1, 1).array().log10().matrix();

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= n_mismatches; ++j)
        model->setLabels(length - j, mismatch_params);
    Matrix<DualNumber, 2, 1> mismatch_data = model->computeCleavageStats(1, 1).array().log10().matrix();

    // Compute the specificity and speed ratio
    DualNumber specificity = match_data(0) - mismatch_data(0);
    DualNumber speed_ratio = mismatch_data(1) - match_data(1);
    VectorXDual stats(2);
    stats << specificity, speed_ratio;

    delete model;
    return stats;
}

std::function<VectorXDual(const Ref<const VectorXDual>&)> cleavageFunc(unsigned n_mismatches)
{
    /*
     * Return the function instance with the given number of mismatches.
     */
    std::function<VectorXDual(const Ref<const VectorXDual>&)> func;
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

VectorXDual mutate_by_delta(const Ref<const VectorXDual>& params, std::mt19937& rng)
{
    /*
     * Mutate the given parameter values by delta = 0.1. 
     */
    VectorXDual mutated(params);
    const DualNumber delta = 0.1;
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
    unsigned n, m;
    sscanf(argv[4], "%u", &m);
    sscanf(argv[5], "%u", &n);
    MatrixXd params(n, 4);
    try
    {
        params = sampleFromConvexPolytopeTriangulation(argv[2], n, rng);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    // Boundary-finding algorithm settings
    double tol = 1e-4;
    unsigned max_step_iter = 10;
    unsigned max_pull_iter = 5;
    bool simplify = true;
    bool verbose = true;
    unsigned sqp_max_iter = 50;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-boundary";

    // Run the boundary-finding algorithm
    LinearConstraints constraints;
    constraints.parse(argv[1]);
    MatrixXd A = constraints.getA();
    VectorXd b = constraints.getb();
    BoundaryFinder<DualNumber> finder(4, tol, rng, A, b);
    std::function<VectorXDual(const Ref<const VectorXDual>&)> func = cleavageFunc(m);
    std::function<VectorXDual(const Ref<const VectorXDual>&, std::mt19937&)> mutate = mutate_by_delta;
    finder.run(
        func, mutate, params, max_step_iter, max_pull_iter, simplify, verbose,
        sqp_max_iter, sqp_tol, sqp_verbose, ss.str()
    );
    params = finder.getParams();

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
