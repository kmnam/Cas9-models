#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include <autodiff/reverse.hpp>
#include <autodiff/reverse/eigen.hpp>
#include <boundaryFinder.hpp>
#include "../include/graphs/grid.hpp"
#include "../include/sample.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     11/13/2019
 */
using namespace Eigen;
using namespace autodiff;

const unsigned length = 5;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

boost::random::uniform_int_distribution<> fair_bernoulli_dist(0, 1);

int coin_toss(boost::random::mt19937& rng)
{
    return fair_bernoulli_dist(rng);
}

template <unsigned n_mismatches = 1>
VectorXvar computeCleavageStats(const Ref<const VectorXvar>& params)
{
    /*
     *
     */
    // Array of DNA/RNA match parameters
    std::array<var, 6> match_params;
    match_params[0] = pow(10.0, params(0));
    match_params[1] = pow(10.0, params(1));
    match_params[2] = pow(10.0, params(0));
    match_params[3] = pow(10.0, params(1));
    match_params[4] = pow(10.0, params(4));
    match_params[5] = pow(10.0, params(5));

    // Array of DNA/RNA mismatch parameters
    std::array<var, 6> mismatch_params;
    mismatch_params[0] = pow(10.0, params(2));
    mismatch_params[1] = pow(10.0, params(3));
    mismatch_params[2] = pow(10.0, params(2));
    mismatch_params[3] = pow(10.0, params(3));
    mismatch_params[4] = pow(10.0, params(4));
    mismatch_params[5] = pow(10.0, params(5));

    // Populate each rung with DNA/RNA match parameters
    GridGraph<var>* model = new GridGraph<var>(length);
    model->setStartLabels(match_params[4], match_params[5]);
    for (unsigned j = 0; j < length; ++j)
        model->setRungLabels(j, match_params);
    
    // Compute cleavage probability and mean first passage time 
    // to cleaved state
    Matrix<var, 2, 1> match_data = model->computeCleavageStatsForests(1, 1).array().log10().matrix();

    // Introduce distal mismatches and re-compute cleavage probability
    // and mean first passage time
    for (unsigned j = 1; j <= n_mismatches; ++j)
        model->setRungLabels(length - j, mismatch_params);
    Matrix<var, 2, 1> mismatch_data = model->computeCleavageStatsForests(1, 1).array().log10().matrix();

    // Compute the specificity and speed ratio
    var specificity = match_data(0) - mismatch_data(0);
    var speed_ratio = mismatch_data(1) - match_data(1);
    VectorXvar stats(2);
    stats << specificity, speed_ratio;

    delete model;
    return stats;
}

VectorXvar mutate_by_delta(const Ref<const VectorXvar>& params, boost::random::mt19937& rng, LinearConstraints* constraints)
{
    /*
     *
     */
    VectorXvar mutated(params);
    const var delta = 0.1;
    for (unsigned i = 0; i < mutated.size(); ++i)
    {
        int toss = coin_toss(rng);
        if (!toss) mutated(i) += delta;
        else       mutated(i) -= delta;
    }
    return constraints->nearestL2(mutated.cast<double>()).cast<var>();
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[4], "%u", &n);
    MatrixXd params(n, 6);
    try
    {
        params = sampleFromConvexPolytopeTriangulation(argv[2], n, rng);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    // Run the boundary-finding algorithm
    LinearConstraints constraints;
    constraints.parse(argv[1]);
    MatrixXd A = constraints.getA();
    VectorXd b = constraints.getb();

    BoundaryFinder finder(6, 1e-5, 20, rng, A, b);
    std::function<VectorXvar(const Ref<const VectorXvar>&)> func = computeCleavageStats<1>;
    std::function<VectorXvar(const Ref<const VectorXvar>&, boost::random::mt19937&, LinearConstraints*)> mutate = mutate_by_delta;
    finder.run(func, mutate, params, true, true, "test");

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[3] << "-sample.tsv";
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
