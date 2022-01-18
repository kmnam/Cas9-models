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
#include <graphs/grid.hpp>
#include <boostMultiprecisionEigen.hpp>

/*
 * Estimates the boundary of the cleavage activity vs. cleavage specificity
 * region in the grid-graph Cas9 model. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     1/18/2022
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

/**
 * Compute:
 * - cleavage activity on the perfect-match substrate and
 * - cleavage specificity with respect to the single-distal-mismatch substrate
 * for the line graph with the given set of parameter values. 
 */
template <typename T>
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

    // Array of conformational change parameters
    std::pair<T, T> switch_params = std::make_pair(
        static_cast<T>(std::pow(10.0, params(4))), 
        static_cast<T>(std::pow(10.0, params(5)))
    ); 

    // Populate each rung with DNA/RNA match parameters
    GridGraph<T, T>* model = new GridGraph<T, T>(length);
    std::array<T, 6> labels; 
    model->setZerothLabels(switch_params.first, switch_params.second);
    for (unsigned j = 0; j < length; ++j)
    {
        labels[0] = match_params.first; 
        labels[1] = match_params.second; 
        labels[2] = match_params.first; 
        labels[3] = match_params.second; 
        labels[4] = switch_params.first; 
        labels[5] = switch_params.second; 
        model->setRungLabels(j, labels); 
    } 
    
    // Compute cleavage probability on the perfect-match substrate
    T unbind_rate = 1;
    T cleave_rate = 1; 
    T prob_perfect = std::get<0>(model->getExitStats(unbind_rate, cleave_rate));

    // Introduce one distal mismatch and re-compute cleavage probability
    labels[0] = mismatch_params.first; 
    labels[1] = mismatch_params.second; 
    labels[2] = mismatch_params.first; 
    labels[3] = mismatch_params.second; 
    model->setRungLabels(length - 1, labels); 
    T prob_mismatched = std::get<0>(model->getExitStats(unbind_rate, cleave_rate)); 

    // Compile results and return 
    VectorXd output(2);
    output << static_cast<double>(prob_perfect),
              std::log10(static_cast<double>(prob_perfect)) - std::log10(static_cast<double>(prob_mismatched)); 

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
    unsigned n_within = 10000;
    unsigned n_bound = 0;
    unsigned min_step_iter = 100;
    unsigned max_step_iter = 500;
    unsigned min_pull_iter = 10;
    unsigned max_pull_iter = 100;
    unsigned max_edges = 500;
    bool verbose = true;
    unsigned sqp_max_iter = 100;
    double sqp_tol = 1e-3;
    bool sqp_verbose = false;
    std::stringstream ss;
    ss << argv[3] << "-boundary";

    // Run the boundary-finding algorithm
    BoundaryFinder finder(tol, rng, argv[1], argv[2]);
    std::function<VectorXd(const Ref<const VectorXd>&, boost::random::mt19937&)> mutate = mutateByDelta<double>;
    finder.run(
        computeCleavageStats<number<mpfr_float_backend<100> > >,
        mutate, filter, n_within, n_bound, min_step_iter, max_step_iter,
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
