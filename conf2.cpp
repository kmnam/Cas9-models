#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <iomanip>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "graphs/grid.hpp"
#include "sample.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     8/16/2019
 */
double computeCleavageProbability(GridGraph* model, double kdis = 1.0, double kcat = 1.0)
{
    /*
     * Compute probability of cleavage in the given model, with the 
     * specified terminal rates of dissociation and cleavage. 
     */
    // Compute weight of spanning trees rooted at (B,N)
    double weightBN = model->weightTreesBN();

    // Compute weights of spanning trees rooted at (A,0) and of spanning
    // forests rooted at {(A,0), (B,N)}
    std::pair<double, double> weights = model->weightTreesA0ForestsA0BN();
    double weightA0 = weights.first;
    double weightA0BN = weights.second;

    // Compute probability of cleavage 
    return 1.0 / (1.0 + (kdis * weightA0 + kdis * kcat * weightA0BN) / (kcat * weightBN));
}

int main(int argc, char** argv)
{
    // Instantiate a GridGraph of length 20
    unsigned length = 20;
    GridGraph* model = new GridGraph(length);

    // Instantiate random number generator 
    boost::random::mt19937 rng(1234567890);

    // Sample model parameter combinations
    unsigned n;
    sscanf(argv[3], "%u", &n);
    MatrixXd params(n, 6);
    try
    {
        params = sampleFromConvexPolytopeTriangulation(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }
    params = pow(10.0, params.array()).matrix();

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[2] << "-sample.tsv";
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

    // Maintain array of cleavage probabilities
    MatrixXd cleavageProbs(n, length + 1);

    // For each parameter combination ...
    for (unsigned i = 0; i < n; i++)
    {
        // Array of DNA/RNA match parameters
        std::array<double, 6> match_params;
        match_params[0] = params(i,0);
        match_params[1] = params(i,1);
        match_params[2] = params(i,0);
        match_params[3] = params(i,1);
        match_params[4] = params(i,4);
        match_params[5] = params(i,5);

        // Array of DNA/RNA mismatch parameters
        std::array<double, 6> mismatch_params;
        mismatch_params[0] = params(i,2);
        mismatch_params[1] = params(i,3);
        mismatch_params[2] = params(i,2);
        mismatch_params[3] = params(i,3);
        mismatch_params[4] = params(i,4);
        mismatch_params[5] = params(i,5);

        // Populate each rung with DNA/RNA match parameters
        model->setStartLabels(params(i,4), params(i,5));
        for (unsigned j = 0; j < length; j++)
            model->setRungLabels(j, match_params);
        
        // Compute cleavage probability
        cleavageProbs(i,0) = computeCleavageProbability(model);

        // Introduce distal mismatches and re-compute cleavage probabilities
        for (unsigned j = 1; j <= length; j++)
        {
            model->setRungLabels(length - j, mismatch_params);
            cleavageProbs(i,j) = computeCleavageProbability(model);
        }
    }
    
    // Write array of cleavage probabilities
    oss << argv[2] << "-cleavage.tsv";
    std::ofstream outfile(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < cleavageProbs.rows(); i++)
        {
            for (unsigned j = 0; j < cleavageProbs.cols() - 1; j++)
            {
                outfile << cleavageProbs(i,j) << "\t";
            }
            outfile << cleavageProbs(i,cleavageProbs.cols()-1) << "\n";
        }
    }
    outfile.close();
    
    delete model;
    return 0;
}
