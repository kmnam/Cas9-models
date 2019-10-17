#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "graphs/grid.hpp"
#include "sample.hpp"

/*
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     10/16/2019
 */
std::pair<double, double> computeCleavageStats(GridGraph* model, double kdis = 1.0,
                                               double kcat = 1.0)
{
    /*
     * Compute probability of cleavage and (conditional) mean first passage
     * time to the cleaved state in the given model, with the specified
     * terminal rates of dissociation and cleavage. 
     */
    // Compute weight of spanning trees rooted at each vertex
    unsigned N = model->getN();
    MatrixX2d wt = MatrixX2d::Zero(N+1, 2);
    for (unsigned i = 0; i <= N; ++i)
    {
        wt(i,0) = model->weightTrees(0, i);
        wt(i,1) = model->weightTrees(1, i);
    }

    // Compute weights of spanning forests rooted at {(A,0), (B,N)}
    double wA0BN = model->weightEndToEndForests();

    // Compute the weights of the spanning forests rooted at (X,i) and (B,N)
    // with paths from (A,0) to (X,i)
    MatrixX2d wf = MatrixX2d::Zero(N+1, 2);
    for (unsigned i = 1; i <= N; ++i)
        wf(i,0) = model->weightDirectedForests(0, i);
    for (unsigned i = 0; i < N; ++i)
        wf(i,1) = model->weightDirectedForests(1, i);
    
    // Compute the weights of the spanning forests rooted at (A,0) and (B,N)
    // with paths from (X,i) and (B,N)
    MatrixX2d wr = MatrixX2d::Zero(N+1, 2);
    for (unsigned i = 1; i <= N; ++i)
        wr(i,0) = model->weightDirectedForests2(0, i);
    for (unsigned i = 0; i < N; ++i)
        wr(i,1) = model->weightDirectedForests2(1, i);

    // For each vertex in the original graph, compute its contribution
    // to the mean first passage time
    double denom = kdis * wt(0,0) + kcat * wt(N,1) + kdis * kcat * wA0BN;
    double prob = (kcat * wt(N,1)) / denom;
    double time = 0.0;
    for (unsigned i = 1; i <= N; ++i)
    {
        double term = (kcat * wf(i,0) + wt(i,0)) / denom;
        term *= (1.0 + (kdis * wr(i,0) / wt(N,1)));
        time += term;
    }
    for (unsigned i = 0; i < N; ++i)
    {
        double term = (kcat * wf(i,1) + wt(i,1)) / denom;
        term *= (1.0 + (kdis * wr(i,1) / wt(N,1)));
        time += term;
    }
    time += (wt(0,0) + wt(N,1) + (kdis + kcat) * wA0BN) / denom;

    return std::make_pair(prob, time);
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
    MatrixXd probs(n, length + 1);
    MatrixXd times(n, length + 1);

    // For each parameter combination ...
    for (unsigned i = 0; i < n; ++i)
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
        for (unsigned j = 0; j < length; ++j)
            model->setRungLabels(j, match_params);
        
        // Compute cleavage probability and mean first passage time 
        // to cleaved state
        std::pair<double, double> data = computeCleavageStats(model);
        probs(i,0) = data.first;
        times(i,0) = data.second;

        // Introduce distal mismatches and re-compute cleavage probabilities
        for (unsigned j = 1; j <= length; ++j)
        {
            model->setRungLabels(length - j, mismatch_params);
            data = computeCleavageStats(model);
            probs(i,j) = data.first;
            times(i,j) = data.second;
        }
    }
    
    // Write array of cleavage probabilities
    oss << argv[2] << "-cleavage.tsv";
    std::ofstream outfile(oss.str());
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile.is_open())
    {
        for (unsigned i = 0; i < probs.rows(); ++i)
        {
            for (unsigned j = 0; j < probs.cols() - 1; ++j)
            {
                outfile << probs(i,j) << "\t";
            }
            outfile << probs(i,probs.cols()-1) << std::endl;
        }
    }
    outfile.close();
    oss.clear();
    oss.str(std::string());

    // Write array of mean first passage times
    oss << argv[2] << "-times.tsv";
    std::ofstream outfile2(oss.str());
    outfile2 << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (outfile2.is_open())
    {
        for (unsigned i = 0; i < times.rows(); ++i)
        {
            for (unsigned j = 0; j < times.cols() - 1; ++j)
            {
                outfile2 << times(i,j) << "\t";
            }
            outfile2 << times(i,times.cols()-1) << std::endl;
        }
    }
    outfile2.close();
    
    delete model;
    return 0;
}
