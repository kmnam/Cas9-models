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
#include <polytopes.hpp>
#include <graphs/line.hpp>
#include <boostMultiprecisionEigen.hpp>

/*
 * Computes cleavage statistics for the line-graph Cas9 model for parameter 
 * combinations taken from the boundary of the given convex polytope.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * 
 * **Last updated:**
 *     2/7/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10;
const unsigned MODEL_LENGTH = 20; 
const unsigned PRECISION = 1000; 
typedef number<mpfr_float_backend<PRECISION> > PreciseType; 

/**
 * Return the cleavage probability, cleavage specificity, unbinding rate, 
 * normalized unbinding rate, cleavage rate, and normalized cleavage rate 
 * for the line-graph Cas9 model with the given set of parameter values
 * on the perfect-match and all single-mismatch substrates.
 *
 * This function returns two matrices:
 * - the first of size 3 x 21, where each column contains the cleavage
 *   probability, unbinding rate, and cleavage rate with respect to each
 *   substrate (perfect-match substrate, substrate with mismatch at position
 *   0, etc.); and
 * - the second of size 3 x 20, where each column contains the cleavage
 *   specificity, normalized unbinding rate, and cleavage rate with respect
 *   to each mismatched substrate.  
 */
std::pair<MatrixXd, MatrixXd> computeCleavageStats(const Ref<const VectorXd>& params) 
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_params; 
    match_params.first = static_cast<PreciseType>(std::pow(10.0, params(0)));
    match_params.second = static_cast<PreciseType>(std::pow(10.0, params(1)));

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_params;
    mismatch_params.first = static_cast<PreciseType>(std::pow(10.0, params(2)));
    mismatch_params.second = static_cast<PreciseType>(std::pow(10.0, params(3)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(MODEL_LENGTH);
    for (unsigned i = 0; i < MODEL_LENGTH; ++i)
        model->setEdgeLabels(i, match_params);
    
    // Compute cleavage probability, unbinding rate, and cleavage rate on the
    // perfect-match substrate
    PreciseType unbind_rate = 1;
    PreciseType cleave_rate = 1;
    Matrix<PreciseType, Dynamic, Dynamic> unnorm_stats(3, MODEL_LENGTH + 1); 
    Matrix<PreciseType, Dynamic, Dynamic> norm_stats(3, MODEL_LENGTH);  
    PreciseType prob_perfect = model->getUpperExitProb(unbind_rate, cleave_rate);
    PreciseType unbind_perfect = model->getLowerExitRate(unbind_rate); 
    PreciseType cleave_perfect = model->getUpperExitRate(unbind_rate, cleave_rate); 
    unnorm_stats(0, 0) = prob_perfect;
    unnorm_stats(1, 0) = unbind_perfect; 
    unnorm_stats(2, 0) = cleave_perfect;  

    // Introduce one mismatch at the specified position and re-compute
    // cleavage probability, unbinding rate, and cleavage rate 
    for (unsigned i = 0; i < MODEL_LENGTH; ++i)
    {
        for (unsigned j = 0; j < MODEL_LENGTH; ++j)
            model->setEdgeLabels(j, match_params);  
        model->setEdgeLabels(i, mismatch_params);
        PreciseType prob_mismatched = model->getUpperExitProb(unbind_rate, cleave_rate);
        PreciseType unbind_mismatched = model->getLowerExitRate(unbind_rate); 
        PreciseType cleave_mismatched = model->getUpperExitRate(unbind_rate, cleave_rate); 
        unnorm_stats(0, i+1) = prob_mismatched;
        unnorm_stats(1, i+1) = unbind_mismatched; 
        unnorm_stats(2, i+1) = cleave_mismatched; 
        norm_stats(0, i) = log10(prob_perfect) - log10(prob_mismatched);
        norm_stats(1, i) = log10(unbind_perfect) - log10(unbind_mismatched); 
        norm_stats(2, i) = log10(cleave_perfect) - log10(cleave_mismatched);  
    } 

    // Compile results and return 
    delete model;
    return std::make_pair(unnorm_stats.cast<double>(), norm_stats.cast<double>()); 
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    int npoints = std::stoi(argv[3]);

    // Parse the convex polytope of valid parameter values and sample the 
    // same number of points from its codim-1, codim-2, and codim-3 boundary
    // faces (note that the polytope is 4-dimensional) 
    for (int codim = 1; codim <= 3; ++codim)
    {
        MatrixXd sample_codim = sampleFromConvexPolytope<PRECISION>(argv[1], npoints, codim, rng);
        MatrixXd probs(npoints, MODEL_LENGTH + 1); 
        MatrixXd specs(npoints, MODEL_LENGTH); 
        MatrixXd unbind(npoints, MODEL_LENGTH + 1); 
        MatrixXd norm_unbind(npoints, MODEL_LENGTH); 
        MatrixXd cleave(npoints, MODEL_LENGTH + 1); 
        MatrixXd norm_cleave(npoints, MODEL_LENGTH);  

        // Obtain cleavage statistics for each sampled parameter combination 
        for (unsigned i = 0; i < npoints; ++i)
        {
            MatrixXd unnorm_stats, norm_stats; 
            std::tie(unnorm_stats, norm_stats) = computeCleavageStats(sample_codim.row(i));
            probs.row(i) = unnorm_stats.row(0); 
            specs.row(i) = norm_stats.row(0); 
            unbind.row(i) = unnorm_stats.row(1); 
            norm_unbind.row(i) = norm_stats.row(1); 
            cleave.row(i) = unnorm_stats.row(2); 
            norm_cleave.row(i) = norm_stats.row(2); 
        }

        // Write sampled parameter combinations to file
        std::ostringstream oss;
        oss << argv[2] << "-boundary-codim" << codim << "-params.tsv";
        std::ofstream samplefile(oss.str());
        samplefile << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (samplefile.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i) 
            {
                for (unsigned j = 0; j < 3; ++j)    // Each combination consists of 4 values
                {
                    samplefile << sample_codim(i, j) << '\t';
                }
                samplefile << sample_codim(i, 3) << std::endl; 
            }
        }
        samplefile.close();
        oss.clear();
        oss.str(std::string());

        // Write matrix of cleavage probabilities
        oss << argv[2] << "-boundary-codim" << codim << "-probs.tsv";
        std::ofstream probsfile(oss.str());
        probsfile << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (probsfile.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i)
            {
                for (unsigned j = 0; j < MODEL_LENGTH; ++j) 
                {
                    probsfile << probs(i, j) << '\t';
                }
                probsfile << probs(i, MODEL_LENGTH) << std::endl; 
            }
        }
        probsfile.close();
        oss.clear();
        oss.str(std::string());

        // Write matrix of cleavage specificities 
        oss << argv[2] << "-boundary-codim" << codim << "-specs.tsv";
        std::ofstream specsfile(oss.str());
        specsfile << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (specsfile.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i)
            {
                for (unsigned j = 0; j < MODEL_LENGTH - 1; ++j)
                {
                    specsfile << specs(i, j) << '\t';
                }
                specsfile << specs(i, MODEL_LENGTH - 1) << std::endl; 
            }
        }
        specsfile.close();
        oss.clear();
        oss.str(std::string());

        // Write matrix of (unnormalized) unconditional unbinding rates
        oss << argv[2] << "-boundary-codim" << codim << "-unbind-rates.tsv";
        std::ofstream unbindfile(oss.str());
        unbindfile << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (unbindfile.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i)
            {
                for (unsigned j = 0; j < MODEL_LENGTH; ++j)
                {
                    unbindfile << unbind(i, j) << "\t";
                }
                unbindfile << unbind(i, MODEL_LENGTH) << std::endl; 
            }
        }
        unbindfile.close();
        oss.clear();
        oss.str(std::string());

        // Write matrix of (unnormalized) conditional cleavage rates 
        oss << argv[2] << "-boundary-codim" << codim << "-cleave-rates.tsv";
        std::ofstream cleavefile(oss.str());
        cleavefile << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (cleavefile.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i)
            {
                for (unsigned j = 0; j < MODEL_LENGTH; ++j) 
                {
                    cleavefile << cleave(i, j) << "\t";
                }
                cleavefile << cleave(i, MODEL_LENGTH) << std::endl; 
            }
        }
        cleavefile.close();
        oss.clear();
        oss.str(std::string());
      
        // Write matrix of normalized unconditional unbinding rates
        oss << argv[2] << "-boundary-codim" << codim << "-norm-unbind.tsv";
        std::ofstream unbindfile2(oss.str());
        unbindfile2 << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (unbindfile2.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i)
            {
                for (unsigned j = 0; j < MODEL_LENGTH - 1; ++j) 
                {
                    unbindfile2 << norm_unbind(i, j) << "\t";  
                }
                unbindfile2 << norm_unbind(i, MODEL_LENGTH - 1) << std::endl; 
            }
        }
        unbindfile2.close();
        oss.clear();
        oss.str(std::string());

        // Write matrix of normalized conditional cleavage rates 
        oss << argv[2] << "-boundary-codim" << codim << "-norm-cleave.tsv";
        std::ofstream cleavefile2(oss.str());
        cleavefile2 << std::setprecision(std::numeric_limits<double>::max_digits10);
        if (cleavefile2.is_open())
        {
            for (unsigned i = 0; i < npoints; ++i)
            {
                for (unsigned j = 0; j < MODEL_LENGTH - 1; ++j)
                {
                    cleavefile2 << norm_cleave(i, j) << "\t"; 
                }
                cleavefile2 << norm_cleave(i, MODEL_LENGTH - 1) << std::endl; 
            }
        }
        cleavefile2.close();
    }
    
    return 0;
}
