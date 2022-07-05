/**
 * Compute cleavage statistics against perfect- and single-mismatch sequences 
 * for the line-graph Cas9 model with edge labels determined by the free 
 * energy differences obtained by Eslami-Mossallam et al. (2022).
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     7/2/2022
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <boostMultiprecisionEigen.hpp>
#include <linearConstraints.hpp>
#include <polytopes.hpp>
#include <SQP.hpp>
#include <graphs/line.hpp>

using namespace Eigen;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::log10; 
constexpr int INTERNAL_PRECISION = 100; 
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;

const unsigned length = 20;

/**
 * Compute the desired cleavage metric (activity, cleavage rate, live unbinding
 * rate, or dead unbinding rate) on each of the given set of sequences with 
 * respect to the line-graph Cas9 model, with the given set of forward/reverse
 * rates. 
 */
Matrix<PreciseType, Dynamic, 1> computeCleavageStats(const Ref<const Matrix<PreciseType, Dynamic, Dynamic> >& forward_rates, 
                                                     const Ref<const Matrix<PreciseType, Dynamic, Dynamic> >& reverse_rates,
                                                     const PreciseType unbind_rate,
                                                     const PreciseType cleave_rate, 
                                                     const std::string metric)
{
    // Check that the dimensions of the two rate matrices are the same 
    if (forward_rates.rows() != reverse_rates.rows())
        throw std::invalid_argument("Forward/reverse rate matrices have inconsistent dimensions");
    else if (forward_rates.cols() != length || reverse_rates.cols() != length)
        throw std::invalid_argument("Forward/reverse rate matrices have inconsistent dimensions");

    // Populate each rung of the line graph with each vector of forward/
    // reverse rates
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    Matrix<PreciseType, Dynamic, 1> data(forward_rates.rows()); 
    for (int j = 0; j < forward_rates.rows(); ++j)
    {
        for (int k = 0; k < length; ++k)
        {
            std::pair<PreciseType, PreciseType> labels = std::make_pair(
                forward_rates(j, k), reverse_rates(j, k)
            ); 
            model->setEdgeLabels(k, labels); 
        }
    
        // Compute the desired cleavage metric
        if (metric == "activity")
            data(j) = model->getUpperExitProb(unbind_rate, cleave_rate);
        else if (metric == "cleave")
            data(j) = model->getUpperExitRate(unbind_rate, cleave_rate); 
        else if (metric == "deadUnbind")
            data(j) = model->getLowerExitRate(unbind_rate); 
        else if (metric == "liveUnbind")
            data(j) = model->getLowerExitRate(unbind_rate, cleave_rate);
        else 
            throw std::invalid_argument("Invalid cleavage metric specified");
    }

    delete model;
    return data; 
}

int main(int argc, char** argv)
{
    // Parse inferred parameters from Eslami-Mossallam et al. (2022)
    int nseqs = 0;
    Matrix<PreciseType, Dynamic, Dynamic> forward_rates(nseqs, length);
    Matrix<PreciseType, Dynamic, Dynamic> reverse_rates(nseqs, length);
    MatrixXi seqs(nseqs, length); 
    PreciseType unbind_rate = 0.0132723;
    PreciseType cleave_rate = 2.39286;
    std::ifstream infile("data/EslamiMossallam2022-seqs-edge-labels.tsv"); 
    std::string line, token;
    int n = 0;        // Index of line being parsed in the following loop
    int perfect = 0;  // Index of perfect-match sequence in the dataset 
    while (std::getline(infile, line))
    {
        // Ignore the first line 
        if (n == 0)
        {
            n++;
            continue;
        } 
        std::stringstream ss;
        ss << line;

        int i = 0;    // Index of entry being parsed in the following loop
        VectorXi curr_seq(length); 
        Matrix<PreciseType, Dynamic, 1> curr_forward(length);
        Matrix<PreciseType, Dynamic, 1> curr_reverse(length);  
        while (std::getline(ss, token, '\t'))
        {
            // Parse the sequence on the current line when encountered 
            if (i == 0)
            {
                int j = 0; 
                for (const char c : token)
                {
                    curr_seq(j) = (c == '1'); 
                    j++; 
                }
                if (curr_seq.sum() == length)   // Is the current sequence the perfect-match sequence? 
                    perfect = nseqs;
            }
            // Subsequently parse each forward/reverse rate
            else if (i >= 1 && i <= length)
            {
                curr_forward(i - 1) = static_cast<PreciseType>(std::stod(token)); 
            }
            else 
            {
                curr_reverse(i - length - 1) = static_cast<PreciseType>(std::stod(token)); 
            }
            i++; 
        }

        // Add the data from the current line to the larger arrays 
        nseqs++; 
        seqs.conservativeResize(nseqs, length); 
        seqs.row(nseqs - 1) = curr_seq;
        forward_rates.conservativeResize(nseqs, length); 
        forward_rates.row(nseqs - 1) = curr_forward; 
        reverse_rates.conservativeResize(nseqs, length); 
        reverse_rates.row(nseqs - 1) = curr_reverse; 
        
        n++;
    }

    Matrix<PreciseType, Dynamic, 1> fit_probs = computeCleavageStats(
        forward_rates, reverse_rates, unbind_rate, cleave_rate, "activity"
    );
    Matrix<PreciseType, Dynamic, 1> fit_dead_unbind_rates = computeCleavageStats(
        forward_rates, reverse_rates, unbind_rate, cleave_rate, "deadUnbind"
    );
    Matrix<PreciseType, Dynamic, 1> fit_cleave_rates = computeCleavageStats(
        forward_rates, reverse_rates, unbind_rate, cleave_rate, "cleave"
    );
    Matrix<PreciseType, Dynamic, 1> fit_live_unbind_rates = computeCleavageStats(
        forward_rates, reverse_rates, unbind_rate, cleave_rate, "liveUnbind"
    );
    Matrix<PreciseType, Dynamic, 1> fit_specs(nseqs);
    Matrix<PreciseType, Dynamic, 1> fit_dead_dissoc(nseqs); 
    Matrix<PreciseType, Dynamic, 1> fit_rapid(nseqs); 
    Matrix<PreciseType, Dynamic, 1> fit_live_dissoc(nseqs);  
    for (int i = 0; i < nseqs; ++i)
    {
        fit_specs(i) = fit_probs(perfect) / fit_probs(i);
        fit_dead_dissoc(i) = fit_dead_unbind_rates(perfect) / fit_dead_unbind_rates(i); 
        fit_rapid(i) = fit_cleave_rates(perfect) / fit_cleave_rates(i); 
        fit_live_dissoc(i) = fit_live_unbind_rates(perfect) / fit_live_unbind_rates(i);  
    }
    
    std::ofstream outfile("data/EslamiMossallam2022-seqs-metrics.tsv");
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    
    // Write a header line
    outfile << "seqid\tactivity\tcleave_rate\tdead_unbind_rate\tlive_unbind_rate\t"
            << "spec\trapid\tdead_dissoc\tlive_dissoc\n"; 

    for (int i = 0; i < nseqs; ++i)
    {
        for (int j = 0; j < length; ++j)
            outfile << (seqs(i, j) ? '1' : '0');
        outfile << '\t'; 

        // ... along with their associated cleavage activities ... 
        outfile << fit_probs(i) << '\t'; 

        // ... and their associated cleavage rates ... 
        outfile << fit_cleave_rates(i) << '\t';

        // ... and their associated dead and live unbinding rates ... 
        outfile << fit_dead_unbind_rates(i) << '\t'
                << fit_live_unbind_rates(i) << '\t';

        // ... and their associated specificities ... 
        outfile << log10(fit_specs(i)) << '\t'; 

        // ... and their associated specific rapidities ... 
        outfile << log10(fit_rapid(i)) << '\t';

        // ... and their associated dead and live specific dissociativities ... 
        outfile << log10(fit_dead_dissoc(i)) << '\t'
                << log10(fit_live_dissoc(i)) << std::endl; 
    }
    outfile.close();
    
    return 0; 
}
