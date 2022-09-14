/**
 * Compute cleavage statistics against perfect- and single-mismatch sequences
 * for the line-graph Cas9 model with edge labels determined by the free
 * energy differences obtained by Zhang et al. (2019). 
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     9/14/2022
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
 * Return the complementarity pattern between the given RNA and DNA sequences.
 * The RNA sequence is assumed to be given from 5' to 3'; the DNA sequence is 
 * assumed to be given from 3' to 5'.
 *
 * The complementarity pattern is returned as a 0/1-vector with int entries. 
 */
VectorXi complementarity(const std::string rseq, const std::string dseq)
{
    // Check that the two sequences have the same length
    if (rseq.size() != dseq.size())
        throw std::invalid_argument("Input RNA and DNA sequences have unequal lengths");

    int length = rseq.size(); 
    VectorXi pattern(length);
    for (int i = 0; i < length; ++i)
    {
        pattern(i) = (
            (rseq[i] == 'A' && dseq[i] == 'T') ||
            (rseq[i] == 'C' && dseq[i] == 'G') ||
            (rseq[i] == 'G' && dseq[i] == 'C') ||
            (rseq[i] == 'U' && dseq[i] == 'A')
        );
    }

    return pattern; 
}

/**
 * Compute the desired cleavage metric (activity, cleavage rate, live unbinding
 * rate, or dead unbinding rate) on each of the given set of sequences with 
 * respect to the line-graph Cas9 model, with the given set of forward/reverse
 * rates. 
 */
Matrix<PreciseType, Dynamic, 1> computeCleavageStats(const Ref<const Matrix<PreciseType, Dynamic, Dynamic> >& forward_rates, 
                                                     const Ref<const Matrix<PreciseType, Dynamic, Dynamic> >& reverse_rates,
                                                     const Ref<const Matrix<PreciseType, Dynamic, 2> >& terminal_rates,
                                                     const std::string metric)
{
    // Check that the dimensions of the three rate matrices are consistent 
    if (forward_rates.rows() != reverse_rates.rows())
        throw std::invalid_argument("Forward/reverse rate matrices have inconsistent dimensions");
    else if (forward_rates.cols() != length || reverse_rates.cols() != length)
        throw std::invalid_argument("Forward/reverse rate matrices have inconsistent dimensions");
    else if (forward_rates.rows() != terminal_rates.rows())
        throw std::invalid_argument("Forward/terminal rate matrices have inconsistent dimensions");  

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
        PreciseType terminal_unbind_rate = terminal_rates(j, 0); 
        PreciseType terminal_cleave_rate = terminal_rates(j, 1); 
    
        // Compute the desired cleavage metric
        if (metric == "activity")
            data(j) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        else if (metric == "cleave")
            data(j) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        else if (metric == "dead_unbind")
            data(j) = model->getLowerExitRate(terminal_unbind_rate); 
        else if (metric == "live_unbind")
            data(j) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        else 
            throw std::invalid_argument("Invalid cleavage metric specified");
    }

    delete model;
    return data; 
}

int main(int argc, char** argv)
{
    // Parse inferred parameters from Zhang et al. (2019)
    //
    // The rows in this dataset are divided into groups, each determined by 
    // guide RNA sequence and choice of forward/cleavage/unbinding rates
    std::vector<std::string> group_indices; 
    std::unordered_map<std::string, int> perfect_indices;

    int nseqs = 0; 
    Matrix<PreciseType, Dynamic, Dynamic> forward_rates(nseqs, length);
    Matrix<PreciseType, Dynamic, Dynamic> reverse_rates(nseqs, length);
    Matrix<PreciseType, Dynamic, 2> terminal_rates(nseqs, 2); 
    MatrixXi seqs(nseqs, length); 
    std::ifstream infile("data/Zhang2019-sampled-seqs-edge-labels.tsv");
    std::string line, token;
    int n = 0;        // Index of line being parsed in the following loop
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
        Matrix<PreciseType, 2, 1> curr_terminal;  
        while (std::getline(ss, token, '\t'))
        {
            // Parse the sequence ID on the current line when encountered 
            if (i == 0)
            {
                std::stringstream ss2; 
                std::string subtoken, rseq, dseq, id1, id2, group_id;
                ss2 << token; 
                std::getline(ss2, subtoken, ':'); 
                rseq = subtoken; 
                std::getline(ss2, subtoken, ':');
                dseq = subtoken;
                std::getline(ss2, subtoken, ':');
                id2 = subtoken;

                // Determine the group that the current line belongs to 
                group_id = rseq + ':' + id2;
                group_indices.push_back(group_id); 
                if (perfect_indices.find(group_id) == perfect_indices.end())
                    perfect_indices[group_id] = -1; 

                // Determine the complementarity pattern between the RNA and DNA
                curr_seq = complementarity(rseq, dseq); 
                if (curr_seq.sum() == length)   // Is the current sequence the perfect-match sequence? 
                    perfect_indices[group_id] = nseqs;
            }
            // Subsequently parse each forward/reverse rate
            else if (i >= 1 && i <= length)
            {
                curr_forward(i - 1) = static_cast<PreciseType>(std::stod(token)); 
            }
            else if (i >= length + 1 && i <= 2 * length) 
            {
                curr_reverse(i - length - 1) = static_cast<PreciseType>(std::stod(token)); 
            }
            else
            {
                curr_terminal(i - 2 * length - 1) = static_cast<PreciseType>(std::stod(token)); 
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
        terminal_rates.conservativeResize(nseqs, 2); 
        terminal_rates.row(nseqs - 1) = curr_terminal;  
        n++;
    }
    
    Matrix<PreciseType, Dynamic, 1> fit_probs = computeCleavageStats(
        forward_rates, reverse_rates, terminal_rates, "activity"
    );
    Matrix<PreciseType, Dynamic, 1> fit_dead_unbind_rates = computeCleavageStats(
        forward_rates, reverse_rates, terminal_rates, "dead_unbind"
    );
    Matrix<PreciseType, Dynamic, 1> fit_cleave_rates = computeCleavageStats(
        forward_rates, reverse_rates, terminal_rates, "cleave"
    );
    Matrix<PreciseType, Dynamic, 1> fit_live_unbind_rates = computeCleavageStats(
        forward_rates, reverse_rates, terminal_rates, "live_unbind"
    );
    Matrix<PreciseType, Dynamic, 1> fit_specs(nseqs);
    Matrix<PreciseType, Dynamic, 1> fit_dead_dissoc(nseqs); 
    Matrix<PreciseType, Dynamic, 1> fit_rapid(nseqs); 
    Matrix<PreciseType, Dynamic, 1> fit_live_dissoc(nseqs);  
    for (int i = 0; i < nseqs; ++i)
    {
        // Determine the group that each sequence belongs to, and locate 
        // the data associated with the corresponding perfect-match sequence
        std::string group_id = group_indices[i]; 
        int perfect = perfect_indices[group_id];
        fit_specs(i) = fit_probs(perfect) / fit_probs(i);
        fit_dead_dissoc(i) = fit_dead_unbind_rates(i) / fit_dead_unbind_rates(perfect); 
        fit_rapid(i) = fit_cleave_rates(perfect) / fit_cleave_rates(i); 
        fit_live_dissoc(i) = fit_live_unbind_rates(i) / fit_live_unbind_rates(perfect); 
    }
    
    std::ofstream outfile("data/Zhang2019-sampled-seqs-metrics.tsv"); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);

    // Write a header line (note that each sequence ID has been modified)
    outfile << "seqid_modified\tactivity\tcleave_rate\tdead_unbind_rate\tlive_unbind_rate\t"
            << "spec\trapid\tdeaddissoc\tlivedissoc\n"; 

    for (int i = 0; i < nseqs; ++i)
    {
        outfile << group_indices[i] << ':'; 
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
