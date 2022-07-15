/**
 * Using line-search SQP, identify the set of line-graph Cas9 model parameter 
 * vectors that yields each given set of unbinding and cleavage rates.  
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     7/13/2022
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
using boost::multiprecision::pow;
using boost::multiprecision::sqrt;
constexpr int INTERNAL_PRECISION = 100; 
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;

const unsigned length = 20;
const PreciseType ten("10");

/**
 * Compute cleavage statistics on the perfect-match sequence, as well as all
 * mismatched sequences specified in the given matrix of complementarity
 * patterns, for the given line-graph Cas9 model. 
 */
Matrix<PreciseType, Dynamic, 5> computeCleavageStats(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                                                     const Ref<const MatrixXi>& seqs, 
                                                     const PreciseType bind_conc,
                                                     const bool normalize = true)
{
    // Array of DNA/RNA match parameters
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, logrates(0)), pow(ten, logrates(1))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, logrates(2)), pow(ten, logrates(3))
    );

    // Exit rates from terminal nodes 
    PreciseType terminal_unbind_rate = pow(ten, logrates(4));
    PreciseType terminal_cleave_rate = pow(ten, logrates(5)); 

    // Binding rate entering state 0
    PreciseType bind_rate = pow(ten, logrates(6));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and 
    // live unbinding rate against perfect-match substrate
    Matrix<PreciseType, 5, 1> stats_perfect;
    stats_perfect(0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats_perfect(2) = model->getLowerExitRate(terminal_unbind_rate); 
    stats_perfect(3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);

    // Compute the composite cleavage time 
    stats_perfect(4) = (
        1 / (bind_conc * bind_rate)
        + (1 / stats_perfect(3) + 1 / (bind_conc * bind_rate)) * (1 - stats_perfect(0)) / stats_perfect(0)
        + 1 / stats_perfect(1)
    );

    // Re-compute cleavage probability, cleavage rate, dead unbinding rate, 
    // and live unbinding rate against each given substrate
    Matrix<PreciseType, Dynamic, 5> stats(seqs.rows(), 5);  
    for (int j = 0; j < seqs.rows(); ++j)
    {
        for (int k = 0; k < length; ++k)
        {
            if (seqs(j, k))
                model->setEdgeLabels(k, match_rates);
            else
                model->setEdgeLabels(k, mismatch_rates);
        }
        stats(j, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(j, 2) = model->getLowerExitRate(terminal_unbind_rate); 
        stats(j, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 4) = (
            1 / (bind_conc * bind_rate)
            + (1 / stats(j, 3) + 1 / (bind_conc * bind_rate)) * (1 - stats(j, 0)) / stats(j, 0)
            + 1 / stats(j, 1)
        ); 

        // Normalize the metrics if desired 
        if (normalize)
        {
            stats(j, 0) = pow(ten, log10(stats_perfect(0)) - log10(stats(j, 0))); 
            stats(j, 1) = pow(ten, log10(stats_perfect(1)) - log10(stats(j, 1))); 
            stats(j, 2) = pow(ten, log10(stats(j, 2)) - log10(stats_perfect(2))); 
            stats(j, 3) = pow(ten, log10(stats(j, 3)) - log10(stats_perfect(3)));
            // Note that computeCleavageStats() returns composite cleavage *times*, not rates  
            stats(j, 4) = pow(ten, log10(stats(j, 4)) - log10(stats_perfect(4))); 
        }
    }

    delete model;
    return stats;
}

/**
 * Compute the error (L2 distance) between a set of cleavage rates and dead
 * unbinding rates inferred from the line-graph Cas9 model and a set of
 * experimentally determined cleavage rates and dead unbinding rates. 
 */
PreciseType errorAgainstData(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                             const Ref<const MatrixXi>& cleave_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
                             const Ref<const MatrixXi>& unbind_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
                             const PreciseType bind_conc,
                             const bool normalize = true)
{
    Matrix<PreciseType, Dynamic, 5> stats1, stats2; 

    if (normalize)
    {
        // If normalization is desired, then assume that there exist cleavage 
        // and dead unbinding rates for the perfect-match substrate
        int unbind_perfect_index = -1;
        for (int i = 0; i < unbind_seqs.rows(); ++i)
        {
            if (unbind_seqs.row(i).sum() == length)
            {
                unbind_perfect_index = i;
                break; 
            }
        } 
        if (unbind_perfect_index == -1)
        {
            throw std::runtime_error(
                "Cannot normalize given data without value corresponding to perfect-match substrate"
            );
        }
        int cleave_perfect_index = -1; 
        for (int i = 0; i < cleave_seqs.rows(); ++i)
        {
            if (cleave_seqs.row(i).sum() == length)
            {
                cleave_perfect_index = i; 
                break;
            }
        }
        if (cleave_perfect_index == -1)
        {
            throw std::runtime_error(
                "Cannot normalize given data without value corresponding to perfect-match substrate"
            );
        }

        // Normalize all given composite cleavage times and dead unbinding rates
        // (with copies of the data matrices that were passed into this function)
        Matrix<PreciseType, Dynamic, 1> unbind_data2(unbind_data.size());
        Matrix<PreciseType, Dynamic, 1> cleave_data2(cleave_data.size());  
        for (int i = 0; i < unbind_data2.size(); ++i)
            unbind_data2(i) = unbind_data(i) / unbind_data(unbind_perfect_index); 
        for (int i = 0; i < cleave_data2.size(); ++i)
            cleave_data2(i) = cleave_data(i) / cleave_data(cleave_perfect_index); 

        // Compute *normalized* cleavage metrics 
        stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc, true);
        stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc, true);

        // Note that computeCleavageStats() returns composite cleavage *times* and 
        // dead unbinding *rates*, whereas the data includes only *times* and no *rates*
        return sqrt(
            (stats1.col(4) - cleave_data2).squaredNorm() +
            (stats2.col(2).array().pow(-1).matrix() - unbind_data2).squaredNorm()
        ); 
    }
    else 
    {
        // Otherwise, compute *unnormalized* cleavage metrics 
        stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc, false);
        stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc, false);

        // Note that computeCleavageStats() returns composite cleavage *times* and 
        // dead unbinding *rates*, whereas the data includes only *times* and no *rates*
        return sqrt(
            (stats1.col(4) - cleave_data).squaredNorm() +
            (stats2.col(2).array().pow(-1).matrix() - unbind_data).squaredNorm()
        ); 
    }
}

void fitCleavageStats(const std::string outfilename, const PreciseType bind_conc,
                      const int n_init, boost::random::mt19937& rng)
{
    // Set up an SQPOptimizer instance that constrains all parameters to lie 
    // between 1e-10 and 1e+10 
    Polytopes::LinearConstraints<mpq_rational>* constraints_opt = new Polytopes::LinearConstraints<mpq_rational>(
        Polytopes::InequalityType::GreaterThanOrEqualTo 
    );
    constraints_opt->parse("polytopes/line-10-terminal-plusbind.poly");
    const int D = constraints_opt->getD(); 
    const int N = constraints_opt->getN(); 
    SQPOptimizer<PreciseType>* opt = new SQPOptimizer<PreciseType>(constraints_opt);

    // Sample a set of points from an initial polytope in parameter space, 
    // which constrains all parameters to lie between 1e-5 and 1e+5
    Delaunay_triangulation* tri = new Delaunay_triangulation(D);
    Polytopes::parseVerticesFile("polytopes/line-5-terminal-plusbind.vert", tri); 
    Matrix<PreciseType, Dynamic, Dynamic> init_points
        = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, n_init, 0, rng);

    // Parse cleavage rates and dead unbinding rates with respect to given 
    // mismatched sequences tested by Singh et al. (2018)
    std::vector<int> n_cleave_data, n_unbind_data; 
    std::vector<MatrixXi> cleave_seqs, unbind_seqs;
    std::vector<Matrix<PreciseType, Dynamic, 1> > cleave_data, unbind_data; 
    for (int i = 0; i < 3; ++i)
    {
        n_cleave_data.push_back(0);
        n_unbind_data.push_back(0);  
        cleave_seqs.emplace_back(MatrixXi::Zero(0, length));
        unbind_seqs.emplace_back(MatrixXi::Zero(0, length));
        cleave_data.emplace_back(Matrix<PreciseType, Dynamic, 1>::Zero(0)); 
        unbind_data.emplace_back(Matrix<PreciseType, Dynamic, 1>::Zero(0)); 
    }
    std::vector<std::string> cleave_infilenames, unbind_infilenames;  
    cleave_infilenames.push_back("data/Singh-2018-NatStructMolBiol-SI2-Fig4c-TimeCleavage-SpCas9.txt");
    cleave_infilenames.push_back("data/Singh-2018-NatStructMolBiol-SI2-Fig4c-TimeCleavage-SpCas9HF1.txt");
    cleave_infilenames.push_back("data/Singh-2018-NatStructMolBiol-SI2-Fig4c-TimeCleavage-eSpCas9.txt");
    unbind_infilenames.push_back("data/Singh-2018-NatStructMolBiol-SI2-Fig1d-TimeBoundDead-SpCas9.txt"); 
    unbind_infilenames.push_back("data/Singh-2018-NatStructMolBiol-SI2-Fig1d-TimeBoundDead-SpCas9HF1.txt"); 
    unbind_infilenames.push_back("data/Singh-2018-NatStructMolBiol-SI2-Fig1d-TimeBoundDead-eSpCas9.txt");
    int i = 0;  
    std::string line;
    for (const std::string& infilename : cleave_infilenames)
    {
        std::ifstream infile(infilename); 
        while (std::getline(infile, line))
        {
            std::stringstream ss;
            ss << line;

            // The first entry is the binary sequence
            std::string seq;
            std::getline(ss, seq, '\t');
            n_cleave_data[i] = n_cleave_data[i] + 1;
            cleave_seqs[i].conservativeResize(n_cleave_data[i], length); 
            cleave_data[i].conservativeResize(n_cleave_data[i]);

            // Parse the binary sequence, character by character
            if (seq.size() != length)
                throw std::runtime_error("Parsed binary sequence of invalid length (!= 20)"); 
            for (int j = 0; j < length; ++j)
            {
                if (seq[j] == '0')
                    cleave_seqs[i](n_cleave_data[i] - 1, j) = 0; 
                else
                    cleave_seqs[i](n_cleave_data[i] - 1, j) = 1;
            }

            // The second entry is the cleavage metric being parsed 
            std::string token; 
            std::getline(ss, token, '\t'); 
            cleave_data[i](n_cleave_data[i] - 1) = std::stod(token);
        }
        infile.close();
        i++;
    }
    i = 0; 
    for (const std::string& infilename : unbind_infilenames)
    {
        std::ifstream infile(infilename); 
        while (std::getline(infile, line))
        {
            std::stringstream ss;
            ss << line;

            // The first entry is the binary sequence
            std::string seq;
            std::getline(ss, seq, '\t');
            n_unbind_data[i]++;
            unbind_seqs[i].conservativeResize(n_unbind_data[i], length); 
            unbind_data[i].conservativeResize(n_unbind_data[i]);

            // Parse the binary sequence, character by character
            if (seq.size() != length)
                throw std::runtime_error("Parsed binary sequence of invalid length (!= 20)"); 
            for (int j = 0; j < length; ++j)
            {
                if (seq[j] == '0')
                    unbind_seqs[i](n_unbind_data[i] - 1, j) = 0; 
                else
                    unbind_seqs[i](n_unbind_data[i] - 1, j) = 1;
            } 

            // The second entry is the cleavage metric being parsed 
            std::string token; 
            std::getline(ss, token, '\t'); 
            unbind_data[i](n_unbind_data[i] - 1) = std::stod(token);
        }
        infile.close();
        i++;
    }

    /** ------------------------------------------------------- //
     *                        SQP SETTINGS                      //
     *  --------------------------------------------------------*/  
    const PreciseType tau = 0.5;
    const PreciseType delta = 1e-8; 
    const PreciseType beta = 1e-4; 
    const int max_iter = 1000; 
    const PreciseType tol = 1e-8;       // Set the y-value tolerance to be small 
    const QuasiNewtonMethod method = QuasiNewtonMethod::BFGS;
    const RegularizationMethod regularize = RegularizationMethod::L2; 
    const PreciseType regularize_weight = 0.1; 
    const bool use_only_armijo = true;
    const bool use_strong_wolfe = false;
    const int hessian_modify_max_iter = 10000;
    const PreciseType c1 = 1e-4;
    const PreciseType c2 = 0.9;
    const PreciseType x_tol = 10000;    // Set the x-value tolerance to be large 
    const bool verbose = false;

    /** ------------------------------------------------------- //
     *         FIT AGAINST GIVEN CLEAVAGE/UNBINDING DATA        //
     *  --------------------------------------------------------*/
    // Collect separate sets of best-fit parameter values for each of the 
    // three Cas9 variants being tested  
    Matrix<PreciseType, Dynamic, Dynamic> best_fit(n_init, 3 * D); 
    Matrix<PreciseType, Dynamic, 1> x_init, l_init;
    Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats(n_init, 3 * 5 * (length + 1));
    Matrix<PreciseType, Dynamic, 3> errors(n_init, 3);
    MatrixXi single_mismatch_seqs_plus_perfect = MatrixXi::Ones(length + 1, length);
    single_mismatch_seqs_plus_perfect(Eigen::seqN(1, length), Eigen::all) -= MatrixXi::Identity(length, length); 
    for (int j = 0; j < 3; ++j)
    {
        MatrixXi curr_cleave_seqs = cleave_seqs[j]; 
        MatrixXi curr_unbind_seqs = unbind_seqs[j];
        Matrix<PreciseType, Dynamic, 1> curr_cleave_data = cleave_data[j];
        Matrix<PreciseType, Dynamic, 1> curr_unbind_data = unbind_data[j];
        for (int i = 0; i < n_init; ++i)
        {
            // Assemble initial parameter values
            x_init = init_points.row(i); 
            l_init = (
                Matrix<PreciseType, Dynamic, 1>::Ones(N)
                - constraints_opt->active(x_init.cast<mpq_rational>()).template cast<PreciseType>()
            );

            // Get the best-fit parameter values from the i-th initial parameter 
            // values for the j-th Cas9 variant
            best_fit(i, Eigen::seqN(j * D, D)) = opt->run([
                &curr_cleave_seqs, &curr_unbind_seqs, &curr_cleave_data, &curr_unbind_data,
                &bind_conc 
            ](
                const Ref<const Matrix<PreciseType, Dynamic, 1> >& x
            )
                {
                    return errorAgainstData(
                        x, curr_cleave_seqs, curr_cleave_data, curr_unbind_seqs,
                        curr_unbind_data, bind_conc, true 
                    ); 
                },
                x_init, l_init, tau, delta, beta, max_iter, tol, x_tol, method,
                regularize, regularize_weight, use_only_armijo, use_strong_wolfe,
                hessian_modify_max_iter, c1, c2, verbose
            );
            errors(i, j) = errorAgainstData(
                best_fit(i, Eigen::seqN(j * D, D)), curr_cleave_seqs, curr_cleave_data, 
                curr_unbind_seqs, curr_unbind_data, bind_conc, true
            );

            // Then compute the *unnormalized* cleavage statistics of the
            // best-fit model against all single-mismatch substrates (and 
            // the perfect-match substrate) 
            Matrix<PreciseType, Dynamic, 5> curr_fit_stats = computeCleavageStats(
                best_fit(i, Eigen::seqN(j * D, D)), single_mismatch_seqs_plus_perfect,
                bind_conc, false
            );
            Matrix<PreciseType, 5, 1> curr_fit_stats_perfect = curr_fit_stats.row(0); 
            fit_single_mismatch_stats(i, Eigen::seqN(j * 5 * (length + 1), 5)) = curr_fit_stats_perfect;
            for (int k = 0; k < length; ++k)
            {
                Matrix<PreciseType, 5, 1> curr_fit_stats_mismatched = curr_fit_stats.row(k + 1);
                fit_single_mismatch_stats(i, j * 5 * (length + 1) + 5 * (k + 1))
                    = log10(curr_fit_stats_perfect(0)) - log10(curr_fit_stats_mismatched(0));
                fit_single_mismatch_stats(i, j * 5 * (length + 1) + 5 * (k + 1) + 1)
                    = log10(curr_fit_stats_perfect(1)) - log10(curr_fit_stats_mismatched(1)); 
                fit_single_mismatch_stats(i, j * 5 * (length + 1) + 5 * (k + 1) + 2)
                    = log10(curr_fit_stats_mismatched(2)) - log10(curr_fit_stats_perfect(2));
                fit_single_mismatch_stats(i, j * 5 * (length + 1) + 5 * (k + 1) + 3)
                    = log10(curr_fit_stats_mismatched(3)) - log10(curr_fit_stats_perfect(3));
                // Note that computeCleavageStats() returns composite cleavage *times*, not rates  
                fit_single_mismatch_stats(i, j * 5 * (length + 1) + 5 * (k + 1) + 4)
                    = log10(curr_fit_stats_mismatched(4)) - log10(curr_fit_stats_perfect(4));   
            }
        }
    }
    
    // Output the fits to file 
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    std::vector<std::string> variants = {"SpCas9", "SpCas9HF1", "eSpCas9"};
    outfile << "fit_attempt\t"; 
    for (const std::string& variant : variants)
    { 
        outfile << variant << "_forward_match\t" << variant << "_reverse_match\t"
                << variant << "_forward_mismatch\t" << variant << "_reverse_mismatch\t"
                << variant << "_terminal_unbind_rate\t"
                << variant << "_terminal_cleave_rate\t"
                << variant << "_terminal_bind_rate\t"
                << variant << "_error\t";
    }
    for (const std::string& variant : variants)
    { 
        outfile << variant << "_perfect_cleave_prob\t"
                << variant << "_perfect_cleave_rate\t"
                << variant << "_perfect_dead_unbind_rate\t"
                << variant << "_perfect_live_unbind_rate\t"
                << variant << "_perfect_composite_cleave_rate\t";
        for (int i = 0; i < length; ++i)
        {
            outfile << variant << "_mm" << i << "_log_spec\t"
                    << variant << "_mm" << i << "_log_rapid\t"
                    << variant << "_mm" << i << "_log_dead_dissoc\t"
                    << variant << "_mm" << i << "_log_live_dissoc\t"
                    << variant << "_mm" << i << "_log_composite_rapid";
            if (variant == "eSpCas9" && i == length - 1)
                outfile << std::endl;
            else 
                outfile << '\t';
        }
    }
    for (int i = 0; i < n_init; ++i)
    {
        outfile << i << '\t'; 

        // Write, for each variant, each best-fit parameter vector ...
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < D; ++k)
                outfile << best_fit(i, j * D + k) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors(i, j) << '\t';
        }

        // ... along with the associated perfect-match and single-mismatch
        // cleavage statistics
        for (int j = 0; j < 3 * 5 * (length + 1) - 1; ++j)
            outfile << fit_single_mismatch_stats(i, j) << '\t';
        outfile << fit_single_mismatch_stats(i, 3 * 5 * (length + 1) - 1) << std::endl;  
    }
    outfile.close();

    delete tri; 
    delete opt;
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    const int n_init = std::stoi(argv[1]);
    const PreciseType bind_conc = static_cast<PreciseType>(std::stod(argv[2]));  
    fitCleavageStats(
        "data/Singh-2018-NatStructMolBiol-fits.tsv", 
        bind_conc, n_init, rng
    );
} 

