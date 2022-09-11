/**
 * Using line-search SQP, identify the set of line-graph Cas9 model parameter 
 * vectors that yields each given set of unbinding and cleavage rates in the 
 * given files.  
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     9/11/2022
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
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
    PreciseType terminal_unbind_rate = 1;
    PreciseType terminal_cleave_rate = pow(ten, logrates(4)); 

    // Binding rate entering state 0
    PreciseType bind_rate = pow(ten, logrates(5));

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
                             const PreciseType bind_conc, const bool normalize = true)
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
        PreciseType error = 0;
        error += (stats1.col(4) - cleave_data2).squaredNorm();
        error += (stats2.col(2).array().pow(-1).matrix() - unbind_data2).squaredNorm();
        return error;
    }
    else 
    {
        // Otherwise, compute *unnormalized* cleavage metrics 
        stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc, false);
        stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc, false);

        // Note that computeCleavageStats() returns composite cleavage *times* and 
        // dead unbinding *rates*, whereas the data includes only *times* and no *rates*
        PreciseType error = 0;
        error += (stats1.col(4) - cleave_data).squaredNorm();
        error += (stats2.col(2).array().pow(-1).matrix() - unbind_data).squaredNorm();
        return error;
    }
}

void fitLineParamsAgainstMeasuredRates(const std::string cleave_infilename,
                                       const std::string unbind_infilename,
                                       const std::string outfilename,
                                       const PreciseType bind_conc,
                                       const int n_init, boost::random::mt19937& rng)
{
    // Parse measured cleavage rates and dead unbinding rates, along with the
    // mismatched sequences on which they were measured 
    int n_cleave_data = 0; 
    int n_unbind_data = 0;
    MatrixXi cleave_seqs = MatrixXi::Zero(0, length); 
    MatrixXi unbind_seqs = MatrixXi::Zero(0, length); 
    Matrix<PreciseType, Dynamic, 1> cleave_data = Matrix<PreciseType, Dynamic, 1>::Zero(0); 
    Matrix<PreciseType, Dynamic, 1> unbind_data = Matrix<PreciseType, Dynamic, 1>::Zero(0);
    
    // Parse the input file of (composite) cleavage rates, if one is given 
    std::ifstream infile;
    std::string line;  
    if (cleave_infilename != "")
    {
        infile.open(cleave_infilename);
        while (std::getline(infile, line))
        {
            std::stringstream ss;
            ss << line;

            // The first entry is the binary sequence
            std::string seq;
            std::getline(ss, seq, '\t');
            n_cleave_data++;
            cleave_seqs.conservativeResize(n_cleave_data, length);  
            cleave_data.conservativeResize(n_cleave_data);

            // Parse the binary sequence, character by character
            if (seq.size() != length)
                throw std::runtime_error("Parsed binary sequence of invalid length (!= 20)"); 
            for (int j = 0; j < length; ++j)
            {
                if (seq[j] == '0')
                    cleave_seqs(n_cleave_data - 1, j) = 0; 
                else
                    cleave_seqs(n_cleave_data - 1, j) = 1;
            }

            // The second entry is the cleavage rate
            std::string token; 
            std::getline(ss, token, '\t'); 
            cleave_data(n_cleave_data - 1) = std::stod(token);
        }
        infile.close();
    }

    // Parse the input file of unbinding rates, if one is given
    if (unbind_infilename != "")
    {
        infile.open(unbind_infilename); 
        while (std::getline(infile, line))
        {
            std::stringstream ss;
            ss << line;

            // The first entry is the binary sequence
            std::string seq;
            std::getline(ss, seq, '\t');
            n_unbind_data++;
            unbind_seqs.conservativeResize(n_unbind_data, length); 
            unbind_data.conservativeResize(n_unbind_data);

            // Parse the binary sequence, character by character
            if (seq.size() != length)
                throw std::runtime_error("Parsed binary sequence of invalid length (!= 20)"); 
            for (int j = 0; j < length; ++j)
            {
                if (seq[j] == '0')
                    unbind_seqs(n_unbind_data - 1, j) = 0; 
                else
                    unbind_seqs(n_unbind_data - 1, j) = 1;
            } 

            // The second entry is the unbinding rate being parsed
            std::string token; 
            std::getline(ss, token, '\t'); 
            unbind_data(n_unbind_data - 1) = std::stod(token);
        }
        infile.close();
    }

    // Exit if no cleavage rates and no unbinding rates were specified 
    if (n_cleave_data == 0 && n_unbind_data == 0)
        return; 

    // Set up an SQPOptimizer instance that constrains all parameters to lie 
    // between 1e-10 and 1e+10 
    Polytopes::LinearConstraints<mpq_rational>* constraints_opt = new Polytopes::LinearConstraints<mpq_rational>(
        Polytopes::InequalityType::GreaterThanOrEqualTo 
    );
    constraints_opt->parse("polytopes/line-10-unbindingunity-plusbind.poly");
    const int D = constraints_opt->getD(); 
    const int N = constraints_opt->getN(); 
    SQPOptimizer<PreciseType>* opt = new SQPOptimizer<PreciseType>(constraints_opt);

    // Sample a set of points from an initial polytope in parameter space, 
    // which constrains all parameters to lie between 1e-5 and 1e+5
    Delaunay_triangulation* tri = new Delaunay_triangulation(D);
    Polytopes::parseVerticesFile("polytopes/line-5-unbindingunity-plusbind.vert", tri); 
    Matrix<PreciseType, Dynamic, Dynamic> init_points
        = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, n_init, 0, rng);

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
    const bool verbose = true;

    /** ------------------------------------------------------- //
     *         FIT AGAINST GIVEN CLEAVAGE/UNBINDING DATA        //
     *  --------------------------------------------------------*/
    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<PreciseType, Dynamic, Dynamic> best_fit(n_init, D); 
    Matrix<PreciseType, Dynamic, 1> x_init, l_init;
    Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats(n_init, 5 * (length + 1));
    Matrix<PreciseType, Dynamic, 1> errors(n_init);
    MatrixXi single_mismatch_seqs_plus_perfect = MatrixXi::Ones(length + 1, length);
    single_mismatch_seqs_plus_perfect(Eigen::seqN(1, length), Eigen::all) -= MatrixXi::Identity(length, length); 
    for (int i = 0; i < n_init; ++i)
    {
        // Assemble initial parameter values
        x_init = init_points.row(i); 
        l_init = (
            Matrix<PreciseType, Dynamic, 1>::Ones(N)
            - constraints_opt->active(x_init.cast<mpq_rational>()).template cast<PreciseType>()
        );

        // Get the best-fit parameter values from the i-th initial parameter vector
        best_fit.row(i) = opt->run([
            &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data, &bind_conc 
        ](
            const Ref<const Matrix<PreciseType, Dynamic, 1> >& x
        )
            {
                return errorAgainstData(
                    x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    bind_conc, true 
                ); 
            },
            x_init, l_init, tau, delta, beta, max_iter, tol, x_tol, method,
            regularize, regularize_weight, use_only_armijo, use_strong_wolfe,
            hessian_modify_max_iter, c1, c2, verbose
        );
        errors(i) = errorAgainstData(
            best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
            bind_conc, true
        );

        // Then compute the *unnormalized* cleavage statistics of the
        // best-fit model against all single-mismatch substrates (and 
        // the perfect-match substrate) 
        Matrix<PreciseType, Dynamic, 5> fit_stats = computeCleavageStats(
            best_fit.row(i), single_mismatch_seqs_plus_perfect, bind_conc, false
        );
        Matrix<PreciseType, 5, 1> fit_stats_perfect = fit_stats.row(0); 
        fit_single_mismatch_stats(i, Eigen::seqN(0, 5)) = fit_stats_perfect;
        for (int k = 0; k < length; ++k)
        {
            Matrix<PreciseType, 5, 1> fit_stats_mismatched = fit_stats.row(k + 1);
            fit_single_mismatch_stats(i, 5 * (k + 1))
                = log10(fit_stats_perfect(0)) - log10(fit_stats_mismatched(0));
            fit_single_mismatch_stats(i, 5 * (k + 1) + 1)
                = log10(fit_stats_perfect(1)) - log10(fit_stats_mismatched(1)); 
            fit_single_mismatch_stats(i, 5 * (k + 1) + 2)
                = log10(fit_stats_mismatched(2)) - log10(fit_stats_perfect(2));
            fit_single_mismatch_stats(i, 5 * (k + 1) + 3)
                = log10(fit_stats_mismatched(3)) - log10(fit_stats_perfect(3));
            // Note that computeCleavageStats() returns composite cleavage *times*, not rates  
            fit_single_mismatch_stats(i, 5 * (k + 1) + 4)
                = log10(fit_stats_mismatched(4)) - log10(fit_stats_perfect(4));   
        }
    }
    
    // Output the fits to file 
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    outfile << "fit_attempt\t"; 
    outfile << "forward_match\treverse_match\tforward_mismatch\treverse_mismatch\t"
            << "terminal_cleave_rate\tterminal_bind_rate\terror\t";
    outfile << "perfect_cleave_prob\tperfect_cleave_rate\tperfect_dead_unbind_rate\t"
            << "perfect_live_unbind_rate\tperfect_composite_cleave_rate\t";
    for (int i = 0; i < length; ++i)
    {
        outfile << "mm" << i << "_log_spec\t"
                << "mm" << i << "_log_rapid\t"
                << "mm" << i << "_log_dead_dissoc\t"
                << "mm" << i << "_log_live_dissoc\t"
                << "mm" << i << "_log_composite_rapid";
        if (i == length - 1)
            outfile << std::endl;
        else 
            outfile << '\t';
    }
    for (int i = 0; i < n_init; ++i)
    {
        outfile << i << '\t'; 

        // Write each best-fit parameter vector ...
        for (int j = 0; j < D; ++j)
            outfile << best_fit(i, j) << '\t';

        // ... along with the associated error against the corresponding data ... 
        outfile << errors(i) << '\t';

        // ... along with the associated perfect-match and single-mismatch
        // cleavage statistics
        for (int j = 0; j < 5 * (length + 1) - 1; ++j)
            outfile << fit_single_mismatch_stats(i, j) << '\t';
        outfile << fit_single_mismatch_stats(i, 5 * (length + 1) - 1) << std::endl;  
    }
    outfile.close();

    delete tri; 
    delete opt;
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    const int n_init = std::stoi(argv[4]);
    const PreciseType bind_conc = static_cast<PreciseType>(std::stod(argv[5]));  
    fitLineParamsAgainstMeasuredRates(argv[1], argv[2], argv[3], bind_conc, n_init, rng);
} 

