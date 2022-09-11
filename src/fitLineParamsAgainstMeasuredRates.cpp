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
Matrix<PreciseType, Dynamic, 4> computeCleavageStats(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                                                     const Ref<const MatrixXi>& seqs, 
                                                     const PreciseType bind_conc)
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
    // live unbinding rate, and composite cleavage time against each given
    // mismatched substrate
    Matrix<PreciseType, Dynamic, 4> stats(seqs.rows(), 4);  
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
        PreciseType unbind_rate = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(j, 3) = (
            1 / (bind_conc * bind_rate)
            + (1 / unbind_rate + 1 / (bind_conc * bind_rate)) * (1 - stats(j, 0)) / stats(j, 0) + 1 / stats(j, 1)
        );

        // Inverse specificity = cleavage probability on mismatched / cleavage probability on perfect
        stats(j, 0) = log10(stats(j, 0)) - log10(stats_perfect(0)); 

        // Inverse rapidity = cleavage rate on mismatched / cleavage rate on perfect
        stats(j, 1) = log10(stats(j, 1)) - log10(stats_perfect(1));  

        // Unbinding rate on mismatched > unbinding rate on perfect, so 
        // return perfect rate / mismatched rate (inverse dissociativity) 
        stats(j, 2) = log10(stats_perfect(2)) - log10(stats(j, 2));
        
        // Cleavage *time* on mismatched > cleavage *time* on perfect (usually), 
        // so return perfect time / mismatched time (inverse composite cleavage
        // time ratio)
        stats(j, 3) = log10(stats_perfect(4)) - log10(stats(j, 3));
    }

    delete model;
    return stats;
}

/**
 * Compute the (unregularized) error between a set of (composite) cleavage
 * rates and dead unbinding rates inferred from the line-graph Cas9 model
 * and a set of experimentally determined cleavage rates/times and dead
 * unbinding rates/times.
 *
 * Note that regularization, if desired, should be built into the optimizer
 * function/class with which this function will be used.  
 */
PreciseType errorAgainstData(const Ref<const Matrix<PreciseType, Dynamic, 1> >& logrates,
                             const Ref<const MatrixXi>& cleave_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data,
                             const Ref<const MatrixXi>& unbind_seqs,
                             const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data,
                             const PreciseType bind_conc, const bool logscale = false)
{
    Matrix<PreciseType, Dynamic, 4> stats1, stats2; 

    // Compute *normalized* cleavage metrics and corresponding error against data
    stats1 = computeCleavageStats(logrates, cleave_seqs, bind_conc);
    stats2 = computeCleavageStats(logrates, unbind_seqs, bind_conc);
    PreciseType error = 0;
    if (logscale)               // Compute error function in log-scale
    {
        error += (stats1.col(3) - cleave_data.array().log10().matrix()).squaredNorm();
        error += (stats2.col(2) - unbind_data.array().log10().matrix()).squaredNorm();
    }
    else                        // Compute error function in linear scale  
    {
        Matrix<PreciseType, Dynamic, 4> stats1_transformed(stats1.rows(), 4);
        Matrix<PreciseType, Dynamic, 4> stats2_transformed(stats2.rows(), 4);
        for (int j = 0; j < 4; ++j)
        {
            for (int i = 0; i < stats1.rows(); ++i)
                stats1_transformed(i, j) = pow(ten, stats1(i, j)); 
            for (int i = 0; i < stats2.rows(); ++i)
                stats2_transformed(i, j) = pow(ten, stats2(i, j));
        }
        error += (stats1_transformed.col(3) - cleave_data).squaredNorm(); 
        error += (stats2_transformed.col(2) - unbind_data).squaredNorm(); 
    }

    return error;
}

void fitLineParamsAgainstMeasuredRates(const std::string cleave_infilename,
                                       const std::string unbind_infilename,
                                       const std::string outfilename,
                                       const PreciseType bind_conc,
                                       const int n_init, boost::random::mt19937& rng,
                                       const bool logscale = false,
                                       const bool data_specified_as_times = false)
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

    // Assume that there exist cleavage and dead unbinding rates for the
    // perfect-match substrate (if any cleavage/unbinding rates have been
    // specified at all) 
    int unbind_perfect_index = -1;
    for (int i = 0; i < unbind_seqs.rows(); ++i)
    {
        if (unbind_seqs.row(i).sum() == length)
        {
            unbind_perfect_index = i;
            break; 
        }
    } 
    if (n_unbind_data > 0 && unbind_perfect_index == -1)
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
    if (n_cleave_data > 0 && cleave_perfect_index == -1)
    {
        throw std::runtime_error(
            "Cannot normalize given data without value corresponding to perfect-match substrate"
        );
    }

    // Normalize all given composite cleavage rates/times and dead unbinding
    // rates/times (with copies of the data matrices that were passed into
    // this function)
    Matrix<PreciseType, Dynamic, 1> unbind_data_norm(unbind_data.size());
    Matrix<PreciseType, Dynamic, 1> cleave_data_norm(cleave_data.size());
    if (data_specified_as_times)    // Data specified as times 
    {
        for (int i = 0; i < unbind_data_norm.size(); ++i)
            unbind_data_norm(i) = unbind_data(i) / unbind_data(unbind_perfect_index);   // rate on perfect / rate on mismatched
        for (int i = 0; i < cleave_data_norm.size(); ++i)
            cleave_data_norm(i) = cleave_data(cleave_perfect_index) / cleave_data(i);   // time on perfect / time on mismatched
    }
    else                            // Data specified as rates 
    { 
        for (int i = 0; i < unbind_data_norm.size(); ++i)
            unbind_data_norm(i) = unbind_data(unbind_perfect_index) / unbind_data(i);   // rate on perfect / rate on mismatched
        for (int i = 0; i < cleave_data_norm.size(); ++i)
            cleave_data_norm(i) = cleave_data(i) / cleave_data(cleave_perfect_index);   // time on perfect / time on mismatched
    }

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
    Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats(n_init, 4 * length);
    Matrix<PreciseType, Dynamic, 1> errors(n_init);
    MatrixXi single_mismatch_seqs = MatrixXi::Ones(length, length) - MatrixXi::Identity(length, length); 
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
            &cleave_seqs, &unbind_seqs, &cleave_data_norm, &unbind_data_norm,
            &bind_conc, &logscale
        ](
            const Ref<const Matrix<PreciseType, Dynamic, 1> >& x
        )
            {
                return errorAgainstData(
                    x, cleave_seqs, cleave_data_norm, unbind_seqs, unbind_data_norm,
                    bind_conc, logscale
                ); 
            },
            x_init, l_init, tau, delta, beta, max_iter, tol, x_tol, method,
            regularize, regularize_weight, use_only_armijo, use_strong_wolfe,
            hessian_modify_max_iter, c1, c2, verbose
        );
        errors(i) = errorAgainstData(
            best_fit.row(i), cleave_seqs, cleave_data_norm, unbind_seqs, unbind_data_norm,
            bind_conc, logscale
        );

        // Then compute the normalized cleavage statistics of the best-fit 
        // model against all single-mismatch substrates 
        Matrix<PreciseType, Dynamic, 4> fit_stats = computeCleavageStats(
            best_fit.row(i), single_mismatch_seqs, bind_conc
        );
        for (int j = 0; j < length; ++j)
        {
            for (int k = 0; k < 4; ++k)
            {
                // Invert all returned statistics  
                fit_single_mismatch_stats(i, 4 * j + k) = -fit_stats(j, k); 
            }
        }
    }
    
    // Output the fits to file 
    std::ofstream outfile(outfilename); 
    outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    outfile << "fit_attempt\t"; 
    outfile << "forward_match\treverse_match\tforward_mismatch\treverse_mismatch\t"
            << "terminal_cleave_rate\tterminal_bind_rate\terror\t";
    for (int i = 0; i < length; ++i)
    {
        outfile << "mm" << i << "_log_spec\t"
                << "mm" << i << "_log_rapid\t"
                << "mm" << i << "_log_deaddissoc\t"
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

        // ... along with the associated single-mismatch cleavage statistics
        for (int j = 0; j < 4 * length - 1; ++j)
            outfile << fit_single_mismatch_stats(i, j) << '\t';
        outfile << fit_single_mismatch_stats(i, 4 * length - 1) << std::endl;  
    }
    outfile.close();

    delete tri; 
    delete opt;
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);
    const int n_init = std::stoi(argv[5]);
    const PreciseType bind_conc = static_cast<PreciseType>(std::stod(argv[6]));
    fitLineParamsAgainstMeasuredRates(
        (strcmp(argv[1], "NULL") == 0 ? "" : argv[1]),
                     // Input file of measured cleavage rates/times
        (strcmp(argv[2], "NULL") == 0 ? "" : argv[2]),
                     // Input file of measured unbinding rates/times 
        argv[3],     // Path to output file 
        bind_conc,   // Nominal concentration of Cas9-RNA
        n_init,      // Number of initial parameter vectors from which to run optimization
        rng,         // Random number generator instance
        false,       // Perform optimization (compute error function) in log-scale
        !(strcmp(argv[4], "0") == 0 || strcmp(argv[4], "false") == 0)
                     // Whether each input file specifies rates (false) or times (true)
    );
} 

