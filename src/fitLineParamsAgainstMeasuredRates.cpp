/**
 * Using line-search SQP, identify the set of line-graph Cas9 model parameter 
 * vectors that yields each given set of unbinding and cleavage rates in the 
 * given files.  
 *
 * **Author:**
 *     Kee-Myoung Nam 
 *
 * **Last updated:**
 *     9/13/2022
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
 * Return a collection of index-sets for the given number of folds into which
 * to divide a dataset of the given size. 
 *
 * @param n      Number of data points. 
 * @param nfolds Number of folds into which to divide the dataset. 
 * @returns      Collection of index-sets, each corresponding to the training
 *               and test subset corresponding to each fold. 
 */
std::vector<std::pair<std::vector<int>, std::vector<int> > > getFolds(const int n,
                                                                      const int nfolds)
{
    int foldsize = static_cast<int>(n / nfolds);
    std::vector<std::pair<std::vector<int>, std::vector<int> > > fold_pairs; 

    // Define the first (nfolds - 1) folds ... 
    int start = 0;  
    for (int i = 0; i < nfolds - 1; ++i)
    {
        std::vector<int> train_fold, test_fold;
        for (int j = 0; j < start; ++j)
            train_fold.push_back(j);  
        for (int j = start; j < start + foldsize; ++j)
            test_fold.push_back(j);
        for (int j = start + foldsize; j < n; ++j)
            train_fold.push_back(j);  
        fold_pairs.emplace_back(std::make_pair(train_fold, test_fold)); 
        start += foldsize; 
    }

    // ... then define the last fold
    std::vector<int> last_train_fold, last_test_fold;
    for (int i = 0; i < start; ++i)
        last_train_fold.push_back(i);  
    for (int i = start; i < n; ++i)
        last_test_fold.push_back(i);  
    fold_pairs.emplace_back(std::make_pair(last_train_fold, last_test_fold)); 

    return fold_pairs;  
}

/**
 * Return a randomly generated permutation of the range [0, 1, ..., n - 1],
 * using the Fisher-Yates shuffle.
 */
PermutationMatrix<Dynamic, Dynamic> getPermutation(const int n, boost::random::mt19937& rng)
{
    VectorXi p(n); 
    for (int i = 0; i < n; ++i)
        p(i) = i;
    
    for (int i = 0; i < n - 1; ++i)
    {
        // Generate a random integer between i and n - 1 (inclusive)
        boost::random::uniform_int_distribution<> dist(i, n - 1);
        int j = dist(rng); 
        
        // Swap p[i] and p[j]
        int tmp = p(j); 
        p(j) = p(i); 
        p(i) = tmp;
    }

    // Convert the vector into a permutation matrix 
    PermutationMatrix<Dynamic, Dynamic> perm(n);
    perm.indices() = p; 
    
    return perm; 
}

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

/**
 * @param cleave_data
 * @param unbind_data
 * @param cleave_seqs
 * @param unbind_seqs
 * @param ninit
 * @param bind_conc
 * @param logscale
 * @param rng
 * @param tau
 * @param delta
 * @param beta
 * @param max_iter
 * @param tol
 * @param method
 * @param regularize
 * @param regularize_weight
 * @param use_only_armijo
 * @param use_strong_wolfe
 * @param hessian_modify_max_iter
 * @param c1
 * @param c2
 * @param x_tol
 * @param verbose
 */
std::tuple<Matrix<PreciseType, Dynamic, Dynamic>, 
           Matrix<PreciseType, Dynamic, Dynamic>,
           Matrix<PreciseType, Dynamic, 1> >
    fitLineParamsAgainstMeasuredRates(const Ref<const Matrix<PreciseType, Dynamic, 1> >& cleave_data, 
                                      const Ref<const Matrix<PreciseType, Dynamic, 1> >& unbind_data, 
                                      const Ref<const MatrixXi>& cleave_seqs, 
                                      const Ref<const MatrixXi>& unbind_seqs,
                                      const int ninit, const PreciseType bind_conc, 
                                      const bool logscale, boost::random::mt19937& rng,
                                      const PreciseType tau, const PreciseType delta,
                                      const PreciseType beta, const int max_iter,
                                      const PreciseType tol, const QuasiNewtonMethod method, 
                                      const RegularizationMethod regularize,
                                      const PreciseType regularize_weight, 
                                      const bool use_only_armijo,
                                      const bool use_strong_wolfe, 
                                      const int hessian_modify_max_iter, 
                                      const PreciseType c1, const PreciseType c2,
                                      const PreciseType x_tol, const bool verbose) 
{
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
        = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, ninit, 0, rng);

    // Collect best-fit parameter values for each of the initial parameter vectors 
    // from which to begin each round of optimization 
    Matrix<PreciseType, Dynamic, Dynamic> best_fit(ninit, D); 
    Matrix<PreciseType, Dynamic, 1> x_init, l_init;
    Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats(ninit, 4 * length);
    Matrix<PreciseType, Dynamic, 1> errors(ninit);
    MatrixXi single_mismatch_seqs = MatrixXi::Ones(length, length) - MatrixXi::Identity(length, length); 
    for (int i = 0; i < ninit; ++i)
    {
        // Assemble initial parameter values
        x_init = init_points.row(i); 
        l_init = (
            Matrix<PreciseType, Dynamic, 1>::Ones(N)
            - constraints_opt->active(x_init.cast<mpq_rational>()).template cast<PreciseType>()
        );

        // Get the best-fit parameter values from the i-th initial parameter vector
        best_fit.row(i) = opt->run([
            &cleave_seqs, &unbind_seqs, &cleave_data, &unbind_data,
            &bind_conc, &logscale
        ](
            const Ref<const Matrix<PreciseType, Dynamic, 1> >& x
        )
            {
                return errorAgainstData(
                    x, cleave_seqs, cleave_data, unbind_seqs, unbind_data,
                    bind_conc, logscale
                ); 
            },
            x_init, l_init, tau, delta, beta, max_iter, tol, x_tol, method,
            regularize, regularize_weight, use_only_armijo, use_strong_wolfe,
            hessian_modify_max_iter, c1, c2, verbose
        );
        errors(i) = errorAgainstData(
            best_fit.row(i), cleave_seqs, cleave_data, unbind_seqs, unbind_data,
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
    delete tri; 
    delete opt;

    return std::make_tuple(best_fit, fit_single_mismatch_stats, errors);
}

int main(int argc, char** argv)
{
    boost::random::mt19937 rng(1234567890);

    // Input file of measured cleavage rates/times
    const std::string cleave_infilename = (strcmp(argv[1], "NULL") == 0 ? "" : argv[1]);

    // Input file of measured unbinding rates/times
    const std::string unbind_infilename = (strcmp(argv[2], "NULL") == 0 ? "" : argv[2]);

    // Path to output file  
    const std::string outfilename = argv[3];

    // Whether each input file specifies rates (false) or times (true)
    const bool data_specified_as_times = !(strcmp(argv[4], "0") == 0 || strcmp(argv[4], "false") == 0); 

    // Number of initial parameter vectors from which to run optimization 
    const int ninit = std::stoi(argv[5]);

    // Number of folds for k-fold cross-validation (no cross-validation if 1, 
    // invalid if < 0)
    const int nfolds = std::stoi(argv[6]);
    if (nfolds <= 0)
        throw std::invalid_argument("Invalid number of folds specified");  

    // Binding concentration 
    const PreciseType bind_conc = static_cast<PreciseType>(std::stod(argv[7]));

    // Whether to evaluate the error function in log-scale 
    const bool logscale = false;
    
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
            try
            {
                cleave_data(n_cleave_data - 1) = std::stod(token);
            }
            catch (const std::out_of_range& e)
            {
                cleave_data(n_cleave_data - 1) = 0; 
            }
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
            try
            {
                unbind_data(n_unbind_data - 1) = std::stod(token);
            }
            catch (const std::out_of_range& e)
            {
                unbind_data(n_unbind_data - 1) = 0; 
            }
        }
        infile.close();
    }

    // Exit if no cleavage rates and no unbinding rates were specified 
    if (n_cleave_data == 0 && n_unbind_data == 0)
        throw std::runtime_error("Both cleavage rate and unbinding rate datasets are empty");

    // Shuffle the rows of the datasets to ensure that there are no biases 
    // in sequence composition 
    PermutationMatrix<Dynamic, Dynamic> cleave_perm = getPermutation(n_cleave_data, rng);  
    PermutationMatrix<Dynamic, Dynamic> unbind_perm = getPermutation(n_unbind_data, rng);
    cleave_data = cleave_perm * cleave_data; 
    cleave_seqs = cleave_perm * cleave_seqs; 
    unbind_data = unbind_perm * unbind_data; 
    unbind_seqs = unbind_perm * unbind_seqs;  

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

    /** ----------------------------------------------------------------------- //
     *     RUN K-FOLD CROSS VALIDATION AGAINST GIVEN CLEAVAGE/UNBINDING DATA    //
     *  ----------------------------------------------------------------------- */
    std::tuple<Matrix<PreciseType, Dynamic, Dynamic>, 
               Matrix<PreciseType, Dynamic, Dynamic>, 
               Matrix<PreciseType, Dynamic, 1> > results;
    if (nfolds == 1)
    {
        results = fitLineParamsAgainstMeasuredRates(
            cleave_data_norm, unbind_data_norm, cleave_seqs, unbind_seqs, ninit, 
            bind_conc, logscale, rng, tau, delta, beta, max_iter, tol, method,
            regularize, regularize_weight, use_only_armijo, use_strong_wolfe,
            hessian_modify_max_iter, c1, c2, x_tol, verbose
        );
        Matrix<PreciseType, Dynamic, Dynamic> best_fit = std::get<0>(results); 
        Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats = std::get<1>(results); 
        Matrix<PreciseType, Dynamic, 1> errors = std::get<2>(results); 

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
        for (int i = 0; i < ninit; ++i)
        {
            outfile << i << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fit.cols(); ++j)
                outfile << best_fit(i, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors(i) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            for (int j = 0; j < 4 * length - 1; ++j)
                outfile << fit_single_mismatch_stats(i, j) << '\t';
            outfile << fit_single_mismatch_stats(i, 4 * length - 1) << std::endl;  
        }
        outfile.close();
    }
    else
    {
        // Divide the indices in each dataset into folds
        auto unbind_fold_pairs = getFolds(n_unbind_data, nfolds);
        auto cleave_fold_pairs = getFolds(n_cleave_data, nfolds); 
        
        // For each fold ...
        Matrix<PreciseType, Dynamic, 1> unbind_data_train, unbind_data_test,
                                        cleave_data_train, cleave_data_test;
        MatrixXi unbind_seqs_train, unbind_seqs_test, cleave_seqs_train, cleave_seqs_test;
        Matrix<PreciseType, Dynamic, Dynamic> best_fit_total(nfolds, 0); 
        Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats_total(nfolds, 4 * length); 
        Matrix<PreciseType, Dynamic, 1> errors_against_test(nfolds); 
        for (int fi = 0; fi < nfolds; ++fi)
        {
            unbind_data_train = unbind_data_norm(unbind_fold_pairs[fi].first);
            unbind_data_test = unbind_data_norm(unbind_fold_pairs[fi].second);
            unbind_seqs_train = unbind_seqs(unbind_fold_pairs[fi].first, Eigen::all);
            unbind_seqs_test = unbind_seqs(unbind_fold_pairs[fi].second, Eigen::all); 
            cleave_data_train = cleave_data_norm(cleave_fold_pairs[fi].first); 
            cleave_data_test = cleave_data_norm(cleave_fold_pairs[fi].second); 
            cleave_seqs_train = cleave_seqs(cleave_fold_pairs[fi].first, Eigen::all); 
            cleave_seqs_test = cleave_seqs(cleave_fold_pairs[fi].second, Eigen::all);

            // Optimize model parameters on the training subset 
            results = fitLineParamsAgainstMeasuredRates(
                cleave_data_train, unbind_data_train, cleave_seqs_train,
                unbind_seqs_train, ninit, bind_conc, logscale, rng, tau, delta,
                beta, max_iter, tol, method, regularize, regularize_weight,
                use_only_armijo, use_strong_wolfe, hessian_modify_max_iter,
                c1, c2, x_tol, verbose
            );
            Matrix<PreciseType, Dynamic, Dynamic> best_fit_per_fold = std::get<0>(results); 
            Matrix<PreciseType, Dynamic, Dynamic> fit_single_mismatch_stats_per_fold = std::get<1>(results); 
            Matrix<PreciseType, Dynamic, 1> errors_per_fold = std::get<2>(results);
            if (fi == 0) 
                best_fit_total.resize(nfolds, best_fit_per_fold.cols()); 

            // Find the parameter vector corresponding to the least error
            Eigen::Index minidx; 
            PreciseType minerror = errors_per_fold.minCoeff(&minidx); 
            Matrix<PreciseType, Dynamic, 1> best_fit = best_fit_per_fold.row(minidx); 
            Matrix<PreciseType, Dynamic, 1> fit_single_mismatch_stats = fit_single_mismatch_stats_per_fold.row(minidx);

            // Evaluate error against the test subset 
            PreciseType error_against_test = errorAgainstData(
                best_fit, cleave_seqs_test, cleave_data_test, unbind_seqs_test,
                unbind_data_test, bind_conc, logscale
            );
            best_fit_total.row(fi) = best_fit; 
            fit_single_mismatch_stats_total.row(fi) = fit_single_mismatch_stats; 
            errors_against_test(fi) = error_against_test; 
        }

        // Output the fits to file 
        std::ofstream outfile(outfilename); 
        outfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        outfile << "fold\t";
        outfile << "forward_match\treverse_match\tforward_mismatch\treverse_mismatch\t"
                << "terminal_cleave_rate\tterminal_bind_rate\ttest_error\t";
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
        for (int fi = 0; fi < nfolds; ++fi)
        {
            outfile << fi << '\t'; 

            // Write each best-fit parameter vector ...
            for (int j = 0; j < best_fit_total.cols(); ++j)
                outfile << best_fit_total(fi, j) << '\t';

            // ... along with the associated error against the corresponding data ... 
            outfile << errors_against_test(fi) << '\t';

            // ... along with the associated single-mismatch cleavage statistics
            for (int j = 0; j < 4 * length - 1; ++j)
                outfile << fit_single_mismatch_stats_total(fi, j) << '\t';
            outfile << fit_single_mismatch_stats_total(fi, 4 * length - 1) << std::endl;  
        }
        outfile.close();
    }

    return 0; 
} 

