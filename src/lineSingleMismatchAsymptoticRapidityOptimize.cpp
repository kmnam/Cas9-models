/**
 * Compute asymptotic rapidity tradeoff constants for the line-graph Cas9 
 * model with specifically chosen parameter values (d changing for various 
 * fixed choices of b and the terminal rates) with respect to single-mismatch
 * substrates. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     1/24/2023
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/random.hpp>
#include <graphs/line.hpp>
#include <linearConstraints.hpp>
#include <polytopes.hpp>
#include <vertexEnum.hpp>

using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::pow;
using boost::multiprecision::log; 
using boost::multiprecision::log1p; 
using boost::multiprecision::log10;
constexpr int INTERNAL_PRECISION = 100;
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;
const PreciseType ten("10");

const int length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

template <typename T>
T logsumexp(const T loga, const T logb, const T base)
{
    T two = static_cast<T>(2); 
    if (loga > logb) 
        return loga + log1p(pow(base, logb - loga)) / log(base);
    else if (logb > loga)
        return logb + log1p(pow(base, loga - logb)) / log(base);
    else 
        return (log(two) / log(base)) + loga; 
}

template <typename Derived>
typename Derived::Scalar logsumexp(const MatrixBase<Derived>& logx,
                                   const typename Derived::Scalar base)
{
    typedef typename Derived::Scalar T; 
    T maxlogx = logx.maxCoeff();
    Matrix<T, Dynamic, 1> x(logx.size());
    for (int i = 0; i < logx.size(); ++i)
        x(i) = pow(base, logx(i) - maxlogx); 
    
    return maxlogx + log(x.sum()) / log(base);  
}

/**
 * Compute the asymptotic specific rapidity tradeoff constants of randomly
 * parametrized line-graph Cas9 models with small b' / d' with respect to
 * single-mismatch substrates.
 */
Matrix<PreciseType, Dynamic, 1> computeLimitsForSmallMismatchRatio(const Ref<const VectorXd>& logrates,
                                                                   const double _logbp,
                                                                   const double _logdp)
{
    // Get DNA/RNA match parameters
    const PreciseType logb = static_cast<PreciseType>(logrates(0)); 
    const PreciseType logd = static_cast<PreciseType>(logrates(1)); 

    // Get DNA/RNA mismatch parameters
    const PreciseType logbp = static_cast<PreciseType>(_logbp);
    const PreciseType logdp = static_cast<PreciseType>(_logdp);

    // Get terminal rates
    const PreciseType terminal_unbind_lograte = static_cast<PreciseType>(logrates(2));
    const PreciseType terminal_cleave_lograte = static_cast<PreciseType>(logrates(3));
    const PreciseType terminal_lograte_sum = terminal_unbind_lograte + terminal_cleave_lograte;  

    // Ratios of match/mismatch parameters
    const PreciseType logc = logb - logd; 
    const PreciseType logcp = logbp - logdp;
    
    // Introduce single mismatches and compute asymptotic normalized statistics 
    // with respect to each single-mismatch substrate 
    Matrix<PreciseType, Dynamic, 1> stats(length);

    // Compute the partial sums of the form log(1), log(1 + c), ..., log(1 + c + ... + c^N)
    Matrix<PreciseType, Dynamic, 1> logc_powers(length + 1);
    Matrix<PreciseType, Dynamic, 1> logc_partial_sums(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        logc_powers(i) = i * logc;
    for (int i = 0; i < length + 1; ++i)
        logc_partial_sums(i) = logsumexp(logc_powers.head(i + 1), ten);

    // Compute the asymptotic rapidity tradeoff constants for all single-mismatch
    // substrates ...
    Matrix<PreciseType, Dynamic, 1> arr_alpha_m(length + 1);
    Matrix<PreciseType, Dynamic, 1> arr_alpha(length + 1);

    // Compute each term of alpha, which does not depend on the mismatch position m
    arr_alpha(0) = logsumexp<PreciseType>(
        0, logc_partial_sums(length - 1) + terminal_cleave_lograte - logd, ten
    );  
    for (int i = 1; i < length; ++i)
    {
        Matrix<PreciseType, Dynamic, 1> subarr(4); 
        subarr << logc_powers(i),
                  logc_partial_sums(i - 1) + terminal_unbind_lograte - logd,
                  logc_partial_sums(length - i - 1) + logc_powers(i) + terminal_cleave_lograte - logd,
                  logc_partial_sums(i - 1) + logc_partial_sums(length - i - 1) + terminal_lograte_sum - (2 * logd); 
        arr_alpha(i) = logsumexp(subarr, ten); 
    }
    arr_alpha(length) = logsumexp<PreciseType>(
        logc_powers(length),
        logc_partial_sums(length - 1) + terminal_unbind_lograte - logd, ten
    ); 
    PreciseType alpha = logsumexp(arr_alpha, ten);

    // Compute each term of alpha_m, for each mismatch position m 
    for (int m = 0; m < length; ++m)
    {
        for (int i = 0; i <= m; ++i)    // First sum in the formula (i = 0, ..., m)
        {
            PreciseType term1 = 0;
            PreciseType term2 = 0; 
            if (i >= 1)
            {
                term1 = logsumexp<PreciseType>(
                    i * logc,
                    logc_partial_sums(i - 1) + terminal_unbind_lograte - logd, 
                    ten
                ); 
            }
            if (m > length - 2)
            {
                term2 = logsumexp<PreciseType>(
                    0, logc_powers(length - 1 - m) + terminal_cleave_lograte - logdp, ten
                );
            } 
            else
            {
                Matrix<PreciseType, Dynamic, 1> subarr(3);
                subarr << 0,
                          logc_powers(length - 1 - m) + terminal_cleave_lograte - logdp,
                          logc_partial_sums(length - 2 - m) + terminal_cleave_lograte - logd; 
                term2 = logsumexp(subarr, ten); 
            }
            arr_alpha_m(i) = term1 + term2;
        }
        for (int i = m + 1; i <= length; ++i)    // Second sum in the formula
        {
            PreciseType term1 = 0;
            PreciseType term2 = 0;
            if (i <= length - 1)
            {
                term1 = logsumexp<PreciseType>(
                    0,
                    logc_partial_sums(length - 1 - i) + terminal_cleave_lograte - logd,
                    ten
                );
            }
            if (m > i - 2)
            {
                term2 = logc_powers(i - 1 - m) + terminal_unbind_lograte - logdp; 
            }
            else 
            {
                term2 = logsumexp<PreciseType>(
                    logc_powers(i - 1 - m) + terminal_unbind_lograte - logdp,
                    logc_partial_sums(i - 2 - m) + terminal_unbind_lograte - logd, 
                    ten
                );
            }
            arr_alpha_m(i) = term1 + term2;
        }
        stats(m) = logsumexp(arr_alpha_m, ten) - alpha + logc - logcp;
    }

    return stats;
}

/**
 * Given a set of equality constraints for a set of variables defining a 
 * polytope, return the lower-dimensional polytope that incorporates the 
 * inequality constraints.  
 */
Polytopes::LinearConstraints* constrainPolytope(Polytopes::LinearConstraints* constraints, 
                                                std::unordered_map<int, mpq_rational> fixed_values)
{
    // Get A, b, number of variables, number of constraints, and inequality type 
    Matrix<mpq_rational, Dynamic, Dynamic> A = constraints->getA(); 
    Matrix<mpq_rational, Dynamic, 1> b = constraints->getb();
    const int D = constraints->getD();
    const int N = constraints->getN();
    Polytopes::InequalityType type = constraints->getInequalityType();

    // Initialize new set of constraints
    int D_new = D;
    for (int i = 0; i < D; ++i)
    {
        if (fixed_values.find(i) != fixed_values.end())
            D_new--;
    }
    int N_new = 0;
    Matrix<mpq_rational, Dynamic, Dynamic> A_new(0, D_new);
    Matrix<mpq_rational, Dynamic, 1> b_new(0);

    // For each constraint ... 
    for (int i = 0; i < N; ++i)
    {
        Matrix<mpq_rational, Dynamic, 1> coefs_nonfixed(D_new);
        mpq_rational b_new_i = b(i);

        // For each variable in the constraint ...
        int curr = 0;
        for (int j = 0; j < D; ++j)
        {
            // Is the variable fixed?
            if (fixed_values.find(j) != fixed_values.end())
            {
                // If so, then transform the corresponding value of b
                b_new_i -= A(i, j) * fixed_values[j]; 
            }
            else
            {
                // Otherwise, keep track of the corresponding entry in A
                coefs_nonfixed(curr) = A(i, j);
                curr++;
            }
        }

        // Define the new constraint vector, given that there is at least one 
        // fixed variable ...
        if ((coefs_nonfixed.array() != 0).any())
        {
            N_new++;
            A_new.conservativeResize(N_new, D_new);
            b_new.conservativeResize(N_new);
            for (int j = 0; j < D_new; ++j)
                A_new(N_new - 1, j) = coefs_nonfixed(j);
            b_new(N_new - 1) = b_new_i;
        }
    }

    Polytopes::LinearConstraints* new_constraints = new Polytopes::LinearConstraints(type, A_new, b_new);
    new_constraints->removeRedundantConstraints(); 
    return new_constraints; 
}

void runConstrainedSampling(const std::string poly_filename, const int n, const int m,
                            const mpq_rational logbp, const mpq_rational logdp, 
                            const std::string prefix)
{
    std::unordered_map<int, mpq_rational> fixed_values;
    fixed_values[1] = 0;    // Add placeholder value
    fixed_values[2] = logbp; 
    fixed_values[3] = logdp; 

    // Read in the given polytope inequality file
    Polytopes::LinearConstraints* constraints = new Polytopes::LinearConstraints(
        Polytopes::InequalityType::GreaterThanOrEqualTo
    );
    constraints->parse(poly_filename);

    // Define the constrained polytope 
    Polytopes::LinearConstraints* new_constraints = constrainPolytope(constraints, fixed_values);

    // Translate the polytope so that the origin is a vertex, given that 
    // lower/upper bounds are given for each variable and each non-trivial 
    // constraint has corresponding right-hand value of zero
    Polytopes::InequalityType type = new_constraints->getInequalityType();
    Matrix<mpq_rational, Dynamic, Dynamic> A_reduced = new_constraints->getA(); 
    Matrix<mpq_rational, Dynamic, 1> b_reduced = new_constraints->getb();
    const int N_reduced = new_constraints->getN();
    const int D_reduced = new_constraints->getD();
    Matrix<mpq_rational, Dynamic, 1> var_min(D_reduced);
    for (int i = 0; i < N_reduced; ++i)
    {
        // Does the constraint specify a lower/upper bound? 
        if ((A_reduced.row(i).array() == 0).cast<int>().sum() == D_reduced - 1)
        {
            // Depending on the inequality type, determine whether the bound
            // is a lower bound or an upper bound
            int k;
            for (int j = 0; j < D_reduced; ++j)
            {
                if (A_reduced(i, j) != 0)
                {
                    k = j;
                    break;
                }
            }
            if ((type == Polytopes::InequalityType::GreaterThanOrEqualTo && A_reduced(i, k) > 0) || 
                (type == Polytopes::InequalityType::LessThanOrEqualTo && A_reduced(i, k) < 0))
            {
                var_min(k) = b_reduced(i) / A_reduced(i, k);
            }
        }
    }
    Matrix<mpq_rational, Dynamic, 1> b_translated(b_reduced);
    for (int i = 0; i < N_reduced; ++i)
        b_translated(i) += A_reduced.row(i).dot(-1 * var_min);

    // Enumerate the vertices of this translated 2-D polytope 
    Polytopes::PolyhedralDictionarySystem* dict = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::GreaterThanOrEqualTo, A_reduced, b_translated
    );
    dict->switchInequalityType();
    dict->removeRedundantConstraints(); 
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices();

    // Translate the vertices back to original coordinates 
    for (int i = 0; i < D_reduced; ++i)
        vertices.col(i) += var_min(i) * Matrix<mpq_rational, Dynamic, 1>::Ones(vertices.rows());

    // Compute the Delaunay triangulation of this reduced polytope 
    Delaunay_triangulation tri = Polytopes::triangulate(vertices); 

    // Sample model parameter combinations from the reduced polytope ...
    MatrixXd logrates_reduced;
    try
    {
        logrates_reduced = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, n, 0, rng); 
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // ... and add values for the excluded parameters
    const int D = constraints->getD();
    MatrixXd logrates = MatrixXd::Zero(n * m, D);
    int curr = 0;
    VectorXd range = VectorXd::LinSpaced(m, -1, 1);
    for (int i = 0; i < n; ++i)
    {
        MatrixXd logrates_i(m, D);
        for (int j = 0; j < m; ++j)
        {
            logrates_i(j, 0) = logrates_reduced(i, 0);
            logrates_i(j, 1) = logrates_reduced(i, 0) + range(j);
            logrates_i(j, 2) = static_cast<double>(logbp);
            logrates_i(j, 3) = static_cast<double>(logdp);
            logrates_i(j, 4) = logrates_reduced(i, 1); 
            logrates_i(j, 5) = logrates_reduced(i, 2);
        }
        logrates(Eigen::seqN(i * m, m), Eigen::all) = logrates_i;
    }
   
    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << prefix << "-logrates.tsv";  
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (samplefile.is_open())
    {
        for (int i = 0; i < logrates.rows(); i++)
        {
            for (int j = 0; j < logrates.cols() - 1; j++)
            {
                samplefile << logrates(i, j) << "\t";
            }
            samplefile << logrates(i, logrates.cols()-1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());

    // Compute asymptotic specificities and tradeoff constants 
    Matrix<PreciseType, Dynamic, Dynamic> asymp_tradeoff_rapid(n * m, length); 
    for (int i = 0; i < n * m; ++i)
    {
        VectorXd logrates_i(D - 2);
        logrates_i.head(2) = logrates.row(i).head(2);
        logrates_i.tail(2) = logrates.row(i).tail(2);
        asymp_tradeoff_rapid.row(i) = computeLimitsForSmallMismatchRatio(
            logrates_i, logrates(i, 2), logrates(i, 3)
        ).transpose();
    }

    // Write matrix of asymptotic rapidity tradeoff constants
    oss << prefix << "-asymp-smallmismatch-rapid.tsv";  
    std::ofstream rapidfile2(oss.str()); 
    rapidfile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (rapidfile2.is_open())
    {
        for (int i = 0; i < n * m; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                rapidfile2 << asymp_tradeoff_rapid(i, j) << "\t"; 
            }
            rapidfile2 << asymp_tradeoff_rapid(i, length-1) << std::endl; 
        }
    }
    rapidfile2.close();
    oss.clear();
    oss.str(std::string());

    delete constraints;
    delete new_constraints;
    delete dict;
}

int main(int argc, char** argv)
{
    const std::string poly_filename = argv[1];
    const std::string prefix = argv[2];
    const int n = std::stoi(argv[3]);
    std::stringstream ss; 
    ss << prefix << "-smallmismatch";
    std::string prefix2 = ss.str();
    runConstrainedSampling(poly_filename, n, 49, -5, 5, prefix2);

    return 0;
}
