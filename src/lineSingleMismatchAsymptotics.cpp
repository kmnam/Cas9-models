/**
 * Compute various asymptotic tradeoff constants for the line-graph Cas9 
 * model with respect to single-mismatch substrates. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/11/2023
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
 * Compute the cleavage probabilities, cleavage rates, dead unbinding rates,
 * and live unbinding rates of randomly parametrized line-graph Cas9 models
 * against single-mismatch substrates. 
 */
Matrix<PreciseType, Dynamic, 8> computeCleavageStats(const Ref<const VectorXd>& logrates)
{
    // Define arrays of DNA/RNA match and mismatch parameters 
    std::pair<PreciseType, PreciseType> match_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(0))),
        pow(ten, static_cast<PreciseType>(logrates(1))) 
    );
    std::pair<PreciseType, PreciseType> mismatch_rates = std::make_pair(
        pow(ten, static_cast<PreciseType>(logrates(2))),
        pow(ten, static_cast<PreciseType>(logrates(3)))
    );

    // Populate each rung with DNA/RNA match parameters
    LineGraph<PreciseType, PreciseType>* model = new LineGraph<PreciseType, PreciseType>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and
    // live unbinding rate
    PreciseType terminal_unbind_rate = pow(ten, static_cast<PreciseType>(logrates(4)));
    PreciseType terminal_cleave_rate = pow(ten, static_cast<PreciseType>(logrates(5)));
    Matrix<PreciseType, Dynamic, 8> stats = Matrix<PreciseType, Dynamic, 8>::Zero(length + 1, 8); 
    stats(0, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 2) = model->getLowerExitRate(terminal_unbind_rate);
    stats(0, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);  

    // Introduce single mismatches and re-compute the four output metrics 
    for (int j = 0; j < length; ++j)
    {
        for (int k = 0; k < length; ++k)
            model->setEdgeLabels(k, match_rates); 
        model->setEdgeLabels(j, mismatch_rates);
        stats(j+1, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate);
        stats(j+1, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
        stats(j+1, 2) = model->getLowerExitRate(terminal_unbind_rate);
        stats(j+1, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);
        stats(j+1, 4) = log10(stats(0, 0)) - log10(stats(j+1, 0));
        stats(j+1, 5) = log10(stats(0, 1)) - log10(stats(j+1, 1)); 
        stats(j+1, 6) = log10(stats(j+1, 2)) - log10(stats(0, 2));
        stats(j+1, 7) = log10(stats(j+1, 3)) - log10(stats(0, 3));
    }

    delete model;
    return stats;
}

/**
 * Compute asymptotic activities for randomly parametrized line-graph Cas9
 * models with large b / d.
 */
PreciseType computeAsymptoticActivityForLargeMatchRatio(
    const Ref<const VectorXd>& logrates, const double _logb, const double _logd)
{
    PreciseType logb = static_cast<PreciseType>(_logb);
    PreciseType terminal_unbind_lograte = static_cast<PreciseType>(logrates(2));
    return -logsumexp<PreciseType>(0, terminal_unbind_lograte - logb, ten);
}

/**
 * Compute asymptotic specificities for randomly parametrized line-graph 
 * Cas9 models with large b / d with respect to the PAM-adjacent single-
 * mismatch substrate. 
 */
PreciseType computeAsymptoticSpecificityProximalForLargeMatchRatio(
    const Ref<const VectorXd>& logrates, const double _logb, const double _logd)
{
    PreciseType logbp = static_cast<PreciseType>(logrates(0));
    PreciseType terminal_unbind_lograte = static_cast<PreciseType>(logrates(2));
    return logsumexp<PreciseType>(0, terminal_unbind_lograte - logbp, ten);
}

/**
 * Compute asymptotic specificities for randomly parametrized line-graph 
 * Cas9 models with small b' / d' with respect to the PAM-adjacent single-
 * mismatch substrate. 
 */
Matrix<PreciseType, Dynamic, 1> computeAsymptoticSpecificityForSmallMismatchRatio(
    const Ref<const VectorXd>& logrates, const double _logbp, const double _logdp)
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

    // Ratios of match/mismatch parameters
    const PreciseType logc = logb - logd; 
    const PreciseType logcp = logbp - logdp;

    // Compute the partial sums of the form log(1), log(1 + c), ..., log(1 + c + ... + c^N)
    Matrix<PreciseType, Dynamic, 1> logc_powers(length + 1);
    Matrix<PreciseType, Dynamic, 1> logc_partial_sums(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        logc_powers(i) = i * logc;
    for (int i = 0; i < length + 1; ++i)
        logc_partial_sums(i) = logsumexp(logc_powers.head(i + 1), ten);

    // Introduce single mismatches and compute asymptotic specificity with
    // respect to each single-mismatch substrate 
    Matrix<PreciseType, Dynamic, 1> stats(length);

    // Compute the asymptotic dissociativity tradeoff constants for all other 
    // single-mismatch substrates ...
    Matrix<PreciseType, Dynamic, 1> arr_gamma(3); 
    Matrix<PreciseType, Dynamic, 1> arr_gamma_m(3);
    arr_gamma << terminal_cleave_lograte + logc_powers(length), 
                 terminal_unbind_lograte, 
                 terminal_cleave_lograte + terminal_unbind_lograte - logd + logc_partial_sums(length - 1); 
    PreciseType gamma = logsumexp(arr_gamma, ten);
    PreciseType gamma_m = 0; 
    for (int m = 0; m < length - 1; ++m)
    {
        arr_gamma_m << 0,
                       terminal_cleave_lograte + logc_powers(length - 1 - m) - logdp,
                       terminal_cleave_lograte + logc_partial_sums(length - 2 - m) - logd;
        gamma_m = terminal_unbind_lograte + logsumexp(arr_gamma_m, ten);
        stats(m) = logc - logcp + gamma_m - gamma; 
    }
    gamma_m = (    // For m = length - 1
        terminal_unbind_lograte
        + logsumexp<PreciseType>(0, terminal_cleave_lograte + logc_powers(0) - logdp, ten)
    ); 
    stats(length - 1) = logc - logcp + gamma_m - gamma; 

    return stats;
}

/**
 * Compute the asymptotic specific rapidity and dead dissociativity tradeoff 
 * constants of randomly parametrized line-graph Cas9 models with large b / d
 * with respect to single-mismatch substrates.
 */
Matrix<PreciseType, Dynamic, 1> computeLimitsForLargeMatchRatio(const Ref<const VectorXd>& logrates,
                                                                const double _logb,
                                                                const double _logd)
{
    // Get DNA/RNA match parameters
    const PreciseType logb = static_cast<PreciseType>(_logb); 
    const PreciseType logd = static_cast<PreciseType>(_logd); 

    // Get DNA/RNA mismatch parameters
    const PreciseType logbp = static_cast<PreciseType>(logrates(0)); 
    const PreciseType logdp = static_cast<PreciseType>(logrates(1));

    // Get terminal rates
    const PreciseType terminal_unbind_lograte = static_cast<PreciseType>(logrates(2));
    const PreciseType terminal_cleave_lograte = static_cast<PreciseType>(logrates(3)); 

    // Ratios of match/mismatch parameters
    const PreciseType logc = logb - logd; 
    const PreciseType logcp = logbp - logdp;
    
    // Introduce single mismatches and compute asymptotic normalized statistics 
    // with respect to each single-mismatch substrate 
    Matrix<PreciseType, Dynamic, 1> stats(length);

    // For m = 0, compute the rapidity tradeoff constant: 
    //
    // log10(1 + ((unbind_rate + cleave_rate) / b'))
    //
    stats(0) = logsumexp<PreciseType>(
        0, logsumexp<PreciseType>(terminal_unbind_lograte, terminal_cleave_lograte, ten) - logbp, ten
    );
    
    // ... and the dead dissociativity tradeoff constant: 
    //
    // log10(c / c') + log10(1 + unbind_rate / b')
    //
    //stats(0, 1) = logc - logcp + logsumexp<PreciseType>(0, terminal_unbind_lograte - logbp, ten); 

    for (int m = 1; m < length - 1; ++m)
    {
        // For 0 < m < length - 1, compute the rapidity tradeoff constant:
        //
        // log10(1 + (cleave_rate / b'))
        //
        stats(m) = logsumexp<PreciseType>(0, terminal_cleave_lograte - logbp, ten);

        // ... and the dead dissociativity tradeoff constant: log10(c / c')
        //stats(m, 1) = logc - logcp;
    }

    // For m = length - 1, compute the rapidity tradeoff constant: 
    //
    // log10(1 + ((d' + cleave_rate) / b'))
    //
    stats(length - 1) = logsumexp<PreciseType>(
        0, logsumexp<PreciseType>(terminal_cleave_lograte, logdp, ten) - logbp, ten
    );

    // ... and the dead dissociativity tradeoff constant: 
    //
    // log10(c / (1 + c')) = log10(c) - log10(1 + c')
    //
    //stats(length - 1, 1) = logc - logsumexp<PreciseType>(0, logcp, ten);

    return stats;
}

/**
 * Compute the asymptotic specific rapidity and dead dissociativity tradeoff
 * constants of randomly parametrized line-graph Cas9 models with small b' / d'
 * with respect to single-mismatch substrates.
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

    // Compute the asymptotic dissociativity tradeoff constants for all other 
    // single-mismatch substrates ...
    /*
    Matrix<PreciseType, Dynamic, 1> arr_gamma(3); 
    Matrix<PreciseType, Dynamic, 1> arr_gamma_m(3);
    arr_gamma << terminal_cleave_lograte + logc_powers(length), 
                 terminal_unbind_lograte, 
                 terminal_cleave_lograte + terminal_unbind_lograte - logd + logc_partial_sums(length - 1); 
    PreciseType gamma = logsumexp(arr_gamma, ten) - logc_partial_sums(length);
    PreciseType gamma_m = 0; 
    for (int m = 0; m < length - 1; ++m)
    {
        arr_gamma_m << 0,
                       terminal_cleave_lograte + logc_powers(length - 1 - m) - logdp,
                       terminal_cleave_lograte + logc_partial_sums(length - 2 - m) - logd;
        gamma_m = terminal_unbind_lograte + logsumexp(arr_gamma_m, ten) - logc_partial_sums(m);
        stats(m, 1) = logc - logcp + gamma_m - gamma; 
    }
    gamma_m = (    // For m = length - 1
        terminal_unbind_lograte
        + logsumexp<PreciseType>(0, terminal_cleave_lograte + logc_powers(0) - logdp, ten)
        - logc_partial_sums(length - 1)
    ); 
    stats(length - 1, 1) = logc - logcp + gamma_m - gamma; 
    */

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

void runConstrainedSampling(const std::string poly_filename, const int n,
                            std::unordered_map<int, mpq_rational> fixed_values,
                            const std::string prefix)
{
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
    MatrixXd logrates(n, D);
    int curr = 0;
    for (int j = 0; j < D; ++j)
    {
        if (fixed_values.find(j) != fixed_values.end())
        {
            logrates.col(j) = (fixed_values[j] * Matrix<mpq_rational, Dynamic, 1>::Ones(n)).cast<double>();
        }
        else
        {
            logrates.col(j) = logrates_reduced.col(curr);
            curr++;
        }
    }
    
    // Compute cleavage probabilities, unbinding rates, cleavage rates and 
    // corresponding normalized quantities 
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> rapid(n, length);
    for (int i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 8> stats = computeCleavageStats(logrates.row(i));
        probs.row(i) = stats.col(0);
        specs.row(i) = stats.col(4).tail(length);
        rapid.row(i) = stats.col(5).tail(length);
    }
    std::cout << specs << std::endl; 
    std::cout << "--\n";
    std::cout << rapid << std::endl;
    std::cout << "--\n";

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

    // Write matrix of cleavage probabilities
    oss << prefix << "-exact-probs.tsv";  
    std::ofstream probsfile(oss.str());
    probsfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (probsfile.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                probsfile << probs(i, j) << "\t";
            }
            probsfile << probs(i, length) << std::endl; 
        }
    }
    probsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage specificities
    oss << prefix << "-exact-specs.tsv";  
    std::ofstream specfile(oss.str());
    specfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (specfile.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                specfile << specs(i, j) << "\t";
            }
            specfile << specs(i, length - 1) << std::endl; 
        }
    }
    specfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of specific rapidities
    oss << prefix << "-exact-rapid.tsv";  
    std::ofstream rapidfile(oss.str());
    rapidfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (rapidfile.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                rapidfile << rapid(i, j) << "\t"; 
            }
            rapidfile << rapid(i, length-1) << std::endl; 
        }
    }
    rapidfile.close();
    oss.clear();
    oss.str(std::string());

    // Compute asymptotic specificities and tradeoff constants 
    if (fixed_values.find(0) != fixed_values.end() && fixed_values.find(1) != fixed_values.end())
    {
        Matrix<PreciseType, Dynamic, 2> asymp_stats(n, 2);    // Activities in column 0, specificities in column 1
        for (int i = 0; i < n; ++i)
        { 
            VectorXd logrates_i = logrates.row(i).tail(D - 2);
            asymp_stats(i, 0) = computeAsymptoticActivityForLargeMatchRatio(
                logrates_i, logrates(i, 0), logrates(i, 1)
            );
            asymp_stats(i, 1) = computeAsymptoticSpecificityProximalForLargeMatchRatio(
                logrates_i, logrates(i, 0), logrates(i, 1)
            );
        }
        std::cout << asymp_stats << std::endl; 
        std::cout << "--\n";  

        // Write matrix of asymptotic activities and specificities
        oss << prefix << "-asymp-activities-specs.tsv";  
        std::ofstream specfile2(oss.str());
        specfile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        if (specfile2.is_open())
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < asymp_stats.cols() - 1; ++j)
                {
                    specfile2 << asymp_stats(i, j) << "\t";
                }
                specfile2 << asymp_stats(i, asymp_stats.cols()-1) << std::endl;
            }
        }
        specfile2.close();
    } 
    else    // fixed_values contains 2 and 3
    {
        Matrix<PreciseType, Dynamic, Dynamic> asymp_tradeoff_rapid(n, length); 
        for (int i = 0; i < n; ++i)
        {
            VectorXd logrates_i(D - 2);
            logrates_i.head(2) = logrates.row(i).head(2);
            logrates_i.tail(2) = logrates.row(i).tail(2);
            asymp_tradeoff_rapid.row(i) = computeLimitsForSmallMismatchRatio(
                logrates_i, logrates(i, 2), logrates(i, 3)
            ).transpose();
        }
        std::cout << asymp_tradeoff_rapid << std::endl;
        std::cout << "--\n";

        // Write matrix of asymptotic rapidity tradeoff constants
        oss << prefix << "-asymp-rapid.tsv";  
        std::ofstream rapidfile2(oss.str()); 
        rapidfile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
        if (rapidfile2.is_open())
        {
            for (int i = 0; i < n; ++i)
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
    }

    delete constraints;
    delete new_constraints;
    delete dict;
}

int main(int argc, char** argv)
{
    const std::string poly_filename = argv[1];
    const std::string prefix = argv[2];
    const int n = std::stoi(argv[3]);
    const mpq_rational p(argv[4]);
    std::stringstream ss; 
    std::unordered_map<int, mpq_rational> fixed_values;
    fixed_values[0] = p;
    fixed_values[1] = -p;
    ss << prefix << "-largematch-exp" << p;
    std::string prefix2 = ss.str();
    runConstrainedSampling(poly_filename, n, fixed_values, prefix2);
    fixed_values.clear();
    fixed_values[3] = p;
    fixed_values[2] = -p;
    ss.clear(); 
    ss.str(std::string());
    ss << prefix << "-smallmismatch-exp" << p;
    prefix2 = ss.str();
    runConstrainedSampling(poly_filename, n, fixed_values, prefix2);

    return 0;
}
