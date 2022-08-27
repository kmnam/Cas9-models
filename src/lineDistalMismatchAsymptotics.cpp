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

/**
 * Compute various asymptotic tradeoff constants for the line-graph Cas9 
 * model with respect to distal-mismatch substrates. 
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     8/26/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::pow;
using boost::multiprecision::log; 
using boost::multiprecision::log1p; 
using boost::multiprecision::log10;
const int INTERNAL_PRECISION = 100;
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;
const mpq_rational BIG_RATIONAL = static_cast<mpq_rational>(std::numeric_limits<int>::max());

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
 * against distal-mismatch substrates. 
 */
template <typename T>
Matrix<T, Dynamic, 8> computeCleavageStats(const Ref<const VectorXd>& logrates)
{
    // Define arrays of DNA/RNA match and mismatch parameters 
    std::pair<T, T> match_rates = std::make_pair(
        static_cast<T>(std::pow(10.0, logrates(0))),
        static_cast<T>(std::pow(10.0, logrates(1)))
    );
    std::pair<T, T> mismatch_rates = std::make_pair(
        static_cast<T>(std::pow(10.0, logrates(2))),
        static_cast<T>(std::pow(10.0, logrates(3)))
    );

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (int j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_rates); 
    
    // Compute cleavage probability, cleavage rate, dead unbinding rate, and
    // live unbinding rate
    T terminal_unbind_rate = 1;
    T terminal_cleave_rate = static_cast<T>(std::pow(10.0, logrates(4))); 
    Matrix<T, Dynamic, 8> stats = Matrix<T, Dynamic, 8>::Zero(length + 1, 8); 
    stats(0, 0) = model->getUpperExitProb(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 1) = model->getUpperExitRate(terminal_unbind_rate, terminal_cleave_rate); 
    stats(0, 2) = model->getLowerExitRate(terminal_unbind_rate);
    stats(0, 3) = model->getLowerExitRate(terminal_unbind_rate, terminal_cleave_rate);  

    // Introduce distal mismatches and re-compute the four output metrics 
    for (int j = 0; j < length; ++j)
    {
        // j here is the position of the first distal mismatch 
        for (int k = 0; k < j; ++k)
            model->setEdgeLabels(k, match_rates); 
        for (int k = j; k < length; ++k)
            model->setEdgeLabels(k, mismatch_rates);
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
 * Compute the asymptotic specific rapidity and dead dissociativity tradeoff 
 * constants of randomly parametrized line-graph Cas9 models with large b / d
 * with respect to distal-mismatch substrates.
 */
template <typename T>
Matrix<T, Dynamic, 2> computeLimitsForLargeMatchRatio(const Ref<const VectorXd>& logrates,
                                                      const double _logb,
                                                      const double _logd)
{
    const T ten = static_cast<T>(10);
    const T two = static_cast<T>(2); 
    const T log_two = log10(two);  

    // Get DNA/RNA match parameters
    const T logb = static_cast<T>(_logb); 
    const T logd = static_cast<T>(_logd); 

    // Get DNA/RNA mismatch parameters
    const T logbp = static_cast<T>(logrates(0)); 
    const T logdp = static_cast<T>(logrates(1));

    // Get terminal rates
    const T terminal_unbind_lograte = 0;
    const T terminal_cleave_lograte = static_cast<T>(logrates(2)); 

    // Ratios of match/mismatch parameters
    const T logc = logb - logd; 
    const T logcp = logbp - logdp;
    
    // Introduce distal mismatches and compute asymptotic normalized statistics 
    // with respect to each distal-mismatch substrate 
    Matrix<T, Dynamic, 2> stats(length, 2);

    // Compute the partial sums of the form log(1), log(1 + c), ..., log(1 + c + ... + c^N)
    Matrix<T, Dynamic, 1> logc_powers(length + 1);
    Matrix<T, Dynamic, 1> logc_partial_sums(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        logc_powers(i) = i * logc;
    for (int i = 0; i < length + 1; ++i)
        logc_partial_sums(i) = logsumexp(logc_powers.head(i + 1), ten);

    // Compute the partial sums of the form log(1), log(1 + c'), ..., log(1 + c + ... + (c')^N)
    Matrix<T, Dynamic, 1> logcp_powers(length + 1); 
    Matrix<T, Dynamic, 1> logcp_partial_sums(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        logcp_powers(i) = i * logcp; 
    for (int i = 0; i < length + 1; ++i)
        logcp_partial_sums(i) = logsumexp(logcp_powers.head(i + 1), ten); 

    // For m = 0, compute the rapidity tradeoff constant: 
    //
    // log10(beta / (c')^N)
    //
    Matrix<T, Dynamic, 1> arr_beta(length + 1);
    for (int i = 0; i < length + 1; ++i)
    {
        Matrix<T, Dynamic, 1> arr_term(4);
        if (i == 0)
        {
            arr_term.resize(2); 
            arr_term << logcp_powers(0),
                        terminal_cleave_lograte - logdp + logcp_partial_sums(length - 1);
        }
        else if (i == length)
        {
            arr_term.resize(2); 
            arr_term << logcp_powers(length),
                        terminal_unbind_lograte - logdp + logcp_partial_sums(length - 1);
        }
        else
        {
            arr_term << logcp_powers(i),
                        terminal_unbind_lograte - logdp + logcp_partial_sums(i - 1),
                        terminal_cleave_lograte - logdp + logcp_powers(i) + logcp_partial_sums(length - 1 - i),
                        terminal_unbind_lograte + terminal_cleave_lograte - 2 * logdp
                            + logcp_partial_sums(i) + logcp_partial_sums(length - 1 - i);
        }
        arr_beta(i) = logsumexp(arr_term, ten); 
    }
    T beta = logsumexp(arr_beta, ten); 
    stats(0, 0) = logsumexp<T>(beta, -length * logcp, ten); 
    
    // ... and the dead dissociativity tradeoff constant: 
    //
    // log10((c / c')^N) + log10(gamma' / cleave_rate)
    //
    Matrix<T, Dynamic, 1> arr_gamma_p(3); 
    arr_gamma_p << logcp_powers(length) + terminal_cleave_lograte,
                   terminal_unbind_lograte,
                   terminal_unbind_lograte + terminal_cleave_lograte - logdp + logcp_partial_sums(length - 1);
    T gamma_p = logsumexp(arr_gamma_p, ten) - logcp_partial_sums(length); 
    stats(0, 1) = logc_powers(length) - logcp_powers(length) + gamma_p - terminal_cleave_lograte; 
    
    for (int m = 1; m < length; ++m)
    {
        // For 0 < m < length - 1, compute the rapidity tradeoff constant:
        //
        // log10(1 + sum_{i=1}^{N-m}{(1 + (cleave_rate / d') * (N - m - i + 1)) / (c')^i}
        //
        Matrix<T, Dynamic, 1> arr_term(length - m);
        for (int i = 1; i <= length - m; ++i)
        {
            T logn = log10(static_cast<T>(length - m - i + 1)); 
            arr_term(i-1) = logsumexp<T>(0, logn + terminal_cleave_lograte - logdp, ten) - logcp_powers(i); 
        }
        stats(m, 0) = logsumexp<T>(0, logsumexp(arr_term, ten), ten); 

        // ... and the dead dissociativity tradeoff constant:
        //
        // log10(c^(N-m)) - log10(1 + c' + ... + (c')^(N-m))
        //
        stats(m, 1) = logc_powers(length - m) - logcp_partial_sums(length - m);
    }
    
    return stats;
}

/**
 * Compute the asymptotic specific rapidity and dead dissociativity tradeoff
 * constants of randomly parametrized line-graph Cas9 models with small b' / d'
 * with respect to distal-mismatch substrates.
 */
template <typename T>
Matrix<T, Dynamic, 2> computeLimitsForSmallMismatchRatio(const Ref<const VectorXd>& logrates,
                                                         const double _logbp,
                                                         const double _logdp)
{
    const T ten = static_cast<T>(10);
    const T two = static_cast<T>(2); 
    const T log_two = log10(two);

    // Get DNA/RNA match parameters
    const T logb = static_cast<T>(logrates(0)); 
    const T logd = static_cast<T>(logrates(1)); 

    // Get DNA/RNA mismatch parameters
    const T logbp = static_cast<T>(_logbp);
    const T logdp = static_cast<T>(_logdp);

    // Get terminal rates
    const T terminal_unbind_lograte = 0;
    const T terminal_cleave_lograte = static_cast<T>(logrates(2));
    const T terminal_lograte_sum = terminal_unbind_lograte + terminal_cleave_lograte;  

    // Ratios of match/mismatch parameters
    const T logc = logb - logd; 
    const T logcp = logbp - logdp;
    
    // Introduce distal mismatches and compute asymptotic normalized statistics 
    // with respect to each distal-mismatch substrate 
    Matrix<T, Dynamic, 2> stats(length, 2);

    // Compute the partial sums of the form log(1), log(1 + c), ..., log(1 + c + ... + c^N)
    Matrix<T, Dynamic, 1> logc_powers(length + 1);
    Matrix<T, Dynamic, 1> logc_partial_sums(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        logc_powers(i) = i * logc;
    for (int i = 0; i < length + 1; ++i)
        logc_partial_sums(i) = logsumexp(logc_powers.head(i + 1), ten);

    // Compute the powers log(1), log(c'), log((c')^2), ..., log((c')^N)
    Matrix<T, Dynamic, 1> logcp_powers(length + 1); 
    for (int i = 0; i < length + 1; ++i)
        logcp_powers(i) = i * logcp; 

    // Compute the asymptotic rapidity tradeoff constants for all distal-mismatch
    // substrates ...
    Matrix<T, Dynamic, 1> arr_alpha(length + 1);

    // Compute each term of alpha, which does not depend on the initial 
    // mismatch position m
    arr_alpha(0) = logsumexp<T>(0, logc_partial_sums(length - 1) - logd, ten);  
    for (int i = 1; i < length; ++i)
    {
        Matrix<T, Dynamic, 1> subarr(4); 
        subarr << logc_powers(i),
                  logc_partial_sums(i - 1) + terminal_unbind_lograte - logd,
                  logc_partial_sums(length - i - 1) + logc_powers(i) + terminal_cleave_lograte - logd,
                  logc_partial_sums(i - 1) + logc_partial_sums(length - i - 1) + terminal_lograte_sum - (2 * logd); 
        arr_alpha(i) = logsumexp(subarr, ten); 
    }
    arr_alpha(length) = logsumexp<T>(
        logc_powers(length),
        logc_partial_sums(length - 1) + terminal_unbind_lograte - logd, ten
    ); 
    T alpha = logsumexp(arr_alpha, ten);

    // Compute each term of beta_m, for each initial mismatch position m; and 
    // finally the rapidity tradeoff constant: 
    //
    // log10((c / c')^(N-m) (beta_m / (c^m alpha)))
    //
    for (int m = 0; m < length; ++m)
    {
        Matrix<T, Dynamic, 1> arr_beta_m(m + 1); 
        for (int i = 0; i <= m; ++i)
        {
            T term1 = logsumexp<T>(0, terminal_cleave_lograte - logdp, ten);
            T term2 = 0; 
            if (i > 0)
            {
                term2 = logsumexp<T>(
                    logc_powers(m),
                    terminal_unbind_lograte + logc_powers(m - i) - logd + logc_partial_sums(i - 1),
                    ten
                );  
            } 
            else
            {
                term2 = logc_powers(m); 
            }
            arr_beta_m(i) = logc_powers(i) + term1 + term2;
        }
        T beta_m = logsumexp<T>(
            log10(length - m) + logc_powers(m) + terminal_unbind_lograte - logdp
            + logsumexp<T>(0, terminal_cleave_lograte - logdp, ten),
            logsumexp(arr_beta_m, ten),
            ten
        ); 
        stats(m, 0) = logc_powers(length - m) - logcp_powers(length - m) + beta_m - logc_powers(m) - alpha; 
    }

    // Compute the dead dissociativity tradeoff constant for m = 0:  
    //
    // log10((c / c')^N * (unbind_rate + cleave_rate * unbind_rate / d') / gamma)
    //
    Matrix<T, Dynamic, 1> arr_gamma(3);
    arr_gamma << logc_powers(length), 0, logc_partial_sums(length - 1) - logd;
    T gamma = logsumexp(arr_gamma, ten);
    T term = logsumexp<T>(
        terminal_unbind_lograte,
        terminal_unbind_lograte + terminal_cleave_lograte - logdp,
        ten
    ); 
    stats(0, 1) = logc_powers(length) - logcp_powers(length) + term - gamma; 

    // ... then compute the dead dissociativity tradeoff constants for m > 0: 
    //
    // log10((c / c')^(N-m) * (unbind_rate + cleave_rate * unbind_rate / d') / (gamma * (1 + c + ... + c^m)))
    //
    for (int m = 1; m < length; ++m)
        stats(m, 1) = logc_powers(length - m) - logcp_powers(length - m) + term - gamma - logc_partial_sums(m); 

    return stats;
}

void runConstrainedSampling(const int idx_fixed_large, const int n, const double exp,
                            const std::string prefix)
{
    int idx_fixed_small, idx_var_large, idx_var_small; 
    if (idx_fixed_large == 0)    // b fixed large, d fixed small, b' variable small, d' variable large
    {
        idx_fixed_small = 1;
        idx_var_large = 3; 
        idx_var_small = 2;
    }
    else                         // d' fixed large, b' fixed small, b variable large, d variable small
    {
        idx_fixed_small = 2; 
        idx_var_large = 0; 
        idx_var_small = 1; 
    } 

    // Read in the given polytope inequality file
    Polytopes::LinearConstraints<mpq_rational>* constraints = new Polytopes::LinearConstraints<mpq_rational>(
        Polytopes::InequalityType::GreaterThanOrEqualTo
    );
    std::stringstream ss; 
    ss << "polytopes/line-" << static_cast<int>(exp) << "-unbindingunity-translated.poly";
    constraints->parse(ss.str()); 
    Matrix<mpq_rational, Dynamic, Dynamic> A = constraints->getA(); 
    Matrix<mpq_rational, Dynamic, 1> b = constraints->getb();
    const int D = constraints->getD();

    // Get the min/max values of all four (translated) parameters
    Matrix<mpq_rational, Dynamic, 1> max = -BIG_RATIONAL * Matrix<mpq_rational, Dynamic, 1>::Ones(D);
    Matrix<mpq_rational, Dynamic, 1> min = BIG_RATIONAL * Matrix<mpq_rational, Dynamic, 1>::Ones(D);
    for (int i = 0; i < A.rows(); ++i)
    {
        std::vector<int> nonzero; 
        for (int j = 0; j < D; ++j)
        {
            if (A(i, j) != 0)
                nonzero.push_back(j); 
        }
        if (nonzero.size() == 1 && A(i, nonzero[0]) < 0)         // A(i, j) * x(i) >= b(i), so -b(i) / -A(i, j) = max(x(i))
            max(nonzero[0]) = b(i) / A(i, nonzero[0]);    
        else if (nonzero.size() == 1 && A(i, nonzero[0]) > 0)    // A(i, j) * x(i) >= b(i), so b(i) / A(i, j) = min(x(i))
            min(nonzero[0]) = b(i) / A(i, nonzero[0]);
    }
    for (int i = 0; i < D; ++i)
    {
        if (max(i) == -BIG_RATIONAL)
            throw std::runtime_error("Maximum value for at least one parameter not specified");
        if (min(i) == BIG_RATIONAL) 
            throw std::runtime_error("Minimum value for at least one parameter not specified"); 
    }

    // Specify reduced constraints with a pair of parameters fixed  
    int nrows_reduced = 0; 
    Matrix<mpq_rational, Dynamic, Dynamic> A_reduced(nrows_reduced, D - 2); 
    Matrix<mpq_rational, Dynamic, 1> b_reduced(nrows_reduced);
    Matrix<mpq_rational, Dynamic, 1> max_reduced(max); 
    Matrix<mpq_rational, Dynamic, 1> min_reduced(min);
    for (int i = 0; i < A.rows(); ++i)
    {
        mpq_rational x = b(i) - A(i, idx_fixed_large) * max(idx_fixed_large) - A(i, idx_fixed_small) * min(idx_fixed_small); 

        // If the constraint merely concerns the smaller variable parameter
        // (d or b'), then change its stored max/min value accordingly 
        if (A(i, idx_var_small) != 0 && A(i, idx_var_large) == 0)
        {
            if (A(i, idx_var_small) > 0 && min_reduced(idx_var_small) > x / A(i, idx_var_small))
                min_reduced(idx_var_small) = x / A(i, idx_var_small);
            else if (A(i, idx_var_small) < 0 && max_reduced(idx_var_small) < x / A(i, idx_var_small))
                max_reduced(idx_var_small) = x / A(i, idx_var_small);
        }
        // If the constraint merely concerns the larger variable parameter
        // (b or d'), then change its stored max/min value accordingly
        else if (A(i, idx_var_small) == 0 && A(i, idx_var_large) != 0)
        {
            if (A(i, idx_var_large) > 0 && min_reduced(idx_var_large) > x / A(i, idx_var_large))
                min_reduced(idx_var_large) = x / A(i, idx_var_large);
            else if (A(i, idx_var_large) < 0 && max_reduced(idx_var_large) < x / A(i, idx_var_large)) 
                max_reduced(idx_var_large) = x / A(i, idx_var_large); 
        }
        // Otherwise, if the constraint concerns both variable parameters, then
        // add the constraint as is 
        else if (A(i, idx_var_small) != 0 && A(i, idx_var_large) != 0)
        { 
            nrows_reduced++;
            A_reduced.conservativeResize(nrows_reduced, D - 2); 
            b_reduced.conservativeResize(nrows_reduced);
            if (idx_var_small == 1)    // Either 1 (in which case idx_var_large is 0) ...
            { 
                A_reduced(nrows_reduced - 1, 1) = A(i, idx_var_small);
                A_reduced(nrows_reduced - 1, 0) = A(i, idx_var_large);
            }                          // ... or 2 (in which case idx_var_large is 3) 
            else
            { 
                A_reduced(nrows_reduced - 1, 0) = A(i, idx_var_small);
                A_reduced(nrows_reduced - 1, 1) = A(i, idx_var_large);
            }
            b_reduced(nrows_reduced - 1) = x; 
        }
    }
     
    // Add the max/min values for the variable parameters as separate constraints 
    const int Dp = 2 * (D - 2); 
    nrows_reduced += Dp;
    A_reduced.conservativeResize(nrows_reduced, D - 2);
    b_reduced.conservativeResize(nrows_reduced);
    // The first two rows provide minimum bounds for the two variable parameters
    A_reduced.block(nrows_reduced - Dp, 0, 2, 2) = Matrix<mpq_rational, Dynamic, Dynamic>::Identity(2, 2);
    if (idx_var_small == 1)    // Either 1 i.e. d (in which case idx_var_large is 0 i.e. b) ...
    {
        b_reduced(nrows_reduced - Dp) = min_reduced(idx_var_large); 
        b_reduced(nrows_reduced - Dp + 1) = min_reduced(idx_var_small);
    }
    else                       // ... or 2 i.e. b' (in which case idx_var_large is 3 i.e. d') 
    {
        b_reduced(nrows_reduced - Dp) = min_reduced(idx_var_small); 
        b_reduced(nrows_reduced - Dp + 1) = min_reduced(idx_var_large); 
    }
    // The second two rows provide maximum bounds for the two variable parameters
    A_reduced.block(nrows_reduced - Dp + 2, 0, 2, 2) = -Matrix<mpq_rational, Dynamic, Dynamic>::Identity(2, 2);
    if (idx_var_small == 1)    // Either 1 i.e. d (in which case idx_var_large is 0 i.e. b) ...
    {
        b_reduced(nrows_reduced - Dp + 2) = -max_reduced(idx_var_large);
        b_reduced(nrows_reduced - Dp + 3) = -max_reduced(idx_var_small);  
    }
    else                       // ... or 2 i.e. b' (in which case idx_var_large is 3 i.e. d')
    {
        b_reduced(nrows_reduced - Dp + 2) = -max_reduced(idx_var_small);
        b_reduced(nrows_reduced - Dp + 3) = -max_reduced(idx_var_large); 
    }
    // Next, provide minimum bounds for the remaining parameters
    int c = nrows_reduced - Dp + 4;
    for (int i = 0; i < D - 4; ++i)
    {
        A_reduced(c + i, 2 + i) = 1; 
        b_reduced(c + i) = min_reduced(2 + i); 
    }
    // Finally, provide maximum bounds for the remaining parameters
    c += D - 4; 
    for (int i = 0; i < D - 4; ++i)
    {  
        A_reduced(c + i, 2 + i) = -1;
        b_reduced(c + i) = -max_reduced(2 + i);
    }

    // Enumerate the vertices of this reduced 2-D polytope 
    Polytopes::PolyhedralDictionarySystem* dict = new Polytopes::PolyhedralDictionarySystem(
        Polytopes::InequalityType::GreaterThanOrEqualTo, A_reduced, b_reduced
    );
    dict->switchInequalityType();
    dict->removeRedundantConstraints(); 
    Matrix<mpq_rational, Dynamic, Dynamic> vertices = dict->enumVertices();

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

    // ... add values for the excluded parameters ...
    MatrixXd logrates(n, D);
    if (idx_var_small == 1)    // Either 1 i.e. d (in which case idx_var_large is 0 i.e. b) ...
    {
        logrates.col(idx_var_large) = logrates_reduced.col(0); 
        logrates.col(idx_var_small) = logrates_reduced.col(1);
    }
    else                       // ... or 2 i.e. b' (in which case idx_var_large is 3 i.e. d') 
    {
        logrates.col(idx_var_small) = logrates_reduced.col(0);
        logrates.col(idx_var_large) = logrates_reduced.col(1); 
    } 
    logrates.col(idx_fixed_large) = static_cast<double>(max(idx_fixed_large)) * VectorXd::Ones(n);
    logrates.col(idx_fixed_small) = static_cast<double>(min(idx_fixed_small)) * VectorXd::Ones(n);
    logrates(Eigen::all, Eigen::seqN(4, D - 4)) = logrates_reduced(Eigen::all, Eigen::seqN(2, D - 4));  

    // ... and translate the sampled values appropriately
    logrates -= exp * MatrixXd::Ones(n, D);

    // Compute cleavage probabilities, unbinding rates, and cleavage rates
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> unbind_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> cleave_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> rapid(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> dead_dissoc(n, length); 
    Matrix<PreciseType, Dynamic, Dynamic> asymp_tradeoff_rapid(n, length); 
    Matrix<PreciseType, Dynamic, Dynamic> asymp_tradeoff_deaddissoc(n, length);  
    for (int i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 8> stats = computeCleavageStats<PreciseType>(logrates.row(i));
        probs.row(i) = stats.col(0);
        unbind_rates.row(i) = stats.col(1);
        cleave_rates.row(i) = stats.col(2);
        specs.row(i) = stats.col(4).tail(length);
        rapid.row(i) = stats.col(5).tail(length);
        dead_dissoc.row(i) = stats.col(6).tail(length);
        Matrix<PreciseType, Dynamic, 2> tradeoffs;
        VectorXd logrates_i(D - 2); 
        if (idx_fixed_large == 0) 
        {
            logrates_i = logrates.row(i).tail(D - 2); 
            tradeoffs = computeLimitsForLargeMatchRatio<PreciseType>(
                logrates_i, logrates(i, 0), logrates(i, 1)
            );
        }
        else
        {
            logrates_i(0) = logrates(i, 0);
            logrates_i(1) = logrates(i, 1);
            for (int j = 0; j < D - 4; ++j)
                logrates_i(2 + j) = logrates(i, 4 + j);
            tradeoffs = computeLimitsForSmallMismatchRatio<PreciseType>(
                logrates_i, logrates(i, 2), logrates(i, 3)
            );
        }
        asymp_tradeoff_rapid.row(i) = tradeoffs.col(0);
        asymp_tradeoff_deaddissoc.row(i) = tradeoffs.col(1);
    }

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-logrates.tsv";
    else 
        oss << prefix << "-smallmismatch-logrates.tsv";  
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
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-probs.tsv";
    else 
        oss << prefix << "-smallmismatch-probs.tsv";  
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
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-specs.tsv";
    else 
        oss << prefix << "-smallmismatch-specs.tsv";  
    std::ofstream specsfile(oss.str());
    specsfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (specsfile.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                specsfile << specs(i, j) << "\t";
            }
            specsfile << specs(i, length - 1) << std::endl; 
        }
    }
    specsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of (unnormalized) unconditional unbinding rates
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-unbind-rates.tsv";
    else 
        oss << prefix << "-smallmismatch-unbind-rates.tsv";  
    std::ofstream unbindfile(oss.str());
    unbindfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (unbindfile.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                unbindfile << unbind_rates(i, j) << "\t";
            }
            unbindfile << unbind_rates(i, length) << std::endl; 
        }
    }
    unbindfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of (unnormalized) conditional cleavage rates
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-cleave-rates.tsv";
    else 
        oss << prefix << "-smallmismatch-cleave-rates.tsv";  
    std::ofstream cleavefile(oss.str());
    cleavefile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (cleavefile.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length; ++j)
            {
                cleavefile << cleave_rates(i, j) << "\t";
            }
            cleavefile << cleave_rates(i, length) << std::endl; 
        }
    }
    cleavefile.close();
    oss.clear();
    oss.str(std::string());
  
    // Write matrix of normalized unconditional unbinding rates
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-deaddissoc.tsv";
    else 
        oss << prefix << "-smallmismatch-deaddissoc.tsv";  
    std::ofstream unbindfile2(oss.str());
    unbindfile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (unbindfile2.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                unbindfile2 << dead_dissoc(i, j) << "\t";  
            }
            unbindfile2 << dead_dissoc(i, length-1) << std::endl; 
        }
    }
    unbindfile2.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of normalized conditional cleavage rates
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-rapid.tsv";
    else 
        oss << prefix << "-smallmismatch-rapid.tsv";  
    std::ofstream cleavefile2(oss.str());
    cleavefile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (cleavefile2.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                cleavefile2 << rapid(i, j) << "\t"; 
            }
            cleavefile2 << rapid(i, length-1) << std::endl; 
        }
    }
    cleavefile2.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of asymptotically determined normalized conditional
    // unbinding rate tradeoff constants
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-asymp-deaddissoc.tsv";
    else 
        oss << prefix << "-smallmismatch-asymp-deaddissoc.tsv";  
    std::ofstream unbindfile3(oss.str()); 
    unbindfile3 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (unbindfile3.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                unbindfile3 << asymp_tradeoff_deaddissoc(i, j) << "\t"; 
            }
            unbindfile3 << asymp_tradeoff_deaddissoc(i, length-1) << std::endl; 
        }
    }
    unbindfile3.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of asymptotically determined normalized conditional
    // cleavage rate tradeoff constants
    if (idx_fixed_large == 0)
        oss << prefix << "-largematch-asymp-rapid.tsv";
    else 
        oss << prefix << "-smallmismatch-asymp-rapid.tsv";  
    std::ofstream cleavefile3(oss.str()); 
    cleavefile3 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (cleavefile3.is_open())
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < length - 1; ++j)
            {
                cleavefile3 << asymp_tradeoff_rapid(i, j) << "\t"; 
            }
            cleavefile3 << asymp_tradeoff_rapid(i, length-1) << std::endl; 
        }
    }
    cleavefile3.close(); 

    delete constraints;
    delete dict;
}

int main(int argc, char** argv)
{
    int n = std::stoi(argv[2]);
    double exp = std::stod(argv[3]);
    runConstrainedSampling(0, n, exp, argv[1]); 
    runConstrainedSampling(3, n, exp, argv[1]); 

    return 0;
}
