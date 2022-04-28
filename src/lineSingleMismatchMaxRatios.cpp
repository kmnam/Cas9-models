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

/*
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     4/28/2022
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;
using boost::multiprecision::mpq_rational;
using boost::multiprecision::pow;
using boost::multiprecision::log; 
using boost::multiprecision::log1p; 
using boost::multiprecision::log10;
const unsigned INTERNAL_PRECISION = 1000;
typedef number<mpfr_float_backend<INTERNAL_PRECISION> > PreciseType;
const mpq_rational BIG_RATIONAL = static_cast<mpq_rational>(std::numeric_limits<int>::max());

const unsigned length = 20;

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
    for (unsigned i = 0; i < logx.size(); ++i)
        x(i) = pow(base, logx(i) - maxlogx); 
    
    return maxlogx + log(x.sum()) / log(base);  
}

template <typename T>
Matrix<T, Dynamic, 6> computeStats(const Ref<const VectorXd>& params)
{
    /*
     * Compute the cleavage probabilities, unbinding rates, and cleavage rates
     * of randomly parametrized Cas9 enzymes with respect to single-mismatch
     * substrates. 
     */
    // Array of DNA/RNA match parameters
    std::pair<T, T> match_params = std::make_pair(
        static_cast<T>(std::pow(10.0, params(0))),
        static_cast<T>(std::pow(10.0, params(1)))
    );

    // Array of DNA/RNA mismatch parameters
    std::pair<T, T> mismatch_params = std::make_pair(
        static_cast<T>(std::pow(10.0, params(2))),
        static_cast<T>(std::pow(10.0, params(3)))
    );

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T, T>* model = new LineGraph<T, T>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setEdgeLabels(j, match_params);
    
    // Compute cleavage probability, unbinding rate, and cleavage rate 
    Matrix<T, Dynamic, 6> stats = Matrix<T, Dynamic, 6>::Zero(length + 1, 6); 
    stats(0, 0) = model->getUpperExitProb(1, 1); 
    stats(0, 1) = model->getLowerExitRate(1); 
    stats(0, 2) = model->getUpperExitRate(1, 1); 

    // Introduce single mismatches and re-compute cleavage probability
    // and mean first-passage time
    for (unsigned j = 0; j < length; ++j)
    {
        for (unsigned k = 0; k < length; ++k)
            model->setEdgeLabels(k, match_params);
        model->setEdgeLabels(j, mismatch_params);
        stats(j+1, 0) = model->getUpperExitProb(1, 1);
        stats(j+1, 1) = model->getLowerExitRate(1);
        stats(j+1, 2) = model->getUpperExitRate(1, 1);
        stats(j+1, 3) = log10(stats(0, 0)) - log10(stats(j+1, 0));
        stats(j+1, 4) = log10(stats(0, 1)) - log10(stats(j+1, 1)); 
        stats(j+1, 5) = log10(stats(0, 2)) - log10(stats(j+1, 2));
    }

    delete model;
    return stats;
}

template <typename T>
Matrix<T, Dynamic, 2> computeLimitsForLarge01(const Ref<const VectorXd>& params,
                                              const double _logb,
                                              const double _logd)
{
    /*
     * Compute the asymptotic normalized statistics of randomly parametrized 
     * Cas9 enzymes with large b / d with respect to single-mismatch substrates.
     */
    const T ten = static_cast<T>(10);
    const T ln_ten = log(ten);  

    // Array of DNA/RNA match parameters
    const T logb = static_cast<T>(_logb); 
    const T logd = static_cast<T>(_logd); 

    // Array of DNA/RNA mismatch parameters
    const T logbp = static_cast<T>(params(0)); 
    const T logdp = static_cast<T>(params(1)); 

    // Ratios of match/mismatch parameters
    const T logc = logb - logd; 
    const T logcp = logbp - logdp;
    
    // Introduce single mismatches and compute asymptotic normalized statistics 
    // with respect to each single-mismatch substrate 
    Matrix<T, Dynamic, 2> stats(length, 2);

    // For m = 0, compute log10(c / c') + log10(1 + (2 / b')) and
    // log10(1 + (2 / b'))
    stats(0, 1) = log1p(2 * pow(ten, -logbp)) / ln_ten; 
    stats(0, 0) = logc - logcp + stats(0, 1);

    for (unsigned m = 1; m < length - 1; ++m)
    {
        // For 0 < m < length - 1, compute log10(c / c') ...
        stats(m, 0) = logc - logcp;

        // ... and log10(1 + (1 / b'))
        stats(m, 1) = log1p(pow(ten, -logbp)) / ln_ten; 
    }

    // For m = length - 1, compute log10(c / (1 + c')) = log10(c) - log10(1 + c')
    // and log10(1 + ((d' + 1) / b'))
    stats(length - 1, 0) = logc - (log1p(pow(ten, logcp)) / ln_ten);
    stats(length - 1, 1) = log1p(pow(ten, (log1p(pow(ten, logdp)) / ln_ten) - logbp)) / ln_ten; 

    return stats;
}

template <typename T>
Matrix<T, Dynamic, 2> computeLimitsForSmall23(const Ref<const VectorXd>& params,
                                              const double _logbp,
                                              const double _logdp)
{
    /*
     * Compute the asymptotic normalized statistics of randomly parametrized 
     * Cas9 enzymes with small b' / d' with respect to single-mismatch substrates.
     */
    const T ten = static_cast<T>(10);
    const T ln_ten = log(ten);
    const T two = static_cast<T>(2); 
    const T log_two = log10(two);
    const T three = static_cast<T>(3);
    const T log_three = log10(three); 

    // Array of DNA/RNA match parameters
    const T logb = static_cast<T>(params(0)); 
    const T logd = static_cast<T>(params(1)); 

    // Array of DNA/RNA mismatch parameters
    const T logbp = static_cast<T>(_logbp);
    const T logdp = static_cast<T>(_logdp); 

    // Ratios of match/mismatch parameters
    const T logc = logb - logd; 
    const T logcp = logbp - logdp;
    
    // Introduce single mismatches and compute asymptotic normalized statistics 
    // with respect to each single-mismatch substrate 
    Matrix<T, Dynamic, 2> stats(length, 2);

    // Compute log10((1 + c + ... + c^N) / (1 + c + ... + c^(N-1)))
    Matrix<T, Dynamic, 1> logcpow(length + 1); 
    for (unsigned i = 0; i < length + 1; ++i)
        logcpow(i) = i * logc;
    T lognumer = logsumexp(logcpow, ten);
    T logdenom = logsumexp(logcpow.head(length), ten);

    // Compute the asymptotic associativity tradeoff constant for the
    // most-distal-mismatch substrate 
    Matrix<T, Dynamic, 1> arr1(3);
    arr1 << length * logc, 0, logdenom - logd;
    stats(length - 1, 0) = logc - logcp + lognumer - logdenom;
    stats(length - 1, 0) += logsumexp<T>(0, -logdp, ten); 
    stats(length - 1, 0) -= logsumexp(arr1, ten);

    // Compute the asymptotic associativity tradeoff constants for all other 
    // single-mismatch substrates ...
    Matrix<T, Dynamic, 1> arr2(3);
    for (unsigned m = 0; m < length - 1; ++m)
    {
        arr2 << 0,
                (length - 1 - m) * logc - logdp,
                logsumexp(logcpow.head(length - 1 - m), ten);
        stats(m, 0) = 2 * (logc - logcp) + lognumer;
        stats(m, 0) -= logsumexp(logcpow.head(m + 1), ten);
        stats(m, 0) += 2 * logsumexp(arr2, ten); 
        stats(m, 0) -= 2 * logsumexp(arr1, ten);  
    } 

    // Compute the asymptotic rapidity tradeoff constants for all single-mismatch
    // substrates ...
    Matrix<T, Dynamic, 1> arrFm(length + 1);
    Matrix<T, Dynamic, 1> arrF(length + 1);

    // Compute each term of F, which does not depend on the mismatch position m 
    for (unsigned i = 0; i <= length - 2; ++i)
    {
        Matrix<T, Dynamic, 1> subarr(4); 
        subarr << 0,
                  -logd,
                  log10(static_cast<T>(length - i)) + log_two - logd,
                  log10(static_cast<T>(length)) + log10(static_cast<T>(i + 1)) - (2 * logd);
        arrF(i) = i * logc + logsumexp(subarr, ten); 
    }
    arrF(length - 1) = logsumexp<T>(logsumexp<T>(0, log_three - logd, ten), (length - 1) * logc, ten);
    arrF(length) = length * logc;
    T F = logsumexp(arrF, ten);

    // Compute each term of F_m, for each mismatch position m 
    for (unsigned m = 0; m < length; ++m)
    {
        for (unsigned i = 0; i <= m; ++i)
        {
            T term1 = 0;
            T term2 = 0; 
            if (i == 0)
            {
                term1 = (i + 1) * logc;
            }
            else 
            {
                term1 = logsumexp<T>(
                    (i + 1) * logc,
                    logc - logd + logsumexp(logcpow(Eigen::seq(0, i - 1)), ten),
                    ten
                ); 
            }
            if (m == length - 1)
            {
                Matrix<T, Dynamic, 1> subarr(2); 
                subarr << 0,
                          (length - m - 1) * logc - logdp;
                term2 = logsumexp(subarr, ten);
            } 
            else
            {
                Matrix<T, Dynamic, 1> subarr(3);
                subarr << 0,
                          (length - m - 1) * logc - logdp,
                          logsumexp(logcpow(Eigen::seq(0, length - 2 - m)), ten) - logd;
                term2 = logsumexp(subarr, ten); 
            }
            arrFm(i) = term1 + term2;
        }
        for (unsigned i = m + 1; i <= length; ++i)
        {
            T term1 = 0;
            T term2 = 0;
            if (i == length)
            {
                term1 = 0; 
            }
            else
            {
                term1 = logsumexp<T>(
                    0,
                    logsumexp(logcpow(Eigen::seq(0, length - 1 - i)), ten) - logd,
                    ten
                );
            }
            if (i == m + 1)
            {
                term2 = -logdp; 
            }
            else 
            {
                term2 = logsumexp<T>(
                    (i - 1 - m) * logc - logdp, 
                    logsumexp(logcpow(Eigen::seq(0, i - 2 - m)), ten) - logd,
                    ten
                );
            }
            arrFm(i) = logc + term1 + term2; 
        }

        stats(m, 1) = logsumexp(arrFm, ten) - F - logcp;  
    }

    return stats;
}

void runConstrainedSampling(const int idx_fixed_large, const int n,
                            const std::string prefix)
{
    int idx_fixed_small, idx_var_large, idx_var_small; 
    if (idx_fixed_large == 0)
    {
        idx_fixed_small = 1;
        idx_var_large = 3; 
        idx_var_small = 2;
    }
    else     // idx_fixed_large == 3
    {
        idx_fixed_small = 2; 
        idx_var_large = 0; 
        idx_var_small = 1; 
    } 

    // Read in the given polytope inequality file
    Polytopes::LinearConstraints<mpq_rational>* constraints = new Polytopes::LinearConstraints<mpq_rational>(
        Polytopes::InequalityType::GreaterThanOrEqualTo
    );
    constraints->parse("/Users/kmnam/Dropbox/gene-regulation/projects/Cas9-models/polytopes/line-5-translated.poly"); 
    Matrix<mpq_rational, Dynamic, Dynamic> A = constraints->getA(); 
    Matrix<mpq_rational, Dynamic, 1> b = constraints->getb();

    // Get the min/max values of all four (translated) parameters
    Matrix<mpq_rational, 4, 1> max, min;
    max << -BIG_RATIONAL, -BIG_RATIONAL, -BIG_RATIONAL, -BIG_RATIONAL; 
    min <<  BIG_RATIONAL,  BIG_RATIONAL,  BIG_RATIONAL,  BIG_RATIONAL; 
    for (unsigned i = 0; i < A.rows(); ++i)
    {
        std::vector<int> nonzero; 
        for (unsigned j = 0; j < 4; ++j)
        {
            if (A(i, j) != 0)
                nonzero.push_back(j); 
        }
        if (nonzero.size() == 1 && A(i, nonzero[0]) < 0)         // A(i, j) * x(i) >= b(i), so -b(i) / -A(i, j) = max(x(i))
            max(nonzero[0]) = b(i) / A(i, nonzero[0]);    
        else if (nonzero.size() == 1 && A(i, nonzero[0]) > 0)    // A(i, j) * x(i) >= b(i), so b(i) / A(i, j) = min(x(i))
            min(nonzero[0]) = b(i) / A(i, nonzero[0]);
    }
    for (unsigned i = 0; i < 4; ++i)
    {
        if (max(i) == -BIG_RATIONAL)
            throw std::runtime_error("Maximum value for at least one parameter not specified");
        if (min(i) == BIG_RATIONAL) 
            throw std::runtime_error("Minimum value for at least one parameter not specified"); 
    }

    // Specify reduced constraints with a pair of parameters fixed  
    int nrows_reduced = 0; 
    Matrix<mpq_rational, Dynamic, Dynamic> A_reduced(nrows_reduced, 2); 
    Matrix<mpq_rational, Dynamic, 1> b_reduced(nrows_reduced);
    Matrix<mpq_rational, 4, 1> max_reduced(max); 
    Matrix<mpq_rational, 4, 1> min_reduced(min);
    for (unsigned i = 0; i < A.rows(); ++i)
    {
        mpq_rational x = b(i) - A(i, idx_fixed_large) * max(idx_fixed_large) - A(i, idx_fixed_small) * min(idx_fixed_small); 

        // If the constraint merely concerns the smaller parameter (d or b'), then
        // change its stored max/min value accordingly 
        if (A(i, idx_var_small) != 0 && A(i, idx_var_large) == 0)
        {
            if (A(i, idx_var_small) > 0 && min_reduced(idx_var_small) > x / A(i, idx_var_small))
                min_reduced(idx_var_small) = x / A(i, idx_var_small);
            else if (A(i, idx_var_small) < 0 && max_reduced(idx_var_small) < x / A(i, idx_var_small))
                max_reduced(idx_var_small) = x / A(i, idx_var_small);
        }
        // If the constraint merely concerns the larger parameter (b or d'), then 
        // change its stored max/min value accordingly
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
            A_reduced.conservativeResize(nrows_reduced, 2); 
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
    nrows_reduced += 4; 
    A_reduced.conservativeResize(nrows_reduced, 2); 
    b_reduced.conservativeResize(nrows_reduced);
    A_reduced.block(nrows_reduced - 4, 0, 2, 2) = Matrix<mpq_rational, Dynamic, Dynamic>::Identity(2, 2);
    if (idx_var_small == 1)    // Either 1 (in which case idx_var_large is 0) ...
    {
        b_reduced(nrows_reduced - 4) = min_reduced(idx_var_large); 
        b_reduced(nrows_reduced - 3) = min_reduced(idx_var_small);
    }
    else                       // ... or 2 (in which case idx_var_large is 3) 
    {
        b_reduced(nrows_reduced - 4) = min_reduced(idx_var_small); 
        b_reduced(nrows_reduced - 3) = min_reduced(idx_var_large); 
    } 
    A_reduced.block(nrows_reduced - 2, 0, 2, 2) = -Matrix<mpq_rational, Dynamic, Dynamic>::Identity(2, 2);
    if (idx_var_small == 1)    // Either 1 (in which case idx_var_large is 0) ...
    {
        b_reduced(nrows_reduced - 2) = -max_reduced(idx_var_large);
        b_reduced(nrows_reduced - 1) = -max_reduced(idx_var_small);  
    }
    else                       // ... or 2 (in which case idx_var_large is 3)
    { 
        b_reduced(nrows_reduced - 2) = -max_reduced(idx_var_small);
        b_reduced(nrows_reduced - 1) = -max_reduced(idx_var_large); 
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
    MatrixXd params_reduced;
    try
    {
        params_reduced = Polytopes::sampleFromConvexPolytope<INTERNAL_PRECISION>(tri, n, 0, rng); 
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // ... add values for the excluded parameters ...
    MatrixXd params(n, 4);
    if (idx_var_small == 1)    // Either 1 (in which case idx_var_large is 0) ...
    {
        params.col(idx_var_large) = params_reduced.col(0); 
        params.col(idx_var_small) = params_reduced.col(1);
    }
    else                       // ... or 2 (in which case idx_var_large is 3) 
    {
        params.col(idx_var_small) = params_reduced.col(0);
        params.col(idx_var_large) = params_reduced.col(1); 
    } 
    params.col(idx_fixed_large) = static_cast<double>(max(idx_fixed_large)) * VectorXd::Ones(n);
    params.col(idx_fixed_small) = static_cast<double>(min(idx_fixed_small)) * VectorXd::Ones(n);  

    // ... and translate the sampled values appropriately
    params -= 5 * MatrixXd::Ones(n, 4);

    // Compute cleavage probabilities, unbinding rates, and cleavage rates
    Matrix<PreciseType, Dynamic, Dynamic> probs(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> unbind_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> cleave_rates(n, length + 1);
    Matrix<PreciseType, Dynamic, Dynamic> specs(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> norm_unbind(n, length); 
    Matrix<PreciseType, Dynamic, Dynamic> norm_cleave(n, length);
    Matrix<PreciseType, Dynamic, Dynamic> lim_norm_unbind(n, length); 
    Matrix<PreciseType, Dynamic, Dynamic> lim_norm_cleave(n, length);  
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<PreciseType, Dynamic, 6> stats = computeStats<PreciseType>(params.row(i));
        probs.row(i) = stats.col(0);
        unbind_rates.row(i) = stats.col(1);
        cleave_rates.row(i) = stats.col(2);
        specs.row(i) = stats.col(3).tail(length); 
        norm_unbind.row(i) = stats.col(4).tail(length); 
        norm_cleave.row(i) = stats.col(5).tail(length);
        Matrix<PreciseType, Dynamic, 2> lims;  
        if (idx_fixed_large == 0) 
        {
            lims = computeLimitsForLarge01<PreciseType>(
                params.row(i).tail(2), params(i, 0), params(i, 1)
            );
        }
        else
        {
            lims = computeLimitsForSmall23<PreciseType>(
                params.row(i).head(2), params(i, 2), params(i, 3)
            );
        }
        lim_norm_unbind.row(i) = lims.col(0); 
        lim_norm_cleave.row(i) = lims.col(1); 
    }

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-params.tsv";
    else 
        oss << prefix << "-small23-params.tsv";  
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (samplefile.is_open())
    {
        for (unsigned i = 0; i < params.rows(); i++)
        {
            for (unsigned j = 0; j < params.cols() - 1; j++)
            {
                samplefile << params(i,j) << "\t";
            }
            samplefile << params(i, params.cols()-1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage probabilities
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-probs.tsv";
    else 
        oss << prefix << "-small23-probs.tsv";  
    std::ofstream probsfile(oss.str());
    probsfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (probsfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                probsfile << probs(i,j) << "\t";
            }
            probsfile << probs(i, length) << std::endl; 
        }
    }
    probsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage specificities
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-specs.tsv";
    else 
        oss << prefix << "-small23-specs.tsv";  
    std::ofstream specsfile(oss.str());
    specsfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (specsfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                specsfile << specs(i,j) << "\t";
            }
            specsfile << specs(i, length - 1) << std::endl; 
        }
    }
    specsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of (unnormalized) unconditional unbinding rates
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-unbind-rates.tsv";
    else 
        oss << prefix << "-small23-unbind-rates.tsv";  
    std::ofstream unbindfile(oss.str());
    unbindfile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (unbindfile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                unbindfile << unbind_rates(i,j) << "\t";
            }
            unbindfile << unbind_rates(i, length) << std::endl; 
        }
    }
    unbindfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of (unnormalized) conditional cleavage rates
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-cleave-rates.tsv";
    else 
        oss << prefix << "-small23-cleave-rates.tsv";  
    std::ofstream cleavefile(oss.str());
    cleavefile << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (cleavefile.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length; ++j)
            {
                cleavefile << cleave_rates(i,j) << "\t";
            }
            cleavefile << cleave_rates(i, length) << std::endl; 
        }
    }
    cleavefile.close();
    oss.clear();
    oss.str(std::string());
  
    // Write matrix of normalized unconditional unbinding rates
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-norm-unbind.tsv";
    else 
        oss << prefix << "-small23-norm-unbind.tsv";  
    std::ofstream unbindfile2(oss.str());
    unbindfile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (unbindfile2.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                unbindfile2 << norm_unbind(i,j) << "\t";  
            }
            unbindfile2 << norm_unbind(i, length-1) << std::endl; 
        }
    }
    unbindfile2.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of normalized conditional cleavage rates
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-norm-cleave.tsv";
    else 
        oss << prefix << "-small23-norm-cleave.tsv";  
    std::ofstream cleavefile2(oss.str());
    cleavefile2 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (cleavefile2.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                cleavefile2 << norm_cleave(i,j) << "\t"; 
            }
            cleavefile2 << norm_cleave(i, length-1) << std::endl; 
        }
    }
    cleavefile2.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of asymptotically determined normalized conditional
    // unbinding rate tradeoff constants
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-lim-norm-unbind.tsv";
    else 
        oss << prefix << "-small23-lim-norm-unbind.tsv";  
    std::ofstream unbindfile3(oss.str()); 
    unbindfile3 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (unbindfile3.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                unbindfile3 << lim_norm_unbind(i,j) << "\t"; 
            }
            unbindfile3 << lim_norm_unbind(i, length-1) << std::endl; 
        }
    }
    unbindfile3.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of asymptotically determined normalized conditional
    // cleavage rate tradeoff constants
    if (idx_fixed_large == 0)
        oss << prefix << "-large01-lim-norm-cleave.tsv";
    else 
        oss << prefix << "-small23-lim-norm-cleave.tsv";  
    std::ofstream cleavefile3(oss.str()); 
    cleavefile3 << std::setprecision(std::numeric_limits<double>::max_digits10 - 1);
    if (cleavefile3.is_open())
    {
        for (unsigned i = 0; i < n; ++i)
        {
            for (unsigned j = 0; j < length - 1; ++j)
            {
                cleavefile3 << lim_norm_cleave(i,j) << "\t"; 
            }
            cleavefile3 << lim_norm_cleave(i, length-1) << std::endl; 
        }
    }
    cleavefile3.close(); 

    delete constraints;
    delete dict;
}

int main(int argc, char** argv)
{
    unsigned n;
    sscanf(argv[2], "%u", &n);
    runConstrainedSampling(0, n, argv[1]); 
    runConstrainedSampling(3, n, argv[1]); 

    return 0;
}
