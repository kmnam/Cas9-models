#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <utility>
#include <iomanip>
#include <tuple>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include "../include/graphs/line.hpp"
#include "../include/sample.hpp"

/*
 * Estimates cleavage probabilities and unbinding rates with respect to 
 * distal-mismatch substrates for the line-graph Cas9 model by running 
 * Markov process simulations.
 *
 * Call as: 
 *     ./bin/lineDistalMismatch [SAMPLING POLYTOPE .delv FILE] [OUTPUT FILE PREFIX] [NUMBER OF POINTS TO SAMPLE] [NUMBER OF SIMULATIONS]
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     7/30/2021
 */
using namespace Eigen;
using boost::multiprecision::number;
using boost::multiprecision::mpfr_float_backend;

const int length = 20;

// Instantiate random number generator 
boost::random::mt19937 rng(1234567890);

std::tuple<double, double, double> estimateStatsFromSimulations(
    std::vector<std::vector<std::pair<int, double> > > simulations,
    const int upper_exit_vertex, const int lower_exit_vertex)
{
    /*
     * Estimate the cleavage probability, conditional cleavage rate, and 
     * conditional unbinding rate from a set of Markov process simulations. 
     */ 
    // Compute the fraction of simulations that result in upper exit, the
    // average time spent until lower exit (among the simulations for which 
    // lower exit occurs), and the average time spent until upper exit
    // (among the simulations for which upper exit occurs)
    double nsims_upper_exit = 0;
    double total_time_to_lower_exit = 0.0;
    double total_time_to_upper_exit = 0.0; 
    for (auto&& sim : simulations)
    {
        std::pair<int, double> last_visited = sim[sim.size() - 1];
        if (last_visited.first == lower_exit_vertex)
        {
            total_time_to_lower_exit += last_visited.second;
        }
        else if (last_visited.first == upper_exit_vertex)
        {
            nsims_upper_exit += 1;
            total_time_to_upper_exit += last_visited.second; 
        } 
    }
    double prob = nsims_upper_exit / simulations.size();
    double avg_time_to_lower_exit = total_time_to_lower_exit / (simulations.size() - nsims_upper_exit); 
    double avg_time_to_upper_exit = total_time_to_upper_exit / nsims_upper_exit; 

    return std::make_tuple(prob, avg_time_to_lower_exit, avg_time_to_upper_exit); 
}

template <typename T>
Matrix<double, Dynamic, Dynamic> estimateStats(const Ref<const Matrix<double, Dynamic, 1> >& params,
                                               const unsigned nsims)
{
    /*
     * Compute the cleavage probabilities and unbinding rates with respect to 
     * single-mismatch substrates. 
     */
    // Array of DNA/RNA match parameters
    std::array<T, 2> match_params;
    match_params[0] = static_cast<T>(std::pow(10.0, params(0)));
    match_params[1] = static_cast<T>(std::pow(10.0, params(1)));

    // Array of DNA/RNA mismatch parameters
    std::array<T, 2> mismatch_params;
    mismatch_params[0] = static_cast<T>(std::pow(10.0, params(2)));
    mismatch_params[1] = static_cast<T>(std::pow(10.0, params(3)));

    // Populate each rung with DNA/RNA match parameters
    LineGraph<T>* model = new LineGraph<T>(length);
    for (unsigned j = 0; j < length; ++j)
        model->setLabels(j, match_params);
    
    // Estimate cleavage probability and conditional mean first passage time
    // to cleaved state
    std::vector<std::vector<std::pair<int, double> > > simulations = model->simulate(nsims, 1, 1, rng);
    std::tuple<double, double, double> sim_stats = estimateStatsFromSimulations(simulations, length + 1, -1);
    Matrix<double, Dynamic, Dynamic> stats(length + 1, 3);
    stats(0, 0) = std::get<0>(sim_stats);
    stats(0, 2) = 1 / std::get<2>(sim_stats);

    // Estimate unconditional mean first passage time to unbound state
    simulations = model->simulate(nsims, 1, 0, rng); 
    sim_stats = estimateStatsFromSimulations(simulations, length + 1, -1); 
    stats(0, 1) = 1 / std::get<1>(sim_stats);  

    // Introduce single mismatches and re-compute cleavage probability
    // and mean first passage times
    for (int j = 1; j <= length; ++j) 
    {
        model->setLabels(length - j, mismatch_params);
        simulations = model->simulate(nsims, 1, 1, rng);
        sim_stats = estimateStatsFromSimulations(simulations, length + 1, -1);
        stats(j, 0) = std::get<0>(sim_stats);
        stats(j, 2) = 1 / std::get<2>(sim_stats);
        simulations = model->simulate(nsims, 1, 0, rng); 
        sim_stats = estimateStatsFromSimulations(simulations, length + 1, -1);  
        stats(j, 1) = 1 / std::get<1>(sim_stats); 
    }

    delete model;
    return stats;
}

int main(int argc, char** argv)
{
    // Sample model parameter combinations
    unsigned n, nsims;
    sscanf(argv[3], "%u", &n);
    sscanf(argv[4], "%u", &nsims); 
    Matrix<double, Dynamic, Dynamic> vertices, params;
    try
    {
        std::tie(vertices, params) = sampleFromConvexPolytopeTriangulation<double>(argv[1], n, rng);
    }
    catch (const std::exception& e)
    {
        throw;
    }

    // Compute cleavage probabilities and unbinding rates
    Matrix<double, Dynamic, Dynamic> probs(n, length + 1);
    Matrix<double, Dynamic, Dynamic> uncond_rates(n, length + 1);
    Matrix<double, Dynamic, Dynamic> cond_upper_rates(n, length + 1);
    //Matrix<double, Dynamic, Dynamic> cond_lower_rates(n, length + 1);
    for (unsigned i = 0; i < n; ++i)
    {
        Matrix<double, Dynamic, Dynamic> stats = estimateStats<number<mpfr_float_backend<100> > >(params.row(i), nsims).transpose();
        probs.row(i) = stats.row(0);
        uncond_rates.row(i) = stats.row(1);
        cond_upper_rates.row(i) = stats.row(2);
        //cond_lower_rates.row(i) = stats.row(3);
    }

    // Write sampled parameter combinations to file
    std::ostringstream oss;
    oss << argv[2] << "-params.tsv";
    std::ofstream samplefile(oss.str());
    samplefile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (samplefile.is_open())
    {
        for (unsigned i = 0; i < params.rows(); i++)
        {
            for (unsigned j = 0; j < params.cols() - 1; j++)
            {
                samplefile << params(i,j) << "\t";
            }
            samplefile << params(i,params.cols()-1) << std::endl;
        }
    }
    samplefile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of cleavage probabilities
    oss << argv[2] << "-probs.tsv";
    std::ofstream probsfile(oss.str());
    probsfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (probsfile.is_open())
    {
        for (unsigned i = 0; i < probs.rows(); i++)
        {
            for (unsigned j = 0; j < probs.cols() - 1; j++)
            {
                probsfile << probs(i,j) << "\t";
            }
            probsfile << probs(i,probs.cols()-1) << std::endl;
        }
    }
    probsfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of unconditional unbinding rates
    oss << argv[2] << "-uncondRates.tsv";
    std::ofstream ratesfile(oss.str());
    ratesfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (ratesfile.is_open())
    {
        for (unsigned i = 0; i < uncond_rates.rows(); i++)
        {
            for (unsigned j = 0; j < uncond_rates.cols() - 1; j++)
            {
                ratesfile << uncond_rates(i,j) << "\t";
            }
            ratesfile << uncond_rates(i, uncond_rates.cols()-1) << std::endl;
        }
    }
    ratesfile.close();
    oss.clear();
    oss.str(std::string());

    // Write matrix of conditional cleavage rates
    oss << argv[2] << "-condUpperRates.tsv";
    std::ofstream upperfile(oss.str());
    upperfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (upperfile.is_open())
    {
        for (unsigned i = 0; i < cond_upper_rates.rows(); i++)
        {
            for (unsigned j = 0; j < cond_upper_rates.cols() - 1; j++)
            {
                upperfile << cond_upper_rates(i,j) << "\t";
            }
            upperfile << cond_upper_rates(i, cond_upper_rates.cols()-1) << std::endl;
        }
    }
    upperfile.close();
    oss.clear(); 
    oss.str(std::string());

    /*
    // Write matrix of conditional unbinding rates
    oss << argv[2] << "-condLowerRates.tsv";
    std::ofstream lowerfile(oss.str());
    lowerfile << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (lowerfile.is_open())
    {
        for (unsigned i = 0; i < cond_lower_rates.rows(); i++)
        {
            for (unsigned j = 0; j < cond_lower_rates.cols() - 1; j++)
            {
                lowerfile << cond_lower_rates(i,j) << "\t";
            }
            lowerfile << cond_lower_rates(i, cond_lower_rates.cols()-1) << std::endl;
        }
    }
    lowerfile.close();
    oss.clear(); 
    oss.str(std::string());
    */ 
   
    return 0;
}
