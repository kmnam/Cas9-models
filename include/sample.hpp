#ifndef SAMPLE_HPP
#define SAMPLE_HPP

#include <assert.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>
#include <random>
#include <Eigen/Dense>

/*
 * Functions for random sampling.
 *
 * Authors:
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 * Last updated:
 *     12/2/2019
 */
using namespace Eigen;

// Instantiate a gamma distribution with alpha = 1
std::gamma_distribution<> gamma_dist(1.0);

MatrixXd sampleFromSimplex(const Ref<const MatrixXd>& vertices, unsigned npoints, std::mt19937& rng)
{
    /*
     * Given an array of vertices for a simplex and a desired number of 
     * points, randomly sample the given number of points from the 
     * uniform density (i.e., flat Dirichlet) on the simplex.
     *
     * Parameters
     * ----------
     * MatrixXd vertices
     *     (D+1) x D matrix of vertex coordinates, with each row a vertex. 
     * unsigned points
     *     Number of points to sample from the simplex.
     * std::mt19937& rng
     *     Reference to random number generator instance.   
     */
    unsigned dim = vertices.cols();     // Dimension of the ambient space
    unsigned nvert = vertices.rows();   // Number of vertices
    assert(nvert == dim + 1);

    // Sample the desired number of points from the flat Dirichlet 
    // distribution on the standard simplex of appropriate dimension
    MatrixXd barycentric(npoints, dim + 1);
    for (unsigned i = 0; i < npoints; i++)
    {
        // Sample (dim + 1) independent Gamma-distributed variables 
        // with alpha = 1, and normalize by their sum
        for (unsigned j = 0; j < dim + 1; j++) barycentric(i,j) = gamma_dist(rng);
        barycentric.row(i) = barycentric.row(i) / barycentric.row(i).sum();
    }
   
    // Convert from barycentric coordinates to Cartesian coordinates
    MatrixXd points(npoints, dim);
    for (unsigned i = 0; i < npoints; i++)
        points.row(i) = barycentric.row(i) * vertices;

    return points;
}

std::pair<MatrixXd, MatrixXd> sampleFromConvexPolytopeTriangulation(std::string triangulation_file,
                                                                  unsigned npoints, std::mt19937& rng)
{
    /*
     * Given a .delv file specifying a convex polytope in terms of its 
     * vertices and its Delaunay triangulation, parse the simplices of
     * the triangulation and sample uniformly from the polytope,
     * returning the vertices of the polytope and the sampled points.  
     */
    // Vector of vertex coordinates
    std::vector<std::vector<double> > vertices;

    // Vector of vertex indices identifying the simplices in the triangulation
    std::vector<std::vector<unsigned> > simplices;

    // Vector of simplex volumes
    std::vector<double> volumes;

    // Parse the input triangulation file to:
    // (1) parse the vertices of the polytope
    // (2) group the simplices by their volumes
    std::string line;
    std::ifstream infile(triangulation_file);
    unsigned dim = 0;
    std::regex regex;
    std::string pattern;
    if (!infile.is_open()) 
    {
        std::cerr << "File not found" << std::endl;
        throw std::exception();
    }

    while (std::getline(infile, line))
    {
        // Each vertex is specified as a space-delimited line
        if (line.compare(0, 1, "{") != 0)
        {
            std::istringstream iss(line);
            std::vector<double> vertex;
            std::string token;
            while (std::getline(iss, token, ' '))
                vertex.push_back(std::stod(token));
            vertices.push_back(vertex);
            if (dim == 0)
            {
                dim = vertex.size();
                // Define a regular expression for subsequent lines
                // in the file specifying the simplices and their volumes
                pattern = "^\\{";
                for (unsigned i = 0; i < dim; i++) pattern = pattern + "([[:digit:]]+) ";
                pattern = pattern + "([[:digit:]]+)\\} ([[:digit:]]+)(\\/[[:digit:]]+)?$";
                regex.assign(pattern);
            }
        }
        // Each simplex is specified as a space-delimited string of  
        // vertex indices, surrounded by braces, followed by its volume
        // as a rational number 
        else
        {
            if (dim == 0)
            {
                std::cerr << "Vertices of polytope not specified" << std::endl;
                throw std::exception();
            }
            else
            {
                // Match the contents of each line to the regular expression
                std::smatch matches;
                std::vector<unsigned> vertex_indices;
                if (std::regex_match(line, matches, regex))
                {
                    if (matches.size() == dim + 4)
                    {
                        for (unsigned i = 1; i < matches.size() - 2; i++)
                        {
                            std::ssub_match match = matches[i];
                            std::string match_str = match.str();
                            vertex_indices.push_back(std::stoul(match_str));
                        }
                        simplices.push_back(vertex_indices);
                        std::string volume_num = matches[matches.size() - 2].str();
                        std::string volume_den = matches[matches.size() - 1].str();
                        double volume = 0.0;
                        // The line matches the regular expression and the volume
                        // was specified as an integer
                        if (volume_den.empty())
                            volume = std::stod(volume_num);
                        // The line matches the regular expression and the volume
                        // was specified as a fraction
                        else
                            volume = std::stod(volume_num) / std::stod(volume_den.erase(0, 1));
                        volumes.push_back(volume);
                    }
                    // The line does not match the regular expression
                    else
                    {
                        std::cerr << "Incorrect number of matches" << std::endl;
                        throw std::exception();
                    }
                }
                else
                {
                    std::cerr << "Does not match regex" << std::endl;
                    throw std::exception();
                }
            }
        }
    }

    // Instantiate a categorical distribution with probabilities 
    // proportional to the simplex volumes 
    double sum_volumes = 0.0;
    for (auto&& v : volumes) sum_volumes += v;
    for (auto&& v : volumes) v /= sum_volumes;
    std::discrete_distribution<> dist(volumes.begin(), volumes.end());

    // Maintain an array of points ...
    MatrixXd sample(npoints, dim);
    for (unsigned i = 0; i < npoints; i++)
    {
        // Sample a simplex with probability proportional to its volume
        int j = dist(rng);

        // Get the vertices of the simplex
        MatrixXd simplex(dim + 1, dim);
        for (unsigned k = 0; k < dim + 1; k++)
        {
            unsigned index = simplices[j][k];
            for (unsigned l = 0; l < dim; l++) simplex(k,l) = vertices[index][l];
        }

        // Sample a point from the simplex
        sample.row(i) = sampleFromSimplex(simplex, 1, rng);
    }

    // Write vertex coordinates to a matrix
    MatrixXd vertices_mat(vertices.size(), dim);
    for (unsigned i = 0; i < vertices.size(); ++i)
    {
        for (unsigned j = 0; j < dim; ++j)
        {
            vertices_mat(i,j) = vertices[i][j];
        }
    }
    
    return std::make_pair(vertices_mat, sample);
}

#endif
