/**
 * Functions for random sampling.
 *
 * **Authors:**
 *     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
 *
 * **Last updated:**
 *     2/1/2022
 */

#ifndef SAMPLE_HPP
#define SAMPLE_HPP

#include <assert.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <iomanip>
#include <limits>
#include <Eigen/Dense>
#include <boost/random.hpp>

using namespace Eigen;

/**
 * Given an array of vertices for a simplex and a desired number of 
 * points, randomly sample the given number of points from the 
 * uniform density (i.e., flat Dirichlet) on the simplex.
 *
 * @param vertices (D+1) x D matrix of vertex coordinates, with each row
 *                 a vertex. 
 * @param npoints  Number of points to sample from the simplex.
 * @param rng      Reference to existing random number generator instance.
 * @returns        (npoints) x D matrix of coordinates of the sampled points. 
 */
MatrixXd sampleFromSimplex(const Ref<const MatrixXd>& vertices, unsigned npoints,
                           boost::random::mt19937& rng)
{
    unsigned dim = vertices.cols();     // Dimension of the ambient space
    unsigned nv = vertices.rows();      // Number of vertices
    if (nv != dim + 1)
        throw std::invalid_argument("Input matrix of vertex coordinates does not specify a valid simplex"); 

    // Sample the desired number of points from the flat Dirichlet 
    // distribution on the standard simplex of appropriate dimension
    MatrixXd barycentric(npoints, dim + 1); 
    boost::random::gamma_distribution<double> gamma_dist(1.0);
    for (unsigned i = 0; i < npoints; ++i)
    {
        // Sample (dim + 1) independent Gamma-distributed variables 
        // with alpha = 1, and normalize by their sum
        for (unsigned j = 0; j < dim + 1; ++j)
            barycentric(i, j) = gamma_dist(rng);
        barycentric.row(i) = barycentric.row(i) / barycentric.row(i).sum();
    }
   
    // Convert from barycentric coordinates to Cartesian coordinates
    MatrixXd points(npoints, dim); 
    for (unsigned i = 0; i < npoints; ++i)
        points.row(i) = barycentric.row(i) * vertices; 

    return points;
}

/**
 * A simple wrapper class that stores information regarding a Delaunay
 * triangulation of a point-set in multi-dimensional space. 
 */
class DelaunayTriangulation 
{
    private:
        // Dimension of the ambient space
        int dim; 

        // Matrix of vertex coordinates (each row is a vertex)  
        Matrix<double, Dynamic, Dynamic> vertices;

        // Matrix of vertex indices identifying the full-dimensional simplices
        // in the triangulation 
        Matrix<int, Dynamic, Dynamic> simplices;

    public:
        /**
         * Empty constructor (assume that the ambient dimension is two). 
         */
        DelaunayTriangulation()
        {
            this->dim = 2;
            this->vertices.resize(0, 2);
            this->simplices.resize(0, 2);  
        }

        /**
         * Constructor with a specified set of vertices and simplices, 
         * given as `std::vector`s. 
         *
         * Every vertex should have the same number of coordinates and 
         * every simplex should have the same number of indices.
         *
         * @param vertices  Vector of vertices. 
         * @param simplices Vector of simplices. 
         * @throws std::invalid_argument If `vertices` or `simplices` has no 
         *                               no entries.
         * @throws std::runtime_error    If `vertices` or `simplices` has
         *                               entries of different lengths.  
         */
        DelaunayTriangulation(std::vector<std::vector<double> > vertices, 
                              std::vector<std::vector<int> > simplices)
        {
            if (vertices.begin() == vertices.end())
                throw std::invalid_argument("No vertices specified");
            if (simplices.begin() == simplices.end())
                throw std::invalid_argument("No simplices specified");
            int dim = vertices[0].size();
            this->vertices.resize(vertices.size(), dim); 
            this->simplices.resize(simplices.size(), dim + 1);  
            for (unsigned i = 0; i < vertices.size(); ++i)
            {
                if (vertices[i].size() != dim)
                    throw std::invalid_argument("Specified vertices do not lie in the same ambient space"); 
                for (unsigned j = 0; j < vertices[i].size(); ++j)
                    this->vertices(i, j) = vertices[i][j]; 
            }
            for (unsigned i = 0; i < simplices.size(); ++i)
            {
                if (simplices[i].size() != dim + 1) 
                    throw std::invalid_argument("Specified simplices do not lie in the same ambient space"); 
                for (unsigned j = 0; j < simplices[i].size(); ++j)
                    this->simplices(i, j) = simplices[i][j]; 
            }
            this->dim = dim; 
        }

        /**
         * Empty destructor. 
         */
        ~DelaunayTriangulation()
        {
        }

        /**
         * Get the dimension of the ambient space. 
         *
         * @returns Dimension of the ambient space.
         */
        int getDim()
        {
            return this->dim; 
        }

        /**
         * Get the i-th vertex in the Delaunay triangulation. 
         *
         * @param i Index of desired vertex. 
         * @returns Desired vertex. 
         */
        VectorXd getVertex(int i) 
        {
            return this->vertices.row(i); 
        }

        /**
         * Get the *vertex coordinates* of the i-th simplex in the Delaunay
         * triangulation.
         *
         * @param i Index of desired simplex. 
         * @returns Matrix of vertex coordinates of desired simplex.  
         */
        MatrixXd getSimplex(int i)
        {
            MatrixXd simplex(this->dim + 1, this->dim);
            for (unsigned j = 0; j < this->dim + 1; ++j)
                simplex.row(j) = this->vertices.row(this->simplices(i, j));

            return simplex;  
        }
        
        /**
         * Return the volumes of the simplices in the Delaunay triangulation,
         * each multiplied by `(this->dim)!`, in a 1-D vector. 
         *
         * @returns VectorXd Vector of scaled simplex volumes.   
         */
        VectorXd getScaledVolumes()
        {
            VectorXd volumes(this->simplices.rows()); 

            // Run through the simplices in the triangulation ... 
            for (unsigned i = 0; i < this->simplices.rows(); ++i)
            {
                // Compute the related determinant 
                MatrixXd A(this->dim, this->dim);
                VectorXd v = this->vertices.row(this->simplices(i, 0)); 
                for (unsigned j = 1; j < this->dim + 1; ++j)
                    A.col(j) = this->vertices.row(this->simplices(i, j)) - v;
                volumes(i) = std::abs(A.determinant());  
            }

            return volumes; 
        } 
}; 

/**
 * Parse the given .delv file specifying a convex polytope in terms of its
 * vertices and its Delaunay triangulation, and return the triangulation as 
 * a `DelaunayTriangulation` instance.
 *
 * @param filename Path to input .delv polytope triangulation file.
 * @returns        `DelaunayTriangulation` instance containing the triangulation
 *                 data. 
 */
DelaunayTriangulation parseTriangulationFile(const std::string filename)
{
    // Vector of vertex coordinates
    std::vector<std::vector<double> > vertices;

    // Vector of vertex indices identifying the simplices in the triangulation
    std::vector<std::vector<int> > simplices;

    // Parse the input triangulation file to:
    // (1) parse the vertices of the polytope
    // (2) group the simplices by their volumes
    std::string line;
    std::ifstream infile(filename); 
    int dim = 0;
    std::regex regex;
    std::string pattern;
    if (!infile.is_open())
        throw std::invalid_argument("Input triangulation file not found"); 

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
                throw std::runtime_error("Vertices of polytope not specified");
            }
            else
            {
                // Match the contents of each line to the regular expression
                std::smatch matches;
                std::vector<int> vertex_indices;
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
                        /*
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
                        */
                    }
                    // The line does not match the regular expression
                    else
                    {
                        throw std::runtime_error("Incorrectly formatted line in .delv file");
                    }
                }
                else
                {
                    throw std::runtime_error("Incorrectly formatted line in .delv file"); 
                }
            }
        }
    }

    return DelaunayTriangulation(vertices, simplices); 
}

/**
 * Given a .delv file specifying a convex polytope in terms of its vertices
 * and its Delaunay triangulation, parse the simplices of the triangulation
 * and sample uniformly from the polytope, returning the vertices of the
 * polytope and the sampled points.
 *
 * @param filename Path to input .delv polytope triangulation file.
 * @param npoints  Number of points to sample from the polytope.
 * @param rng      Reference to existing random number generator instance.
 * @returns        Delaunay triangulation parsed from the given file and the 
 *                 matrix of sampled points. 
 */
std::pair<DelaunayTriangulation, MatrixXd> sampleFromConvexPolytopeTriangulation(std::string filename,
                                                                                 unsigned npoints,
                                                                                 boost::random::mt19937& rng)
{
    // Parse the given .delv file
    DelaunayTriangulation tri = parseTriangulationFile(filename); 

    // Instantiate a categorical distribution with probabilities 
    // proportional to the simplex volumes 
    VectorXd volumes = tri.getScaledVolumes(); 
    VectorXd probs = volumes / volumes.sum();
    std::vector<double> probs_vec; 
    for (unsigned i = 0; i < probs.size(); ++i)
        probs_vec.push_back(probs(i)); 
    boost::random::discrete_distribution<> dist(probs_vec); 

    // Maintain an array of points ...
    int dim = tri.getDim(); 
    MatrixXd sample(npoints, dim); 
    for (unsigned i = 0; i < npoints; i++)
    {
        // Sample a simplex with probability proportional to its volume
        int j = dist(rng);

        // Get the corresponding simplex 
        MatrixXd simplex = tri.getSimplex(j); 

        // Sample a point from the simplex
        sample.row(i) = sampleFromSimplex(simplex, 1, rng);
    }
   
    return std::make_pair(tri, sample);
}

#endif
