/**
 * Abbreviations in the below comments:
 * - LG:   line graph
 * - LGPs: line graph parameters
 * - SQP:  sequential quadratic programming
 *
 * **Author:**
 *     Kee-Myoung Nam
 *
 * **Last updated:**
 *     9/22/2022
 */

#include <fstream>
#include <string>
#include <boost/json/src.hpp>

using namespace boost::json; 

/**
 * Parse a JSON file specifying configurations for identifying optimal LGPs
 * against a set of measured cleavage/unbinding rates with SQP through
 * `fitLineParamsAgainstMeasuredRates`.
 *
 * @param filename Input JSON configurations file.
 * @returns `boost::json::value` instance containing the JSON data.  
 */
value parseConfigFile(const std::string filename)
{
    std::string line;
    std::ifstream infile(filename);
    stream_parser p; 
    error_code ec;  
    while (std::getline(infile, line))
    {
        p.write(line, ec); 
        if (ec)
            return nullptr; 
    }
    p.finish(ec); 
    if (ec)
        return nullptr;
    
    return p.release(); 
}

