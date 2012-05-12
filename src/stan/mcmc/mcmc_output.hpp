#ifndef __STAN__MCMC__MCMC_OUTPUT_HPP__
#define __STAN__MCMC__MCMC_OUTPUT_HPP__

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <stan/math/matrix.hpp>

/*#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>*/


#include <iostream>

namespace stan {
  namespace mcmc {

    
    /** 
     * Post-processing for MCMC samples.
     *   
     * This class calculates
     * <ul>
     * <li> effective sample size
     * <li> split R-hat
     * </ul>
     * for MCMC samples in multiple chains.
     *
     * Currently, the implementation assumes everything
     * is done in batch.
     */
    class mcmc_output {
    public:
      /** 
       * Default constructor
       */
      mcmc_output() :
        nChains_(0), nSamplesPerChain_(0) {
      }
      
      /** 
       * Constructs mcmc_output with samples.
       * 
       * @param samples MCMC samples. Each vector of doubles represents
       *   a chain. Each chain must be the same length.
       */
      mcmc_output(std::vector< std::vector<double> > samples) : 
        samples_(samples),
        nChains_(samples.size()),
        nSamplesPerChain_(0) {
        // check chain
        if (nChains_ > 1) {
          nSamplesPerChain_ = samples_[0].size();
          for (size_t chain = 1; chain < nSamplesPerChain_; chain++) {
            if (samples_[chain].size() != nSamplesPerChain_) {
              throw std::domain_error("chain length must be the same");
            }
          }
        }
      }
      
      /** 
       * Add samples.
       * 
       * @param chain A single chain to add to the existing chains in the
       *  object.
       */
      void add_chain(std::vector<double> chain) {
        // check chain
        if (nChains_ > 0) {
          if (chain.size() != nSamplesPerChain_) {
            throw std::domain_error("chain length must be the same");
          }
        }
        samples_.push_back(chain);
        nChains_++;
        nSamplesPerChain_ = chain.size();
      }

      /** 
       * Effective sample size calculation.
       *
       * Implementation matches BDA3's effective size description.
       * 
       * @return the effective sample size.
       */
      double effectiveSize() {
        size_t m = nChains_;
        size_t n = nSamplesPerChain_;
        
        std::vector<double> chain_mean;
        std::vector<double> chain_var;
        for (size_t chain = 0; chain < m; chain++) {
          chain_mean.push_back(stan::math::mean(samples_[chain]));
          chain_var.push_back(stan::math::variance(samples_[chain]));
        }
        double var_plus = stan::math::mean(chain_var)*(n-1)/n + stan::math::variance(chain_mean);
        
        std::vector<double> rho_hat_t;
        double rho_hat = 0;
        for (size_t t = 1; (t < n && rho_hat >= 0); t++) {
          double variogram = 0;
          for (size_t chain = 0; chain < m; chain++) {
            for (size_t ii = 0; ii < n-t; ii++) {
              double diff = samples_[chain][ii] - samples_[chain][ii+t]; 
              variogram += diff * diff;
            }
          }
          variogram /= m * (n-t);
          rho_hat = 1 - variogram / (2 * var_plus);
          if (rho_hat >= 0)
            rho_hat_t.push_back(rho_hat);
        }        
      
        double ess = m*n;
        if (rho_hat_t.size() > 0) {
          ess /= 1 + 2 * stan::math::sum(rho_hat_t);
        }
        return ess;
      }

      /**
       * Convergence monitoring: calculates R-hat on split chains.
       *
       * Implementation matches BDA3's R-hat on split chains. Each
       * chain is split into halves.
       *
       * @return the split R-hat value.
       */
      double splitRHat() {
        //using namespace boost::accumulators;
        size_t m = nChains_*2;
        size_t n = nSamplesPerChain_/2;
        
        std::vector<double> chain_mean;
        std::vector<double> chain_var;
        std::vector<double> chain(n);
        for (size_t ii = 0; ii < m; ii++) {
          if (ii % 2 == 0)
            chain.assign(samples_[ii/2].begin(), samples_[ii/2].begin()+n);
          else
            chain.assign(samples_[ii/2].end()-n, samples_[ii/2].end());
          chain_mean.push_back(stan::math::mean(chain));
          chain_var.push_back(stan::math::variance(chain));
        }
        double var_between = n * stan::math::variance(chain_mean);
        double var_within = stan::math::mean(chain_var);
        
        // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
        return sqrt((var_between/var_within + n-1)/n);
      }

      /** 
       * Returns all samples as a vector.
       * 
       * 
       * @return all samples as a single vector
       */
      std::vector<double> allSamples() {
        std::vector<double> samples;
        for (size_t chain = 0; chain < nChains_; chain++) {
          samples.insert(samples.end(), samples_[chain].begin(), samples_[chain].end());
        }
        return samples;
      }

      /**
       * Mean across all chains.
       *
       * @return the mean of the samples.
       */
      double mean() {
        /*std::vector<double> samples;
        for (size_t chain = 0; chain < nChains_; chain++) {
          
        }
        std::vector<double> chain_mean;
        for (size_t chain = 0; chain < nChains_; chain++) {
          chain_mean.push_back(stan::math::mean(samples_[chain]));
        }*/
        return (stan::math::mean(this->allSamples()));
      }
      
      /**
       * Variance across all chains.
       *
       * @return the variance of the samples.
       */
      double variance() {
        return (stan::math::variance(this->allSamples()));
      }

      /**
       * Number of chains in the mcmc_object.
       *
       * @return number of chains
       */
      size_t nChains() {
        return nChains_;
      }

      /**
       * Number of samples per chain.
       *
       * @return number of samples per chain.
       */
      size_t nSamplesPerChain() {
        return nSamplesPerChain_;
      }
      
      friend std::ostream& operator<<(std::ostream& os, const mcmc_output& mcmc_output) {
        os << mcmc_output.samples_.size() << " chains  ";
        if (mcmc_output.samples_.size() > 0)
          os << ", " << mcmc_output.samples_[0].size() << " samples per chain";
        os << std::endl;
        return os;
      }

    private:
      std::vector< std::vector<double> > samples_;
      size_t nChains_;
      size_t nSamplesPerChain_;
    };


    class mcmc_output_factory {
    private:
      // List of filenames
      std::vector<std::string> filenames_;
      
      /** 
       * Strips the header from a Stan samples file.
       *
       * The file is advanced past the header. Each line of the header
       * is assumed to start with the character #.
       * 
       * @param[in,out] file Input file stream.
       * @return Number of lines stripped from the output.
       */
      int stripHeader(std::fstream& file) {
        int n = 0;
        while (file.peek() == '#') {
          file.ignore(10000, '\n');
          n++;
        }
        return n;
      }
      
      /** 
       * Indentifies the column number of the variable.
       *
       * The file must be at the header of file. After this function, file is
       * moved to the next line.
       * 
       * @param[in,out] file The filestream.
       * @param variable The variable to find in the file.
       * @return Zero-indexed column index of the variable.
       * @exception std::runtime_error if the variable is not found in the file.
       */
      int identifyColumnNumber(std::fstream& file, std::string variable) {
        int column = 0;
        std::stringstream currvar("");
        char c;
        file.get(c);
        
        while (currvar.str() != variable) {
          currvar.str("");
          if (c == ',') {
            column++;
            file.get(c);
          } else if (c == '\n') {
            throw std::runtime_error("variable could not be found in file");
          }
          while (c != ',' && c != '\n') {
            currvar << c;
            file.get(c);
          }
        }
        if (c != '\n') {
          file.ignore(10000, '\n');
        }
        return column;
      }
      
      double getSample(std::fstream &file, int column) {
        char c;
        double sample;
        file >> sample;
        for (int ii = 1; ii <= column; ii++) {
          file.get(c);
          file >> sample;
        }
        if (c != '\n') {
          file.ignore(10000, '\n');
        }
        return sample;
      }
      
      std::vector<double> getSamples(std::fstream &file, int column) {
        std::vector<double> samples;
        while (file.peek() != std::istream::traits_type::eof()) {
          samples.push_back(getSample(file, column));
        }
        return samples;
      }
      
    public:
      mcmc_output_factory() {
      }
      void addFile(std::string filename) {
        filenames_.push_back(filename);
      }
      
      /** 
       * Returns the variables available.
       *
       * Defaults to the first file. If file is beyond the size number
       * of files, throws an exception.
       * 
       * @param fileNumber File to return available variables
       * @return Vector of available variables.
       * @throws std::runtime_error if called with an unavailable fileNumber
       */
      std::vector<std::string> availableVariables(int fileNumber=0) {
        if (fileNumber >= filenames_.size()) {
          throw std::runtime_error("availableVariables called with fileNumber greater than number of files");
        }
        std::vector<std::string> vars;
        std::fstream file(filenames_[fileNumber].c_str(), std::fstream::in);
        stripHeader(file);
        
        std::stringstream currvar("");
        char c;
        file.get(c);
        while (c != '\n') {
          currvar.str("");
          if (c == ',') {
            file.get(c);
          }
          while (c != ',' && c != '\n') {
            currvar << c;
            file.get(c);
          }
          vars.push_back(currvar.str());
        }
        file.close();
        return vars;
      }
      
      /** 
       * Creates an <code>stan::mcmc::mcmc_output</code> variable for the variable
       * specified.
       *
       * If the variable is not a part of the files attached, should throw an exception.
       * 
       * @param variable Variable to create the mcmc_output for.
       * @return An mcmc_output variable.
       */
      mcmc_output create(std::string variable) {
        mcmc_output var;
        // loop over all files
        for (size_t ii = 0; ii < filenames_.size(); ii++) {
          // 1. open file: filenames_[ii];
          std::fstream in(filenames_[ii].c_str(), std::fstream::in);
          if (!in.is_open()) {
            std::cerr << "mcmc_output warning: " << filenames_[ii] << " can not be open." << std::endl;
            continue;
          }
          // 1.1. Strip header
          stripHeader(in);
          // 1.2 count number of commas to skip
          int column = identifyColumnNumber(in, variable);
          // 2. read the column with the variable listed
          std::vector<double> samples = getSamples(in, column);
          in.close();
          
          // 3. add samples to mcmc_output
          var.add_chain(samples);
        }
        return var;
      }
    };

   
  }
}
#endif
