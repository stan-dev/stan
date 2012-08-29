#ifndef __STAN__MCMC__CHAINS_HPP__
#define __STAN__MCMC__CHAINS_HPP__

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <fstream>
#include <cstdlib>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
// #include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/covariance.hpp>
#include <boost/accumulators/statistics/variates/covariate.hpp>


#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/additive_combine.hpp>

#include <stan/math/matrix.hpp>
#include <stan/prob/autocorrelation.hpp>
#include <stan/prob/autocovariance.hpp>

namespace stan {  

  namespace mcmc {

    const std::vector<std::string>& 
    test_match_return_names(const std::vector<std::string>& names,
                            const std::vector<std::vector<size_t> >& dimss) {
      if (names.size() == dimss.size())
        return names;
      std::stringstream msg;
      msg << "names and dimss mismatch in size"
          << " names.size()=" << names.size()
          << " dimss.size()=" << dimss.size();
      throw std::invalid_argument(msg.str());
    }

    void validate_prob(double p) {
      // test this way so NaN fails
      if (p >= 0.0 && p <= 1.0) 
        return;
      std::stringstream msg;
      msg << "require probabilities to be finite between 0 and 1 inclusive."
          << " found p=" << p;
      throw std::invalid_argument(msg.str());
    }

    /**
     * Validate the specified indexes with respect to the
     * specified dimensions.
     *
     * @param dims Dimensions of array.
     * @param idxs Indexes into array.
     * @throw std::invalid_argument If the two arrays are different
     * sizes.
     * @throw std::out_of_range If any of the indexes is greater than
     * or equal to its correpsonding dimension.
     */
    void validate_dims_idxs(const std::vector<size_t>& dims,
                            const std::vector<size_t>& idxs) {
      if (idxs.size() != dims.size()) {
        std::stringstream msg;
        msg << "index vector and dims vector must be same size."
            << "; idxs.size()=" << idxs.size()
            << "; dims.size()=" << dims.size();
        throw std::invalid_argument(msg.str());
      }
      for (size_t i = 0; i < idxs.size(); ++i) {
        if (idxs[i] >= dims[i]) {
          std::stringstream msg;
          msg << "indexes must be within bounds."
              << "; idxs[" << i << "]=" << idxs[i]
              << "; dims[" << i << "]=" << dims[i];
          throw std::out_of_range(msg.str());
        }
      }
    }

    /**
     * Return the offset in last-index major indexing for the
     * specified indexes given the specified number of dimensions.
     * If both sequences are empty, the index returned is 0.
     *
     * @param dims Sequence of dimensions.
     * @param idxs Sequence of inndexes.
     * @return Offset of indexes given dimensions.
     * @throw std::invalid_argument If the sizes of the index
     * and dimension sequences is different.
     * @throw std::out_of_range If one of the indexes is greater
     * than or equal to the corresponding index.
     */
    size_t get_offset(const std::vector<size_t>& dims, 
                      const std::vector<size_t>& idxs) {
      validate_dims_idxs(dims,idxs);
      if (idxs.size() == 0)
        return 0;
      if (idxs.size() == 1)
        return idxs[0];
      size_t pos(0);
      // OK, stop at 1
      for (size_t i = idxs.size(); --i != 0; ) {
        pos += idxs[i];
        pos  *= dims[i-1];
      }
      return pos + idxs[0];
    }

    /**
     * Increments the specified indexes to refer to the next value
     * in an array given by the specified dimensions.  The indexing
     * is in last-index major order, which is column-major for
     * matrices.
     *
     * <p>The first index in the sequence is all zeroes.
     * Incrementing the last index, whose values are the dimensions
     * minus one, returns the all-zero matrix.
     *
     * <p>Given <code>dims == (2,2,2)</code>, the sequence of
     * indexes are 
     *
     * <code>[0 0 0]</code>, 
     * <code>[1 0 0]</code>, 
     * <code>[0 1 0]</code>, 
     * <code>[1 1 0]</code>, 
     * <code>[0 0 1]</code>, 
     * <code>[1 0 1]</code>, 
     * <code>[0 1 1]</code>, 
     * <code>[1 1 1]</code>,
     * <code>[0 0 0]</code>, 
     * <code>[1 0 0]</code>, ...
     *
     * @param dims Dimensions of array.
     * @param idxs Indexes into array.
     * @throws std::invalid_argument If the dimensions and indexes
     * are not the same size.
     * @throws std::out_of_range If an index is greater than or equal
     * to the corresponding dimension.
     */
    void
    increment_indexes(const std::vector<size_t>& dims,
                      std::vector<size_t>& idxs) {
      validate_dims_idxs(dims,idxs);
      for (size_t i = 0; i < dims.size(); ++i) {
        ++idxs[i];
        if (idxs[i] < dims[i]) 
          return;
        idxs[i] = 0;
      }
    }

    /**
     * Write a permutation into the specified vector of the specified
     * size using the specified Boost random number generator.  The
     * vector will be resized to the specified size.
     *
     * @tparam RNG Type of random number geneation engine
     * @param x Vector into which to write the permutation
     * @param n Size of permutation to create
     * @param rng Random-number generator.
     */
    template <class RNG>
    void permutation(std::vector<size_t>& x,
                     size_t n,
                     RNG& rng) {
      x.resize(n);
      for (size_t i = 0; i < n; ++i)
        x[i] = i;
      if (x.size() < 2) return;
      for (int i = x.size(); --i != 0; ) {
        boost::random::uniform_int_distribution<size_t> uid(0,i);
        size_t j = uid(rng);
        size_t temp = x[i];
        x[i] = x[j];
        x[j] = temp;
      }
    }
    

    /**
     * Write the specified permutation of the first vector into
     * the second vector.  The second vector will be resized to
     * the size of the permutation.
     *
     * @tparam T Type of elements to permute
     * @param pi Permutation.
     * @param x_from Vector of elements to permute
     * @param x_to Vector into which permutation of elements is written
     * @throw std::invalid_argument If the permutation vector and
     * source vector from which to copy are not the same size.
     */
    template <typename T>
    void permute(const std::vector<size_t>& pi,
                 const std::vector<T>& x_from,
                 std::vector<T>& x_to) {
      size_t N = pi.size();
      if (N != x_from.size()) {
        std::stringstream msg;
        msg << "Require permutation to be same size as source vector."
            << "; found pi.size()=" << pi.size()
            << "; x_from.size()=" << x_from.size();
      }
      x_to.resize(N);
      for (size_t i = 0; i < N; ++i)
        x_to[i] = x_from[pi[i]];
    }

    
    /**
     * An <code>mcmc::chains</code> object stores parameter names and
     * dimensionalities along with samples from multiple chains.
     *
     * <p><b>Synchronization</b>: For arbitrary concurrent use, the
     * read and write methods need to be read/write locked.  Multiple
     * writers can be used concurrently if they write to different
     * chains.  Readers for single chains need only be read/write locked
     * with writers of that chain.  For reading across chains, full
     * read/write locking is required.  Thus methods will be classified
     * as global or single-chain read or write methods.
     *
     * <p><b>Storage Order</b>: Storage is column/last-index major.
     */
    template <typename RNG = boost::random::ecuyer1988>
    class chains {
    private:

      size_t _warmup;
      const std::vector<std::string> _names;
      const std::vector<std::vector<size_t> > _dimss;
      const size_t _num_params; // total
      const std::vector<size_t> _starts;
      const std::map<std::string,size_t> _name_to_index;
      // [chain,param,sample]
      std::vector<std::vector<std::vector<double > > > _samples; 
      std::vector<size_t> _permutation;
      RNG _rng; // defaults to time-based init

      static size_t calc_num_params(const std::vector<size_t>& dims) {
        size_t num_params = 1;
        for (size_t i = 0;  i < dims.size(); ++i)
          num_params *= dims[i];
        return num_params;
      }

      static size_t 
      calc_total_num_params(const std::vector<std::vector<size_t> >& dimss) {
        int num_params = 0;
        for (size_t i = 0; i < dimss.size(); ++i)
          num_params += calc_num_params(dimss[i]);
        return num_params;
      }

      static std::vector<size_t> 
      calc_starts(const std::vector<std::vector<size_t> >& dimss) {
        std::vector<size_t> starts(dimss.size());
        starts[0] = 0;
        for (size_t i = 1; i < dimss.size(); ++i)
          starts[i] = starts[i - 1] + calc_num_params(dimss[i - 1]);
        return starts;
      }

      static std::map<std::string,size_t>
      calc_name_to_index(const std::vector<std::string> names) {
        std::map<std::string,size_t> name_to_index;
        for (size_t i = 0; i < names.size(); ++i)
          name_to_index[names[i]] = i;
        return name_to_index;
      }

      inline void validate_param_name_idx(size_t j) {
        if (j < num_param_names()) 
          return;
        std::stringstream msg;
        msg << "parameter name index must be less than number of params"
            << "; found j=" << j;
        throw std::out_of_range(msg.str());
      }

      inline void validate_param_idx(size_t n) {
        if (n < num_params())
          return;
        std::stringstream msg;
        msg << "parameter index must be less than number of params"
            << "; found n=" << n;
        throw std::out_of_range(msg.str());
      }

      inline void validate_chain_idx(size_t k) {
        if (k >= num_chains()) {
          std::stringstream msg;
          msg << "chain must be less than number of chains."
              << "; num chains=" << num_chains()
              << "; chain=" << k;
          throw std::out_of_range(msg.str());
        }
      }

      // k = chain idx, m = iteration
      void validate_iteration(size_t k,    
                              size_t m) {  
        validate_chain_idx(k);
        if (m >= _samples[k][0].size()) {
          std::stringstream msg;
          msg << "require sample index below number of samples"
              << "; sample index m=" << m
              << "; chain index k=" << k
              << "; num samples in chain" << k << "=" 
              << _samples[k][0].size();
          throw std::out_of_range(msg.str());
        }
      }

      void resize_permutation(size_t K) {
        if (_permutation.size() != K) 
          permutation(_permutation,K,_rng);
      }

    public:

      /**
       * Construct a chains object with the specified number of Markov
       * chains, and the specified parameter names and matching
       * parameter dimensions.
       * 
       * <p>The order of the parameter names and dimesnions should match
       * the order in which samples are added to the constructed
       * object.  
       *
       * <p>The total number of parameters is determined by adding the
       * parameters for each name.  The number of parameters for each
       * name is determined by multiplying its dimensionalities.  For
       * example, a 2 x 3 x 4 matrix parameter produces of 24 total
       * parameters.
       *
       * @param num_chains Number of Markov chains.
       * @param names Sequence of paramter names.
       * @param dimss Sequence of parameter dimensionalities.
       * @throws std::invalid_argument If the name and dimensions
       * sequences are not the same size.
       */
      chains(const size_t num_chains,
             const std::vector<std::string>& names,
             const std::vector<std::vector<size_t> >& dimss) 
        : _warmup(0),
          // function call tests dimensionality match & returns names
          _names(names), // test_match_return_names(names,dimss)),
          _dimss(dimss),
          _num_params(calc_total_num_params(dimss)),
          _starts(calc_starts(dimss)),               // copy
          _name_to_index(calc_name_to_index(names)), // copy
          _samples(num_chains,std::vector<std::vector<double> >(_num_params))
      {  }
      
      /**
       * Return the number of chains.
       *
       * <p><b>Synchronization</b>: Thread safe.
       * 
       * @return The number of chains.
       */
      inline size_t num_chains() {
        return _samples.size();
      }

      /**
       * Return the total number of parameters.  
       *
       * <p>This is not the number of parameter names, but the total
       * number of scalar parameters.
       *
       * <p><b>Synchronization</b>: Thread safe.
       * 
       * @return The total number of parameters.
       */
      inline size_t num_params() { 
        return _num_params;
      }

      /**
       * Return the total number of parameter names.
       *
       * <p><b>Synchronization</b>: Thread safe.
       * 
       * @return The total number of parameter names.
       */
      inline size_t num_param_names() {
        return _names.size();
      }

      /**
       * Return the sequence of parameter names.
       *
       * <p><b>Synchronization</b>: Thread safe after construction.
       * 
       * @return The sequence of parameter names.
       */
      const std::vector<std::string>& param_names() {
        return _names;
      }

      /**
       * Return the name of the parameter with the specified index.
       *
       * <p><b>Synchronization</b>: Thread safe.
       * 
       * @param j Index of parameter.
       * @return The parameter with the specified index.
       * @throw std::out_of_range If the parameter identifier is
       * greater than or equal to the number of parameters.
       */
      const std::string& param_name(size_t j) {
        validate_param_name_idx(j);
        return _names[j];
      }

      /**
       * Return the sequence of named parameter dimensions.  
       *
       * <p><b>Synchronization</b>: Thread safe after construction.
       *
       * @return The sequence of named parameter dimensions.
       */
      const std::vector<std::vector<size_t> >& param_dimss() {
        return _dimss;
      }

      /**
       * Return the dimensions of the parameter name with the
       * specified index.
       *
       * <p><b>Synchronization</b>: Thread safe.
       *
       * @param j Index of a parameter name.
       * @return The dimensions of the parameter name with the specified
       * index.
       * @throw std::out_of_range If the index is greater than or equal
       * to the number of parameter names.
       */
      const std::vector<size_t>& param_dims(size_t j) {
        validate_param_name_idx(j);
        return _dimss[j];
      }

      /**
       * Return the sequence of starting indexes for the named
       * parameters in the underlying sequence of scalar parameters.
       *
       * <p><b>Synchronization</b>: Thread safe.
       *
       * @return The sequence of named parameter start indexes.
       */
      const std::vector<size_t>& param_starts() {
        return _starts;
      }

      /**
       * Return the starting position of the named parameter with the
       * specified index in the underlying sequence of scalar parameters.
       *
       * <p><b>Synchronization</b>: Thread safe.
       *
       * @param j The parameter name index.
       * @return The start index of the specified parameter.
       * @throw std::out_of_range If the parameter name index is
       * greater than or equal to the number of named parameters.
       */
      size_t param_start(size_t j) {
        validate_param_name_idx(j);
        return _starts[j];
      }

      /**
       * Return a copy of the sequence of named parameter sizes.  The
       * size of a named parameter is the prouct of its dimensions.
       *
       * <p><b>Synchronization</b>: Thread safe.
       *
       * @return The sequence of named parameter sizes.
       */
      const std::vector<size_t> param_sizes() {
        std::vector<size_t> s(num_param_names());
        for (unsigned int j = 0; j < num_param_names(); ++j)
          s[j] = param_size(j); // could optimize tests in param_sizes() out
        return s;
      }

      /**
       * Return the size of the named parameter with the specified index.
       * The size of a named parameter is the prouct of its dimensions.
       *
       * <p><b>Synchronization</b>: Thread safe after construction.
       *
       * @param j The index of a named parameter.
       * @return The size of the specified named parameter.
       * @throw std::out_of_range If the index is greater than or
       * equal to the number of named parameters.
       */
      size_t param_size(size_t j) {
        validate_param_name_idx(j);
        if (j + 1 < _starts.size()) 
          return _starts[j+1] - _starts[j];
        return num_params() - _starts[j];
      }

      /**
       * Return the named parameter index for the specified parameter
       * name.
       *
       * <p><b>Synchronization</b>: Thread safe.
       *
       * @param name Parameter name.
       * @return Index of parameter name.
       * @throw std::out_of_range If the parameter is not one of the
       * named parameters.
       */
      size_t param_name_to_index(const std::string& name) {
        std::map<std::string,size_t>::const_iterator it
          = _name_to_index.find(name);
        if (it == _name_to_index.end()) {
          std::stringstream ss;
          ss << "unknown parameter name=" << name;
          throw std::out_of_range(ss.str());
        }
        return it->second;
      }
      

      /**
       * Return the index in the underlying sequence of scalar parameters
       * for the parameter with the specified name index and
       * indexes.
       *
       * <p><b>Synchronization</b>: Thread safe.
       *
       * @param j Index of parameter name.
       * @param idxs Indexes into parameter.
       * @return Offset into the underlying sequence of scalar paramters.
       * @throw std::out_of_range If the named parameter index is greater than
       * or equal to the number of named parameters or if any of the indexes
       * is out of range for the named parameter with the specified index.
       */ 
      size_t get_total_param_index(size_t j, // param id
                                   const std::vector<size_t>& idxs) {
        return get_offset(param_dims(j),idxs)
          + param_start(j);
      }


      /**
       * Set the warmup cutoff to the specified number of
       * iterations.  The first samples in each chain up to
       * this number will be treated as warmup samples.
       * 
       * <p><b>Synchronization</b>: Warmup write method. 
       *
       * @param warmup_iterations Number of warmup iterations.
       */
      void set_warmup(size_t warmup_iterations) {
        _warmup = warmup_iterations;
      }


      /**
       * Return the warmup iteration cutoff.
       * 
       * <p><b>Synchronization</b>: Warmup read method.
       *
       * @return Number of warmup iterations.
       */
      inline size_t warmup() {
        return _warmup;
      }


      /**
       * Add the specified sample to the end of the specified chain.
       *
       * <p><b>Synchronization:</b> Chain-specific write.
       *
       * @param chain Markov chain identifier.
       * @param theta Parameter values.
       * @throws std::invalid_argument if the size of the sample
       * vector does not match the number of parameters.
       */
      void add(size_t chain,
               std::vector<double> theta) {
        validate_chain_idx(chain);
        if (theta.size() != _num_params) {
          std::stringstream msg;
          msg << "parameter vector size must match num params"
              << "; num params=" << _num_params
              << "; theta.size()=" << theta.size();
          throw std::invalid_argument(msg.str());
        }
        for (size_t i = 0; i < theta.size(); ++i)
          _samples[chain][i].push_back(theta[i]); // _samples very non-local
      }
      
      



      /**
       * Return the number of warmup samples in the
       * specified chain.
       *
       * <p><b>Synchronization</b>:  Warmup and chain-specific read.
       *
       * @param k Chain index.
       * @return Number of warmup samples availabe in chain.
       */
      size_t num_warmup_samples(size_t k) {
        return std::min<size_t>(num_samples(k), warmup());
      }

      /**
       * Return the total number of warmup samples across chains.
       *
       * <p><b>Synchronization</b>:  Warmup and cross-chain read.
       *
       * @return The total number of warmup samples.
       */
      size_t num_warmup_samples() {
        size_t total = 0;
        for (size_t k = 0; k < num_chains(); ++k)
          total += num_warmup_samples(k);
        return total;
      }

      /**
       * Return the number of samples in the specified chain not
       * including warmup samples.
       *
       * <p><b>Synchronization</b>:  Warmup and chain-specific read.
       *
       * @param k Chain index.
       * @return Number of warmup samples availabe in chain.
       */
      size_t num_kept_samples(size_t k) {
        if (num_samples(k) > warmup())
          return num_samples(k) - warmup();
        return 0U;
      }


      /**
       * Return the total number of samples in all chains not
       * including warmup samples.
       *
       * <p><b>Synchronization</b>:  Warmup and cross-chain read.
       *
       * @return Total number of warmup samples
       */
      size_t num_kept_samples() {
        size_t total = 0;
        for (size_t k = 0; k < num_chains(); ++k)
          total += num_kept_samples(k);
        return total;
      }

      /**
       * Return the total number of samples across chains including
       * warmup and kept samples.
       *
       * <p><b>Synchronization</b>: Cross-chain read.
       *
       * @return Total number of samples.
       */
      size_t num_samples() {
        size_t M = 0;
        for (size_t k = 0; k < num_chains(); ++k)
          M += num_samples(k);
        return M;
      }

      /**
       * Return the number of samples including warmup and kept samples
       * in the specified chain.
       *
       * <p><b>Synchronization</b>: Chain-specific read.
       *
       * @param k Markov chain index.
       * @return Number of samples in the specified chain.
       * @throw std::out_of_range If the identifier is greater than
       * or equal to the number of chains.
       */
      size_t num_samples(size_t k) {
        validate_chain_idx(k);
        return _samples[k][0].size();
      }


      /**
       * Write into the specified vector the warmup and kept samples
       * for the scalar parameter with the specified index.  The order
       * of samples is by chain, then by order in which the sample was
       * added to the chain.
       *
       * <p><b>Synchronization</b>: Cross-chain read.
       * 
       * @param n Index of parameter.
       * @param samples Vector into which samples are written.
       * @throw std::out_of_range If the parameter index is greater
       * than or equal to the total number of scalar parameters.
       */
      void
      get_samples(size_t n,
                  std::vector<double>& samples) {
        validate_param_idx(n);
        samples.resize(0);
        samples.reserve(num_samples());
        for (size_t k = 0; k < num_chains(); ++k)
          samples.insert(samples.end(),
                         _samples[k][n].begin(), 
                         _samples[k][n].end());
      }

      /**
       * Write into the specified vector the warmup and kept samples
       * for the scalar parameter with the specified index in the
       * chain with the specified index.  The order of samples is the
       * order in which they were added.
       *
       * <p><b>Synchronization</b>: Chain-specific read.
       *
       * @param k Index of chain.
       * @param n Index of parameter.
       * @param samples Vector into which to write samples
       * @throw std::out_of_range If the specified chain index is greater
       * than or equal to the number of chains, or if the specified parameter
       * index is greater than or equal to the total number of parameters.
       */
      void get_samples(size_t k, 
                       size_t n, 
                       std::vector<double>& samples) {  
        validate_chain_idx(k);
        validate_param_idx(n); 
        samples.resize(0);
        samples.reserve(num_samples(k));
        samples.insert(samples.end(),
                       _samples[k][n].begin(),
                       _samples[k][n].end());
      }


      /**
       * Write into the specified vector the kept samples for the
       * scalar parameter with the specified index.  The order of
       * samples is permuted, but as long as no samples have been
       * added in the interim, subsequent calls to this method will
       * use the same permutation for all parameter indexes.
       *
       * <p><b>Synchronization</b>: Cross-chain read.
       * 
       * @param n Index of parameter.
       * @param samples Vector into which samples are written.
       * @throw std::out_of_range If the parameter index is greater
       * than or equal to the total number of scalar parameters.
       */
      void
      get_kept_samples_permuted(size_t n,
                                std::vector<double>& samples) {
        validate_param_idx(n);
        size_t M = num_kept_samples();
        samples.resize(M);
        resize_permutation(M);
        // const std::vector<size_t>& permutation = _permutation;
        size_t pos = 0;
        for (size_t k = 0; k < num_chains(); ++k) {
          // const std::vector<double>& samples_k_n = _samples[k][n];
          for (size_t m = warmup(); m < num_samples(k); ++m) {
            samples[_permutation[pos]] = _samples[k][n][m]; // _samples_k_n[m];
            ++pos;
          }
        }
      }

      /**
       * Apply the specified functor to each kept sample for the
       * specified parameter in the specified chain.  The samples are
       * visited in the order they were added.
       *
       * @tparam F Type of functor to apply
       * @param k Chain index
       * @param n Parameter index
       * @param f Functor to apply to kept samples
       */
      template <typename F>
      void
      apply_kept_samples(size_t k,
                         size_t n,
                         F& f) {
        using std::vector;
        for (vector<double>::const_iterator it = _samples[k][n].begin() + warmup();
             it != _samples[k][n].end();
             ++it)
          f(*it);
      }

      /**
       * Apply the specified functor to each kept sample for the
       * specified parameter across all chains.  The samples are
       * visited in the order of chain index, and within a chain, in
       * the order they were added.
       *
       * @tparam F Type of functor to apply
       * @param n Parameter index
       * @param f Functor to apply to kept samples
       */
      template <typename F>
      void
      apply_kept_samples(size_t n,
                         F& f) {
        for (size_t k = 0; k < num_chains(); ++k)
          apply_kept_samples(k,n,f);
      }

      /**
       * Write into the specified vector the kept samples for the
       * scalar parameter with the specified index in the chain with
       * the specified index.  The order of samples is the order in
       * which they were added.
       *
       * <p><b>Synchronization</b>: Chain-specific read.
       *
       * @param k Index of chain.
       * @param n Index of parameter.
       * @param samples Vector into which to write samples
       * @throw std::out_of_range If the specified chain index is greater
       * than or equal to the number of chains, or if the specified parameter
       * index is greater than or equal to the total number of parameters.
       */
      void
      get_kept_samples(size_t k,
                       size_t n,
                       std::vector<double>& samples) {
        validate_param_idx(n);
        samples.resize(0);
        samples.reserve(num_kept_samples(k));
        samples.insert(samples.end(),
                       _samples[k][n].begin() + warmup(),
                       _samples[k][n].end());
      }



      /**
       * Write into the specified vector the warmup samples for the
       * scalar parameter with the specified index.  The order of
       * samples is by chain, then by order in which the sample was
       * added to the chain.
       *
       * <p><b>Synchronization</b>: Cross-chain read.
       * 
       * @param n Index of parameter.
       * @param samples Vector into which samples are written.
       * @throw std::out_of_range If the parameter index is greater
       * than or equal to the total number of scalar parameters.
       */
      void
      get_warmup_samples(size_t n,
                         std::vector<double>& samples) {
        validate_param_idx(n);
        samples.resize(0);
        samples.reserve(num_warmup_samples());
        for (size_t k = 0; k < num_chains(); ++k) {
          if (num_warmup_samples(k) < warmup())
            samples.insert(samples.end(),
                           _samples[k][n].begin(),
                           _samples[k][n].end());
          else
            samples.insert(samples.end(),
                           _samples[k][n].begin(),
                           _samples[k][n].begin() + warmup());
        }
      }

      /**
       * Write into the specified vector the warmup samples for the
       * parameter with the specified index in the chain with the
       * specified index.  The order of samples is the order in which
       * they were added.
       *
       * <p><b>Synchronization</b>: Chain-specific read.
       *
       * @param k Index of chain.
       * @param n Index of parameter.
       * @param samples Vector into which to write samples
       * @throw std::out_of_range If the specified chain index is greater
       * than or equal to the number of chains, or if the specified parameter
       * index is greater than or equal to the total number of parameters.
       */
      void
      get_warmup_samples(size_t k,
                         size_t n,
                         std::vector<double>& samples) {
        validate_param_idx(n);
        samples.resize(0);
        samples.reserve(num_warmup_samples(k));
        if (num_warmup_samples(k) < warmup())
            samples.insert(samples.end(),
                           _samples[k][n].begin(),
                           _samples[k][n].end());
        else
            samples.insert(samples.end(),
                           _samples[k][n].begin(),
                           _samples[k][n].begin() + warmup());
      }


      /**
       * Return the sample mean of the kept samples in the
       * specified chain for the specified parameter.
       *
       * @param k Chain index.
       * @param n Parameter index.
       * @return Sample mean of parameter in chain.
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or the parameter index is
       * greater than or equal to the number of parameters.
       */
      double mean(size_t k,
                  size_t n) {
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::mean;
        validate_chain_idx(k);
        validate_param_idx(n);
        accumulator_set<double, stats<mean> > acc;
        apply_kept_samples(k,n,acc);
        return boost::accumulators::mean(acc);
      }

      /**
       * Return the sample mean of the kept samples in all
       * chains for the specified parameter.
       *
       * @param n Parameter index.
       * @throw std::out_of_range If the parameter index is
       * greater than or equal to the number of parameters.
       */
      double mean(size_t n) {
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::mean;
        validate_param_idx(n);
        accumulator_set<double, stats<mean> > acc;
        apply_kept_samples(n,acc);
        return boost::accumulators::mean(acc);
      }

      /**
       * Return the sample standard deviation of the kept samples in
       * the specified chain for the specified parameter.  This method
       * uses the unbiased variance estimator (and thus divides by M-1
       * rather than M in the denominator)
       *
       * @param k Chain index.
       * @param n Parameter index.
       * @return Sample mean of parameter in chain.
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or the parameter index is
       * greater than or equal to the number of parameters.
       */
      double sd(size_t k,
                size_t n) {
        validate_chain_idx(k);
        validate_param_idx(n);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        accumulator_set<double, stats<variance> > acc;
        apply_kept_samples(k,n,acc);
        double M = num_kept_samples(k);
        return std::sqrt((M / (M-1)) * boost::accumulators::variance(acc));
      }

      /**
       * Return the sample standard deviation of the kept samples in
       * all chains for the specified parameter.  This method divides
       * by the number of kept samples minus 1 (and is htus based on
       * an unbiased variance estimate from the samples).
       *
       * @param n Parameter index.
       * @return Sample standard deviation of kept samples for
       * parameter.
       * @throw std::out_of_range If the parameter index is
       * greater than or equal to the number of parameters.
       */
      double sd(size_t n) {
        validate_param_idx(n);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        accumulator_set<double, stats<variance> > acc;
        apply_kept_samples(n,acc);
        double M = num_kept_samples();
        return std::sqrt((M / (M-1)) * boost::accumulators::variance(acc));
      }

      /**
       * Return the variance of the kept samples in
       * the specified chain for the specified parameter.  
       *
       * @param k Chain index.
       * @param n Parameter index.
       * @return Variance of parameter in chain.
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or the parameter index is
       * greater than or equal to the number of parameters.
       */
      double variance(size_t k,
                      size_t n) {
        validate_chain_idx(k);
        validate_param_idx(n);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        accumulator_set<double, stats<variance> > acc;
        apply_kept_samples(k,n,acc);
        double M = num_kept_samples(k);
        return (M / (M-1)) *boost::accumulators::variance(acc);
      }

      /**
       * Return the variance of the kept samples in
       * all chains for the specified parameter.  
       *
       * @param n Parameter index.
       * @return Variance of kept samples for
       * parameter.
       * @throw std::out_of_range If the parameter index is
       * greater than or equal to the number of parameters.
       */
      double variance(size_t n) {
        validate_param_idx(n);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        accumulator_set<double, stats<variance> > acc;
        apply_kept_samples(n,acc);
        double M = num_kept_samples(n);
        return (M / (M-1)) *boost::accumulators::variance(acc);
      }

      /**
       * Return the covariance of the kept samples in
       * the specified chain for the specified parameters.  
       *
       * @param k Chain index.
       * @param n1 Parameter index 1.
       * @param n2 Parameter index 2.
       * @return Covariance of parameters in chain.
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or the parameter index is
       * greater than or equal to the number of parameters.
       */
      double covariance(size_t k,
                        size_t n1,
                        size_t n2) {
        validate_chain_idx(k);
        validate_param_idx(n1);
        validate_param_idx(n2);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        using boost::accumulators::tag::covariance;
        using boost::accumulators::tag::covariate1;
        
        accumulator_set<double, stats<covariance<double, covariate1> > > acc;
        std::vector<double> samples1, samples2;
        this->get_kept_samples(k, n1, samples1);
        this->get_kept_samples(k, n2, samples2);
        for (size_t kk = 0; kk < this->num_kept_samples(k); kk++) {
          acc(samples1[kk], boost::accumulators::covariate1 = samples2[kk]);
        }
        double M = num_kept_samples(k);
        return (M / (M-1)) * boost::accumulators::covariance(acc);
      }

      /**
       * Return the covariance of the kept samples for
       * the specified parameters.  
       *
       * @param n1 Parameter index 1.
       * @param n2 Parameter index 2.
       * @return Covariance of kept samples for
       * parameter.
       * @throw std::out_of_range If the parameter index is
       * greater than or equal to the number of parameters.
       */
      double covariance(size_t n1, size_t n2) {
        validate_param_idx(n1);
        validate_param_idx(n2);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::stats;
        using boost::accumulators::tag::variance;
        using boost::accumulators::tag::covariance;
        using boost::accumulators::tag::covariate1;
        
        accumulator_set<double, stats<covariance<double, covariate1> > > acc;
        for (size_t chain = 0; chain < this->num_chains(); chain++) {
          std::vector<double> samples1, samples2;
          this->get_kept_samples(chain, n1, samples1);
          this->get_kept_samples(chain, n2, samples2);
          for (size_t kk = 0; kk < this->num_kept_samples(chain); kk++) {
            acc(samples1[kk], boost::accumulators::covariate1 = samples2[kk]);
          }
        }
        double M = this->num_kept_samples();
        return (M / (M-1)) * boost::accumulators::covariance(acc);
      }
      

      /**
       * Return the correlation of the kept samples in
       * the specified chain for the specified parameters.  
       *
       * @param k Chain index.
       * @param n1 Parameter index 1.
       * @param n2 Parameter index 2.
       * @return Correlation of parameters in chain.
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or the parameter index is
       * greater than or equal to the number of parameters.
       */
      double correlation(size_t k,
                        size_t n1,
                        size_t n2) {
        double cov = covariance(k, n1, n2);
        double sd1 = sd(k, n1);
        double sd2 = sd(k, n2);
                
        return cov / sd1 / sd2;
      }

      /**
       * Return the correlation of the kept samples for
       * the specified parameters.  
       *
       * @param n1 Parameter index 1.
       * @param n2 Parameter index 2.
       * @return Correlation of kept samples for
       * parameter.
       * @throw std::out_of_range If the parameter index is
       * greater than or equal to the number of parameters.
       */
      double correlation(size_t n1, size_t n2) {
        double cov = covariance(n1, n2);
        double sd1 = sd(n1);
        double sd2 = sd(n2);

        return cov / sd1 / sd2;
      }
      

      /**
       * Return the specified sample quantile for kept samples for the
       * specified parameter in the specified chain.
       *
       * @param k Chain index
       * @param n Parameter index
       * @param prob Quantile probability
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or if the parameter index is
       * greater than the number of parameters
       * @throw std::invalid_argument If the probabilty is not between
       * 0 and 1 inclusive.
       */
      double quantile(size_t k,
                      size_t n,
                      double prob) {
        validate_chain_idx(k);
        validate_param_idx(n);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::left;
        using boost::accumulators::quantile_probability;
        using boost::accumulators::right;
        using boost::accumulators::stats;
        using boost::accumulators::tag::tail;
        using boost::accumulators::tag::tail_quantile;
        double size = num_kept_samples(k);
        if (prob < 0.5) {
          std::size_t cs(2 + prob * size); // 2+ for interpolation
          accumulator_set<double, stats<tail_quantile<left> > > 
            acc(tail<left>::cache_size = cs);
          apply_kept_samples(k,n,acc);
          return boost::accumulators::quantile(acc, quantile_probability = prob);
        } else {
          std::size_t cs(2 + (1.0 - prob) * size);
          accumulator_set<double, stats<tail_quantile<right> > > 
            acc(tail<right>::cache_size = cs);
          apply_kept_samples(k,n,acc);
          return boost::accumulators::quantile(acc, quantile_probability = prob);
        }
      }

      /**
       * Return the specified sample quantile for kept samples for the
       * specified parameter across all chains.
       *
       * @param n Parameter index
       * @param prob Quantile probability
       * @throw std::out_of_range If the parameter index is
       * greater than the number of parameters
       * @throw std::invalid_argument If the probabilty is not between
       * 0 and 1 inclusive.
       */
      double quantile(size_t n,
                      double prob) {
        validate_param_idx(n);
        using boost::accumulators::accumulator_set;
        using boost::accumulators::left;
        using boost::accumulators::quantile_probability;
        using boost::accumulators::right;
        using boost::accumulators::stats;
        using boost::accumulators::tag::tail;
        using boost::accumulators::tag::tail_quantile;
        double size = num_kept_samples();
        if (prob < 0.5) {
          std::size_t cs(2 + prob * size); // 2+ for interpolation
          accumulator_set<double, stats<tail_quantile<left> > > 
            acc(tail<left>::cache_size = cs);
          apply_kept_samples(n,acc);
          return boost::accumulators::quantile(acc, quantile_probability = prob);
        } else {
          std::size_t cs(2 + (1.0 - prob) * size);
          accumulator_set<double, stats<tail_quantile<right> > > 
            acc(tail<right>::cache_size = cs);
          apply_kept_samples(n,acc);
          return boost::accumulators::quantile(acc, quantile_probability = prob);
        }
      }

      /**
       * Write the specified sample quantiles into the specified
       * vector for the kept samples for the specified parameter
       * in the specified chain.
       *
       * @param k Chain index
       * @param n Parameter index
       * @param probs Quantile probabilities
       * @param quantiles Vector into which to write sample quantiles
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or if the parameter index is
       * greater than the number of parameters
       * @throw std::invalid_argument If the any quantile probabilty is 
       * not between 0 and 1 inclusive.
       */
      void quantiles(size_t k,
                     size_t n,
                     const std::vector<double>& probs,
                     std::vector<double>& quantiles) {
        validate_chain_idx(k);
        validate_param_idx(n);
        for (size_t i = 0; i < probs.size(); ++i)
          validate_prob(probs[i]);
        quantiles.resize(probs.size());

        using boost::accumulators::accumulator_set;
        using boost::accumulators::quantile;
        using boost::accumulators::quantile_probability;
        using boost::accumulators::right;
        using boost::accumulators::stats;
        using boost::accumulators::tag::tail;
        using boost::accumulators::tag::tail_quantile;
        // keeps all (more efficient to have two for (min,0.5) and (0.5,max))
        accumulator_set<double, stats<tail_quantile<right> > >
          acc(tail<right>::cache_size = num_kept_samples(k));
        apply_kept_samples(k,n,acc);
        
        for (size_t i = 0; i < probs.size(); ++i)
          quantiles[i] = quantile(acc, quantile_probability = probs[i]);
      }

      /**
       * Write the specified sample quantiles into the specified
       * vector for the kept samples for the specified parameter
       * across all chains.
       *
       * @param n Parameter index
       * @param probs Quantile probabilities
       * @param quantiles Vector into which to write sample quantiles
       * @throw std::out_of_range If the parameter index is greater
       * than the number of parameters
       * @throw std::invalid_argument If the any quantile probabilty is 
       * not between 0 and 1 inclusive.
       */
      void quantiles(size_t n,
                     const std::vector<double>& probs,
                     std::vector<double>& quantiles) {
        validate_param_idx(n);
        for (size_t i = 0; i < probs.size(); ++i)
          validate_prob(probs[i]);
        quantiles.resize(probs.size());

        using boost::accumulators::accumulator_set;
        using boost::accumulators::quantile;
        using boost::accumulators::quantile_probability;
        using boost::accumulators::right;
        using boost::accumulators::stats;
        using boost::accumulators::tag::tail;
        using boost::accumulators::tag::tail_quantile;
        accumulator_set<double, stats<tail_quantile<right> > >
          acc(tail<right>::cache_size = num_kept_samples());
        apply_kept_samples(n,acc);
        for (size_t i = 0; i < probs.size(); ++i)
          quantiles[i] = quantile(acc, quantile_probability = probs[i]);
      }

      /**
       * Return the specified sample central interval for the specified
       * parameter in the kept samples in the specified chain.  
       *
       * <p>The central interval of width p is defined to be (ql,qh)
       * where ql is the (1-p)/2 quantile and qh is the 1 - ql
       * quantile.
       *
       * @param k Chain index.
       * @param n Parameter index.
       * @param prob Width of central interval.
       * @return Pair consisting of low and high quantile.
       * @throw std::out_of_range If the chain index is greater than
       * or equal to the number of chains or if the parameter index is
       * greater than the number of parameters
       * @throw std::invalid_argument If the the interval width is not
       * between 0 and 1 inclusive.
       */
      std::pair<double,double>
      central_interval(size_t k,
                       size_t n,
                       double prob) {
        // validate k,n in calls to quantile
        validate_prob(prob);
        double low_prob = (1.0 - prob) / 2.0;
        double high_prob = 1.0 - low_prob;
        double low_quantile = quantile(k,n,low_prob);
        double high_quantile = quantile(k,n,high_prob);
        return std::pair<double,double>(low_quantile,high_quantile);
      }

      /**
       * Return the specified central interval for the specified
       * parameter in the kept samples of all chains.
       *
       * <p>The central interval of width p is defined to be (ql,qh)
       * where ql is the (1 - p) / 2 quantile and qh is the 1 - ql
       * quantile.
       *
       * @param n Parameter index.
       * @param prob Width of central interval.
       * @return Pair consisting of low and high quantile.
       * @throw std::out_of_range If the parameter index is
       * greater than the number of parameters
       * @throw std::invalid_argument If the the interval width is not
       * between 0 and 1 inclusive.
       */
      std::pair<double,double>
      central_interval(size_t n,
                       double prob) {
        // validate n in calls to quantile
        validate_prob(prob);
        double low_prob = (1.0 - prob) / 2.0;
        double high_prob = 1.0 - low_prob;
        double low_quantile = quantile(n,low_prob);
        double high_quantile = quantile(n,high_prob);
        return std::pair<double,double>(low_quantile,high_quantile);
      }

      /** 
       * Returns the autocorrelations for the specified parameter in the
       * kept samples of the chain specified.
       * 
       * @param[in] k Chain index
       * @param[in] n Parameter index
       * @param[out] ac Autocorrelations
       */
      void autocorrelation(const size_t k, const size_t n, 
                           std::vector<double>& ac) {
        std::vector<double> samples;
        get_kept_samples(k,n,samples);
        stan::prob::autocorrelation(samples,
                                    ac);
      }
      
      /** 
       * Returns the autocovariance for the specified parameter in the
       * kept samples of the chain specified.
       * 
       * @param[in] k Chain index
       * @param[in] n Parameter index
       * @param[out] acov Autocovariances
       */
      void autocovariance(const size_t k, const size_t n, 
                           std::vector<double>& acov) {
        std::vector<double> samples;
        get_kept_samples(k,n,samples);
        stan::prob::autocovariance(samples,
                                   acov);
      }
      
      /** 
       * Returns the effective sample size for the specified parameter
       * across all kept samples.
       *
       * The implementation matches BDA3's effective size description.
       * 
       * Current implementation takes the minimum number of samples
       * across chains as the number of samples per chain.
       *
       * @param[in] n Parameter index
       * 
       * @return the effective sample size.
       */
      // FIXME: reimplement using autocorrelation.
      double effective_sample_size(size_t n) {
        validate_param_idx(n);
        size_t m = this->num_chains();
        // need to generalize to each jagged samples per chain
        size_t n_samples = this->num_kept_samples(0U);
        for (size_t chain = 1; chain < m; chain++) {
          n_samples = std::min(n_samples, this->num_kept_samples(chain));
        }

        using std::vector;
        vector< vector<double> > acov;
        for (size_t chain = 0; chain < m; chain++) {
          vector<double> acov_chain;
          autocovariance(chain, n, acov_chain);
          acov.push_back(acov_chain);
        }
        
        vector<double> chain_mean;
        vector<double> chain_var;
        for (size_t chain = 0; chain < m; chain++) {
          double n_kept_samples = num_kept_samples(chain);
          chain_mean.push_back(this->mean(chain,n));
          chain_var.push_back(acov[chain][0]*n_kept_samples/(n_kept_samples-1));
        }
        double mean_var = stan::math::mean(chain_var);
        double var_plus = mean_var*(n_samples-1)/n_samples;
        if (m > 1) var_plus += stan::math::variance(chain_mean);
        vector<double> rho_hat_t;
        double rho_hat = 0;
        for (size_t t = 1; (t < n_samples && rho_hat >= 0); t++) {
          vector<double> acov_t(m);
          for (size_t chain = 0; chain < m; chain++) {
            acov_t[chain] = acov[chain][t];
          }
          rho_hat = 1 - (mean_var - stan::math::mean(acov_t)) / var_plus;
          if (rho_hat >= 0)
            rho_hat_t.push_back(rho_hat);
        }
        
        double ess = m*n_samples;
        if (rho_hat_t.size() > 0) {
          ess /= 1 + 2 * stan::math::sum(rho_hat_t);
        }
        return ess;
      }
      
      /** 
       * Return the split potential scale reduction (split R hat)
       * for the specified parameter.
       *
       * Current implementation takes the minimum number of samples
       * across chains as the number of samples per chain.
       * 
       * @param[in] n Parameter index
       * 
       * @return split R hat.
       */
      double split_potential_scale_reduction(size_t n) {
        size_t n_chains = this->num_chains();
        size_t n_samples = this->num_kept_samples(0U);
        for (size_t chain = 1; chain < n_chains; chain++) {
          n_samples = std::min(n_samples, this->num_kept_samples(chain));
        }
        if (n_samples % 2 == 1)
          n_samples--;
        
        std::vector<double> split_chain_mean;
        std::vector<double> split_chain_var;

        for (size_t chain = 0; chain < n_chains; chain++) {
          std::vector<double> samples(n_samples);
          this->get_kept_samples(chain, n, samples);
          
          std::vector<double> split_chain(n_samples/2);
          split_chain.assign(samples.begin(),
                             samples.begin()+n_samples/2);
          split_chain_mean.push_back(stan::math::mean(split_chain));
          split_chain_var.push_back(stan::math::variance(split_chain));
          
          split_chain.assign(samples.end()-n_samples/2, 
                             samples.end());
          split_chain_mean.push_back(stan::math::mean(split_chain));
          split_chain_var.push_back(stan::math::variance(split_chain));
        }

        double var_between = n_samples/2 * stan::math::variance(split_chain_mean);
        double var_within = stan::math::mean(split_chain_var);
        
        // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
        return sqrt((var_between/var_within + n_samples/2 -1)/(n_samples/2));
      }

    };

    
    namespace {
      /** 
       * Returns the header from a csv output file
       * 
       * @param file csv output file
       * 
       * @return csv header
       */
      std::string read_header(std::fstream& file) {
        std::string header = "";
        // strip comments
        while (file.peek() == '#') {
          file.ignore(10000, '\n');
        }
        std::getline(file, header, '\n');
        return header;
      }
      
      /** 
       * Tokenize line by delimiter.
       * 
       * @param[in]  stream String to tokenize.
       * @param[in]  delimiter Delimiter.
       * @param[out] tokens Output vector of tokens.
       */
      void
      tokenize(const std::string& line, const char delimiter, 
               std::vector<std::string>& tokens) {
        tokens.clear();
        std::stringstream stream(line);
        std::string token;
        while (std::getline(stream,token,delimiter)) {
          tokens.push_back(token);
        }
      }

      /** 
       * Get names from a list of tokens.
       * 
       * @param[in] tokens a vector of tokens from the header.
       * @param[in] skip number of variables to skip
       * @param[out] names names of the variables in the tokens.
       */
      void
      get_names(const std::vector<std::string>& tokens, 
                const size_t skip,
                std::vector<std::string>& names) {
        names.clear();
        for (size_t i = 0; i < tokens.size(); i++) {
          std::stringstream token(tokens[i]);
          std::string name;
          std::getline(token,name,'.');
          if (names.size() == 0 || names.back()!=name) {
            names.push_back(name);
          }
        }
        names.erase(names.begin(), names.begin()+skip);
      }

      /** 
       * Get dimensions of variables from a list of tokens.
       * 
       * @param[in] tokens a vector of tokens from the header.
       * @param[in] number number of variables to skip
       * @param[out] dimss a vector of dims
       */
      void
      get_dimss(const std::vector<std::string>& tokens, 
                const size_t skip, 
                std::vector<std::vector<size_t> >& dimss) {
        dimss.clear();
        std::string last_name;
        std::vector<std::string> split;
        for (int i = tokens.size()-1; i >= 0; --i) {
          tokenize(tokens[i], '.', split);
          
          std::vector<size_t> dims;
          if (split.size() == 1) {
            dims.push_back(1);
            dimss.insert(dimss.begin(), dims);
          } else if (split.front() != last_name) {
            for (size_t j = 1; j < split.size(); j++) {
              dims.push_back((size_t)atol(split[j].c_str()));
            }
            dimss.insert(dimss.begin(), dims);
          }
          last_name = split.front();
        }
        dimss.erase(dimss.begin(), dimss.begin()+skip);
      }
      
      /** 
       * Reads values from a csv file. Reads the last variables in
       * each file.
       * 
       * @param[in,out] file csv output file.
       * @param[in] num_values number of values to read per line
       * @param[out] values Values from csv file. Order has not been altered.
       */
      void
      read_values(std::fstream& file, const size_t num_values,
                  std::vector<std::vector<double> >& thetas) {
        thetas.clear();
        read_header(file); // ignore header
        std::vector<double> theta;
        std::string line;
        std::vector<std::string> tokens;
        while (file.peek() != std::istream::traits_type::eof()) {
          while (file.peek() == '#') { // ignore comments
            file.ignore(10000, '\n');
          }
          std::getline(file, line, '\n');
          tokenize(line, ',', tokens);
          theta.clear();
          for (size_t i = tokens.size()-num_values; i < tokens.size(); i++) {
            theta.push_back(atof(tokens[i].c_str()));
          }
          if (theta.size() > 0) {
            thetas.push_back(theta);
          }
        }
      }
      
      /** 
       * Reorders the values in thetas. Each vector has the elements in the
       * index in from placed in the location to.
       * 
       * @param[in,out] thetas Values to reorder
       * @param[in] from Indexes of the elements to move.
       * @param[in] to Indexes of the locations to move.
       */
      void
      reorder_values(std::vector<std::vector<double> >& thetas,
                     const std::vector<size_t>& from,
                     const std::vector<size_t>& to) {
        std::vector<double> temp(from.size());
        for (size_t ii = 0; ii < thetas.size(); ii++) {
          for (size_t jj = 0; jj < from.size(); jj++) {
            temp[jj] = thetas[ii][from[jj]];
          }
          for (size_t jj = 0; jj < to.size(); jj++) {
            thetas[ii][to[jj]] = temp[jj];
          }
        }
      }

      /** 
       * Calculates the reordering necessary to change variables
       * from row major order to column major order.
       * 
       * @param[in]  dimss The dimension sizes of the variables
       * @param[out] from Index locations of where to move from (row-major)
       * @param[out] to Index locations of where to move to (col-major)
       */
      void
      get_reordering(const std::vector<std::vector<size_t> >& dimss,
                     std::vector<size_t>& from,
                     std::vector<size_t>& to) {
        from.clear();
        to.clear();
        
        size_t offset = 0;
        for (size_t ii = 0; ii < dimss.size(); ii++) {
          size_t curr_size = dimss[ii][0];
          if (dimss[ii].size() > 1) {
            for (size_t jj = 1; jj < dimss[ii].size(); jj++)
              curr_size *= dimss[ii][jj];
            
            std::vector<size_t> idxs;
            for (size_t jj = 0; jj < dimss[ii].size(); jj++) {
              idxs.push_back(0);
            }
            size_t from_index = 0;
            for (size_t to_index = 0; to_index < curr_size; to_index++) {
              from_index = 0;
              size_t offset_temp = 1;
              for (size_t kk = idxs.size(); kk > 0; --kk) {
                from_index += idxs[kk-1] * offset_temp;
                offset_temp *= dimss[ii][kk-1];
              }
              if (from_index != to_index) {
                from.push_back(offset+from_index);
                to.push_back(offset+to_index);
              }
              increment_indexes(dimss[ii], idxs);
            }
          }
          offset += curr_size;
        }
      }
    }

    /** 
     * Reads variable names and dims from a csv
     * output file.
     * 
     * @param[in] filename Name of a csv output file.
     * @param[in] skip Number of variables to skip
     * @param[out] names Names of the variables 
     * @param[out] dimss Dimensions of the variables
     */
    void
    read_variables(const std::string filename, const size_t skip, 
                   std::vector<std::string>& names,
                   std::vector<std::vector<size_t> >& dimss) {
      names.clear();
      dimss.clear();
      std::fstream csv_output_file(filename.c_str(), std::fstream::in);
      if (!csv_output_file.is_open()) {
        throw new std::runtime_error("Could not open" + filename);
      }
      std::string header = read_header(csv_output_file);
      csv_output_file.close();

      std::vector<std::string> tokens;
      tokenize(header, ',', tokens);
      get_names(tokens, skip, names);
      get_dimss(tokens, skip, dimss);
    }
    
    /** 
     * Adds a chain from a csv file.
     * 
     * @param[in,out] chains The chains object to modify
     * @param chain chain number
     * @param filename file name of a csv output file
     * @param skip number of variables to skip
     * 
     * @return number of samples added
     */
    template <typename RNG>
    size_t add_chain(stan::mcmc::chains<RNG>& chains, 
                     const size_t chain, 
                     const std::string filename,
                     const size_t skip) {
      std::fstream csv_output_file(filename.c_str(), std::fstream::in);
      if (!csv_output_file.is_open()) {
        throw new std::runtime_error("Could not open" + filename);
      }
      std::vector<std::vector<double> > thetas;
      read_values(csv_output_file, chains.num_params(), thetas);
      csv_output_file.close();
      
      std::vector<size_t> from, to;
      get_reordering(chains.param_dimss(), from, to);
      reorder_values(thetas, from, to);
      for (size_t i = 0; i < thetas.size(); i++) {
        chains.add(chain, thetas[i]);
      }
      return thetas.size();
    }

  }
}


#endif


/*


pair<double,double> smallest_interval(size_t n,
                                      double prob);

double potential_scale_reduction(size_t n)

double mcmc_error_mean(size_t n);
                   
void print(ostream&);

ostream& operator<<(ostream&, const chains&);
*/
