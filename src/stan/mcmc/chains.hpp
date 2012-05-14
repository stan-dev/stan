#ifndef __STAN__MCMC__CHAINS_HPP__
#define __STAN__MCMC__CHAINS_HPP__

#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace stan {  

  namespace mcmc {
    
    /**
     * Storage is column/last-index major.
     */
    class chains {
    private:
      size_t _warmup;
      const std::vector<std::string> _names;
      const std::vector<std::vector<size_t> > _dimss;
      const size_t _num_params; // total
      const std::vector<size_t> _starts;
      const std::map<std::string,size_t> _name_to_index;
      std::vector<std::vector<std::vector<double > > > _samples; // [chain,param,sample]

      static size_t calc_num_params(const std::vector<size_t>& dims) {
        size_t num_params = 1;
        for (size_t i = 0;  i < dims.size(); ++i)
          num_params *= dims[i];
        return num_params;
      }

      static size_t calc_total_num_params(const std::vector<std::vector<size_t> >& dimss) {
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
              << "; num samples in chain" << k << "=" << _samples[k][0].size();
          throw std::out_of_range(msg.str());
        }
      }

      static void 
      validate_dims_idxs(const std::vector<size_t>& idxs, 
                         const std::vector<size_t>& dims) {
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
      
    public:
      chains(size_t num_chains,
             const std::vector<std::string>& names,
             const std::vector<std::vector<size_t> >& dimss) 
        : _warmup(0),
          _names(names),
          _dimss(dimss),
          _num_params(calc_total_num_params(dimss)),
          _starts(calc_starts(dimss)),               // copy
          _name_to_index(calc_name_to_index(names)), // copy
          _samples(num_chains,std::vector<std::vector<double> >(_num_params))
      {
        if (names.size() != dimss.size()) {
          std::stringstream msg;
          msg << "names and dimss mismatch in size"
              << " names.size()=" << names.size()
              << " dimss.size()=" << dimss.size();
          throw std::out_of_range(msg.str());
        }
      }
      
      
      // WRITE METHODS (need read/write synch externally)
      
      void set_warmup(size_t warmup_iterations) {
        _warmup = warmup_iterations;
      }

      // only requires synch per chain
      void add_sample(size_t chain,
                      std::vector<double> theta) {
        validate_chain_idx(chain);
        if (theta.size() != _num_params) {
          std::stringstream msg;
          msg << "parameter vector size must match num params"
              << "; num params=" << _num_params
              << "; theta.size()=" << theta.size();
          throw std::out_of_range(msg.str());
        }
        for (size_t i = 0; i < theta.size(); ++i)
          _samples[chain][i].push_back(theta[i]); // _samples very non-local
      }
      

      inline size_t warmup() {
        return _warmup;
      }

      inline size_t num_chains() {
        return _samples.size();
      }

      inline size_t num_params() { 
        return _num_params;
      }

      inline size_t num_param_names() {
        return _names.size();
      }

      size_t num_samples(size_t k) {
        validate_chain_idx(k);
        return _samples[k][0].size();
      }

      size_t num_warmup_samples(size_t k) {
        return std::min<size_t>(num_samples(k), warmup());
      }

      size_t num_saved_samples(size_t k) {
        return std::max<size_t>(0, num_samples(k) - warmup());
      }

      const std::vector<std::string>& param_names() {
        return _names;
      }
      const std::string& param_name(size_t j) {
        validate_param_name_idx(j);
        return _names[j];
      }

      const std::vector<std::vector<size_t> >& param_dimss() {
        return _dimss;
      }
      const std::vector<size_t>& param_dims(size_t j) {
        validate_param_name_idx(j);
        return _dimss[j];
      }

      const std::vector<size_t>& param_starts() {
        return _starts;
      }
      size_t param_start(size_t j) {
        validate_param_name_idx(j);
        return _starts[j];
      }

      const std::vector<size_t> param_sizes() {
        std::vector<size_t> s(num_param_names());
        for (unsigned int j = 0; j < num_param_names(); ++j)
          s[j] = param_size(j); // could optimize tests in param_sizes() out
        return s;
      }
      size_t param_size(size_t j) {
        validate_param_name_idx(j);
        if (j + 1 < _starts.size()) 
          return _starts[j+1] - _starts[j];
        return num_params() - _starts[j];
      }

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

      static size_t get_offset(const std::vector<size_t>& idxs, 
                               const std::vector<size_t>& dims) {
        validate_dims_idxs(idxs,dims);
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

      size_t get_param_num(size_t j, // param id
                           const std::vector<size_t>& idxs) {
        return get_offset(idxs,param_dims(j))
          + param_start(j);
      }

      size_t total_samples() {
        size_t total(0);
        for (size_t k = 0; k < num_chains(); ++k)
          total += num_samples(k);
        return total;
      }

      // merges across chains with concat
      void
      get_samples(size_t n,
                  std::vector<double>& samples) {
        validate_param_idx(n);
        samples.resize(total_samples());
        std::vector<double>::iterator it = samples.begin();
        size_t pos(0);
        for (size_t k = 0; k < num_chains(); ++k) {
          samples.insert(it + pos,
                         get_samples(k,n).begin(), 
                         get_samples(k,n).end());
          pos += num_samples(k);
        }
      }

      const std::vector<double>&
      get_samples(size_t k,     // chain id
                  size_t n) {   // param id
        validate_chain_idx(k);
        validate_param_idx(n); 
        return _samples[k][n];
      }

      double
      get_sample(size_t k, // chain id
                 size_t n, // param id
                 size_t m) { // iteration id
        validate_iteration(k,m);
        return get_samples(k,n)[m];
      }

      /**
       * Returns index of next element of sequence.  Initial index is
       * all zero vector, std::vector<size_t>(dims.size(),0);
       */
      static void
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

    };

  }
}


#endif
