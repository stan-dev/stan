
#ifndef __RSTAN__CHAINS_FOR_R_HPP__
#define __RSTAN__CHAINS_FOR_R_HPP__
#include <stan/mcmc/chains.hpp>

namespace rstan {

  namespace {

    size_t product(std::vector<size_t> dims) {
      size_t y = 1U;
      for (size_t i = 0; i < dims.size(); ++i)
        y *= dims[i];
      return y;
    }

    /**
     *  Get the parameter indices for a vector(array) parameter.
     *  For example, we have parameter beta, which has 
     *  dimension [2,3]. Then this function gets 
     *  the indices as (if col_major = false)
     *  [0,0], [0,1], [0,2] 
     *  [1,0], [1,1], [1,2] 
     *  or (if col_major = true) 
     *  [0,0], [1,0] 
     *  [0,1], [1,1] 
     *  [0,2], [121] 
     *
     *  @param dim[in] the dimension of parameter
     *  @param idx[out] for keeping all the indices
     *
     *  <p> when idx is empty (size = 0), idx 
     *  would be inserted an empty vector. 
     * 
     *
     */
    
    void expand_indices(std::vector<size_t> dim,
                        std::vector<std::vector<size_t> >& idx,
                        bool col_major = false) {
    
      size_t len = dim.size();
    
      idx.resize(0);
      size_t total = product(dim);
    
      std::vector<size_t> loopj;
      for (size_t i = 1; i <= len; ++i)
        loopj.push_back(len - i);
    
      if (col_major)
        for (size_t i = 0; i < len; ++i)
          loopj[i] = len - 1 - loopj[i];
    
      idx.push_back(std::vector<size_t>(len, 0));
      for (size_t i = 1; i < total; i++) {
        std::vector<size_t>  v(idx.back());
        for (size_t j = 0; j < len; ++j) {
          size_t k = loopj[j];
          if (v[k] < dim[k] - 1) {
            v[k] += 1;
            break;
          }
          v[k] = 0;
        }
        idx.push_back(v);
      }
    }

    /**
     * Get the names for an array of given dimensions 
     * in the way of column majored. 
     * For example, if we know an array named `a`, with
     * dimensions of [2, 3, 4], the names then are (starting
     * from 0):
     * a[0,0,0]
     * a[1,0,0]
     * a[0,1,0]
     * a[1,1,0]
     * a[0,2,0]
     * a[1,2,0]
     * a[0,0,1]
     * a[1,0,1]
     * a[0,1,1]
     * a[1,1,1]
     * a[0,2,1]
     * a[1,2,1]
     * a[0,0,2]
     * a[1,0,2]
     * a[0,1,2]
     * a[1,1,2]
     * a[0,2,2]
     * a[1,2,2]
     * a[0,0,3]
     * a[1,0,3]
     * a[0,1,3]
     * a[1,1,3]
     * a[0,2,3]
     * a[1,2,3]
     *
     * @param name The name of the array variable 
     * @param dims The dimensions of the array 
     * @param allnames[out] Where the names would be pushed. 
     * @param first_is_one[true] Where to start for the first index: 0 or 1. 
     *
     */
    void
    get_col_major_names(const std::string& name,
                        const std::vector<size_t>& dims,
                        std::vector<std::string>& allnames,
                        bool first_is_one = true) {

      allnames.resize(0);

      if (0 == dims.size()) {
        allnames.push_back(name);
        return;
      }

      std::vector<std::vector<size_t> > idx;
      expand_indices(dims, idx, true);
      size_t first = first_is_one ? 1 : 0;
      for (std::vector<std::vector<size_t> >::const_iterator it = idx.begin();
           it != idx.end();
           ++it) {
        std::stringstream stri;
        stri << name << "[";

        size_t lenm1 = it -> size() - 1;
        for (size_t i = 0; i < lenm1; i++)
          stri << ((*it)[i] + first) << ",";
        stri << ((*it)[lenm1] + first) << "]";
        allnames.push_back(stri.str());
      }
    }
  }

  /**
   * 
   * Improve efficiency (for example less copying) and 
   * use different indices (i.e., starting from 1, not 0) from error message. 
   *
   * 
   */ 
  // placeholder for the time being 
  template <typename RNG = boost::random::ecuyer1988>
  class chains_for_R : public stan::mcmc::chains<RNG> {
  private: 
    // names for the parameters with dimensions:
    // for example, `a' is `a';
    // `b' (vector of length 3) is `b[1]', `b[2]', `b[3].'
    // std::vector<std::string > _flatnames;  

  public:

    chains_for_R(const size_t num_chains,
                 const std::vector<std::string>& names,
                 const std::vector<std::vector<size_t> >& dimss) : 
      stan::mcmc::chains<RNG>(num_chains, names, dimss)
    { 
      /*
      for (std::vector<std::string>::const_iterator it = names.begin(); 
           it != names.end();
           ++it) {
        size_t j = this -> param_name_to_index(*it);
        std::vector<std::string> names2 
          = get_col_major_names(*it, this -> param_dims(j));
        _flatnames.insert(_flatnames.end(), names2.begin(), names2.end()); 
      }
      if (_flatnames.size() != this -> num_params()) 
        throw(std::out_of_range("construct flat names wrong")); 
      */
    } 
  };
}

#endif 
