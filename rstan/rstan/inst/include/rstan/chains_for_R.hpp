
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
     * Get the names for an array of given dimensions 
     * in the way of column majored. 
     * For exmaple, if we know an array named `a`, with
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
     * @param first_is_one[true] Where to start for the first index: 0 or 1. 
     * @return All the names for the array 
     *
     */
    std::vector<std::string>
    get_col_major_names(std::string name,
                        std::vector<size_t> dims,
                        bool first_is_one = true) {

      size_t s = dims.size();
      if (0 == s) return std::vector<std::string>(1, name);
      std::vector<size_t> steps(1, 1);
      for (size_t i = 0; i < (s - 1); i++)
        steps.push_back(steps.back() * dims[i]);

      /*
      for (tyepname std::vector<size_t>::const_iterator i = steps.begin(); 
           i != steps.end();
           ++i) {
        std::cout << *i << std::endl;
      } 
      */

      size_t total = product(dims);
      // std::cout << "total = " << total << std::endl;
      std::vector<size_t> idx(s);

      std::vector<std::string> allnames;

      for (size_t i = 0; i < total; ++i) {
        size_t ii = i;
        for (size_t j = s - 1; j > 0; --j) {
          idx[j] = ii / steps[j];
          ii -= idx[j] * steps[j];
        }
        idx[0] = ii;

        std::stringstream stri;
        stri << name << "[";

        size_t first =  first_is_one ? 1 : 0;
        for (size_t j = 0; j < s - 1 ; ++j)
          stri << idx[j] + first << ",";
        stri << idx[s - 1] + first << "]";
        allnames.push_back(stri.str());
      }
      return allnames;
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
