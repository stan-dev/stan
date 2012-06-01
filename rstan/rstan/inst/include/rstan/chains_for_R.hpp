
#ifndef __RSTAN__CHAINS_FOR_R_HPP__
#define __RSTAN__CHAINS_FOR_R_HPP__
#include <stan/mcmc/chains.hpp>

namespace rstan {
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
  public:
    chains_for_R(const size_t num_chains,
                 const std::vector<std::string>& names,
                 const std::vector<std::vector<size_t> >& dimss) : 
      stan::mcmc::chains<RNG>(num_chains, names, dimss) 
    { } 
  };
}

#endif 
