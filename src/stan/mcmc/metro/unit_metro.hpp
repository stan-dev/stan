#ifndef __STAN__MCMC__UNIT__METRO__HPP
#define __STAN__MCMC__UNIT__METRO__HPP

#include <stan/mcmc/metro/base_metro.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

namespace stan {

  namespace mcmc {
    
    template <class M, class BaseRNG>
    class unit_metro: public base_metro<M, BaseRNG> {
      
    public:
      
      unit_metro(M& m, BaseRNG& rng, std::ostream* error_msg)
        : base_metro<M, BaseRNG>(m, rng, error_msg)
      { this->_name = "Metropolis with a unit metric"; }

      void propose(std::vector<double>& q,
                    BaseRNG& rng) {
        for (size_t i = 0; i < q.size(); ++i) 
          q[i] = stan::prob::normal_rng(0,this->_nom_epsilon,this->_rand_int);
      }
                        
    };

  } // mcmc

} // stan
          

#endif
