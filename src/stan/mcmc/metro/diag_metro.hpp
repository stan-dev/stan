#ifndef __STAN__MCMC__DIAG__METRO__HPP
#define __STAN__MCMC__DIAG__METRO__HPP

#include <stan/mcmc/metro/base_metro.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

namespace stan {

  namespace mcmc {
    
    template <typename M, class BaseRNG>
    class diag_metro: public base_metro<M, BaseRNG> {
      
    public:
      
      diag_metro(M& m, 
                 BaseRNG& rng, 
                 std::ostream* error_msg)
        : base_metro<M, BaseRNG>(m, rng, error_msg) { 
        this->_name = "Metropolis with a diagonal Euclidean metric"; 
        this->_nom_epsilon = 1;
      }

      void propose(std::vector<double>& q,
                    BaseRNG& rng) {
        for (size_t i = 0; i < q.size(); ++i) 
          q[i] = stan::prob::normal_rng(0,this->_nom_epsilon * _prop_cov_diag(i),
                                        this->_rand_int);
      }
             
    protected:

      Eigen::VectorXd _prop_cov_diag;           
    };

  } // mcmc

} // stan
          

#endif
