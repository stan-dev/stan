#ifndef __STAN__MCMC__DENSE__METRO__HPP
#define __STAN__MCMC__DENSE__METRO__HPP

#include <stan/mcmc/metro/base_metro.hpp>
#include <stan/prob/distributions/multivariate/continuous/multi_normal.hpp>

namespace stan {

  namespace mcmc {
    
    template <typename M, class BaseRNG>
    class dense_metro: public base_metro<M, BaseRNG> {
      
    public:
      
      dense_metro(M& m, 
                 BaseRNG& rng, 
                 Eigen::MatrixXd& propCov, 
                 std::ostream* error_msg)
        : base_metro<M, BaseRNG>(m, rng, error_msg),
          _propCov(propCov) { 
        this->_name = "Metropolis with a dense Euclidean metric"; 
        this->_nom_epsilon = 1;
      }

      void propose(std::vector<double>& q,
                   BaseRNG& rng) {
        Eigen::MatrixXd zer(q.size(), q.size());
        zer.setZero();

        Eigen::VectorXd prop(q.size());
        prop = this->_nom_epsilon 
          * stan::prob::multi_normal_rng(zer, _propCov, this->_rand_int);

        for(size_t i = 0; i < q.size(); i++)
          q[i] = prop(i);
      }                                  
                        
    protected:

      Eigen::MatrixXd _propCov;
    };

  } // mcmc

} // stan
          

#endif
