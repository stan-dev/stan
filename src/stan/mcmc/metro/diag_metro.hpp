#ifndef __STAN__MCMC__DIAG__METRO__HPP
#define __STAN__MCMC__DIAG__METRO__HPP

#include <stan/mcmc/metro/base_metro.hpp>

namespace stan {

  namespace mcmc {

    // Metropolis on a 
    // Euclidean manifold with diagonal metric
    // and static integration time
    
    template <typename M, class BaseRNG>
    class diag_metro: public base_metro<M,
                                        BaseRNG> {
      
    public:
      
      diag_metro(M& m, 
                 BaseRNG& rng, 
                 Eigen::VectorXd& propCovDiag, 
                 std::ostream* error_msg)
        : base_metro<M, BaseRNG>(m, rng, error_msg),
          _propCovDiag(propCovDiag) { 
        this->_name = "Metropolis with a diagonal Euclidean metric"; 
        this->_nom_epsilon = 1;
      }

      void _propose(std::vector<double>& q,
                    BaseRNG& rng) {
        for (size_t i = 0; i < q.size(); ++i) 
          q[i] = stan::prob::normal_rng(0,this->_nom_epsilon * _propCovDiag(i),
                                        this->_rand_int);
      }
             
    protected:

      Eigen::VectorXd _propCovDiag;           
    };

  } // mcmc

} // stan
          

#endif
