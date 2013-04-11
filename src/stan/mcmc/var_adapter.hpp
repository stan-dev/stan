#ifndef __STAN__MCMC__STATIC__ADAPTER__VAR__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__VAR__BETA__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
  
  namespace mcmc {
        
    class var_adapter: public stepsize_adapter {
      
    public:
      
      var_adapter(int n): _adapt_m(Eigen::VectorXd::Zero(n)),
                          _adapt_m2(Eigen::VectorXd::Zero(n)) 
      { init(); }
      
      void init() {
        
        stepsize_adapter::init();
        
        _adapt_var_counter = 0;
        _adapt_var_next = 10;
        
        _adapt_n = 0;
        _adapt_m.setZero();
        _adapt_m2.setZero();
        
      }
      
      bool learn_variance(Eigen::VectorXd& var, std::vector<double>& q) {
        
        ++_adapt_var_counter;
        
        Eigen::Map<Eigen::VectorXd> x(&q[0], q.size());
        
        // Welford algorithm for online variance estimate
        ++_adapt_n;
        
        Eigen::VectorXd delta(x - _adapt_m);
        _adapt_m  += delta / _adapt_n;
        _adapt_m2 += delta.cwiseProduct(x - _adapt_m);
        
        if (_adapt_var_counter == _adapt_var_next) {
          
          _adapt_var_next *= 2;
          
          var = _adapt_m2 / (_adapt_n - 1.0);
          
          var = (_adapt_n / (_adapt_n + 5.0)) * var
          + (5.0 / (_adapt_n + 5.0)) * Eigen::VectorXd::Ones(var.size());
          
          _adapt_n = 0;
          _adapt_m.setZero();
          _adapt_m2.setZero();
          
          return true;
          
        }
        
        return false;
        
      }
      
    protected:
      
      double _adapt_var_counter;
      double _adapt_var_next;
      
      double _adapt_n;
      Eigen::VectorXd _adapt_m;
      Eigen::VectorXd _adapt_m2;
      
    };
    
  } // mcmc
  
} // stan

#endif