#ifndef __STAN__MCMC__STATIC__ADAPTER__COVAR__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__COVAR__BETA__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
  
  namespace mcmc {
        
    class covar_adapter: public stepsize_adapter {
      
    public:
      
      covar_adapter(int n): _adapt_m(Eigen::VectorXd::Zero(n)),
                            _adapt_m2(Eigen::MatrixXd::Zero(n, n)) 
        { init(); }
      
      void init() {
        
        stepsize_adapter::init();
        
        _adapt_covar_counter = 0;
        _adapt_covar_next = 10;
        
        _adapt_n = 0;
        _adapt_m.setZero();
        _adapt_m2.setZero();
        
      }
      
      bool learn_covariance(Eigen::MatrixXd& covar, std::vector<double>& q) {
        
        ++_adapt_covar_counter;
        
        Eigen::Map<Eigen::VectorXd> x(&q[0], q.size());
        
        // Welford algorithm for online covariance estimate
        ++_adapt_n;
        
        Eigen::VectorXd delta(x - _adapt_m);
        _adapt_m  += delta / _adapt_n;
        _adapt_m2 += (x - _adapt_m) * delta.transpose();
        
        if (_adapt_covar_counter == _adapt_covar_next) {
          
          _adapt_covar_next *= 2;
          
          covar = _adapt_m2 / (_adapt_n - 1.0);
          
          covar = (_adapt_n / (_adapt_n + 5.0)) * covar
          + (5.0 / (_adapt_n + 5.0)) * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());
          
          _adapt_n = 0;
          _adapt_m.setZero();
          _adapt_m2.setZero();
          
          return true;
          
        }
        
        return false;
        
      }
      
    protected:
      
      double _adapt_covar_counter;
      double _adapt_covar_next;
      
      double _adapt_n;
      Eigen::VectorXd _adapt_m;
      Eigen::MatrixXd _adapt_m2;
      
    };
    
  } // mcmc
  
} // stan

#endif