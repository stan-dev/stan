#ifndef __STAN__MCMC__COVAR__ADAPTATION__BETA__
#define __STAN__MCMC__COVAR__ADAPTATION__BETA__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

#include <stan/prob/welford_covar_estimator.hpp>
#include <stan/mcmc/base_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class covar_adaptation: public base_adaptation {
      
    public:
      
      covar_adaptation(int n, int max_adapt): _estimator(n) {
        
        restart();
      
        int delta = 0.1 * max_adapt;
        delta = delta > 100 ? 100 : delta;
        
        _adapt_max_adapt = max_adapt - delta;
        
      }
      
      void restart() {
        _adapt_covar_counter = 0;
        _adapt_covar_next = 10;
        _estimator.restart();
      }
      
      bool learn_covariance(Eigen::MatrixXd& covar, std::vector<double>& q) {
        
        ++_adapt_covar_counter;
        
        _estimator.add_sample(q);
        
        if (_adapt_covar_counter == _adapt_covar_next) {
          
          _adapt_covar_next *= 2;
          
          // If the following window would straddle the total
          // number of adaptive iterations then stetch the
          // current window to meet the total number of iterations
          if (_adapt_max_adapt && 2 * _adapt_covar_next > _adapt_max_adapt)
            _adapt_covar_next = _adapt_max_adapt;
          
          _estimator.sample_covariance(covar);
          
          double n = static_cast<double>(_estimator.num_samples());
          covar = (n / (n + 5.0)) * covar
                  + 1e-3 * (5.0 / (n + 5.0)) * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());
          
          _estimator.restart();
          
          return true;
          
        }
        
        return false;
        
      }
      
    protected:
      
      double _adapt_covar_counter;
      double _adapt_covar_next;
      int _adapt_max_adapt;
      
      prob::welford_covar_estimator _estimator;
      
    };
    
  } // mcmc
  
} // stan

#endif