#ifndef __STAN__MCMC__COVAR__ADAPTATION__BETA__
#define __STAN__MCMC__COVAR__ADAPTATION__BETA__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

#include <stan/prob/welford_covar_estimator.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class covar_adaptation: public windowed_adaptation {
      
    public:
      
      covar_adaptation(int n): windowed_adaptation("covariance"), _estimator(n) {}
      
      bool learn_covariance(Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
        
        if (adaptation_window()) _estimator.add_sample(q);
        
        if (end_adaptation_window()) {
          
          compute_next_window();
          
          _estimator.sample_covariance(covar);
          
          double n = static_cast<double>(_estimator.num_samples());
          covar = (n / (n + 5.0)) * covar
                  + 1e-3 * (5.0 / (n + 5.0)) * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

          _estimator.restart();
          
          ++_adapt_window_counter;
          return true;
          
        }
        
        ++_adapt_window_counter;
        return false;
        
      }
      
    protected:
      
      prob::welford_covar_estimator _estimator;
      
    };
    
  } // mcmc
  
} // stan

#endif
