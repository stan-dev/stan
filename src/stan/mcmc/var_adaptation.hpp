#ifndef __STAN__MCMC__VAR__ADAPTATION__BETA__
#define __STAN__MCMC__VAR__ADAPTATION__BETA__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

#include <stan/prob/welford_var_estimator.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class var_adaptation: public windowed_adaptation {
      
    public:
      
      var_adaptation(int n): windowed_adaptation("variance"), _estimator(n) {}

      bool learn_variance(Eigen::VectorXd& var, const Eigen::VectorXd& q) {

        if (adaptation_window()) _estimator.add_sample(q);

        if (end_adaptation_window()) {
          
          compute_next_window();
          
          _estimator.sample_variance(var);
          
          double n = static_cast<double>(_estimator.num_samples());
          var = (n / (n + 5.0)) * var
                + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());
          
          _estimator.restart();
          
          ++_adapt_window_counter;
          return true;
          
        }
        
        ++_adapt_window_counter;
        return false;
        
      }
      
    protected:

      prob::welford_var_estimator _estimator;
      
    };
    
  } // mcmc
  
} // stan

#endif