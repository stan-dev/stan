#ifndef STAN__MCMC__VAR__ADAPTATION__BETA
#define STAN__MCMC__VAR__ADAPTATION__BETA

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

#include <stan/prob/welford_var_estimator.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class var_adaptation: public windowed_adaptation {
      
    public:
      
      var_adaptation(int n): windowed_adaptation("variance"), estimator_(n) {}

      bool learn_variance(Eigen::VectorXd& var, const Eigen::VectorXd& q) {

        if (adaptation_window()) estimator_.add_sample(q);

        if (end_adaptation_window()) {
          
          compute_next_window();
          
          estimator_.sample_variance(var);
          
          double n = static_cast<double>(estimator_.num_samples());
          var = (n / (n + 5.0)) * var
                + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());
          
          estimator_.restart();
          
          ++adapt_window_counter_;
          return true;
          
        }
        
        ++adapt_window_counter_;
        return false;
        
      }
      
    protected:
      
      prob::welford_var_estimator estimator_;
      
      
    };
    
  } // mcmc
  
} // stan

#endif
