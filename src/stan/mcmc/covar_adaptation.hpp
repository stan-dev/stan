#ifndef STAN__MCMC__COVAR__ADAPTATION__BETA
#define STAN__MCMC__COVAR__ADAPTATION__BETA

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

#include <stan/prob/welford_covar_estimator.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
        
    class covar_adaptation: public windowed_adaptation {
      
    public:
      
      covar_adaptation(int n): windowed_adaptation("covariance"), estimator_(n) {}
      
      bool learn_covariance(Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
        
        if (adaptation_window()) estimator_.add_sample(q);
        
        if (end_adaptation_window()) {
          
          compute_next_window();
          
          estimator_.sample_covariance(covar);
          
          double n = static_cast<double>(estimator_.num_samples());
          covar = (n / (n + 5.0)) * covar
                  + 1e-3 * (5.0 / (n + 5.0)) * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

          estimator_.restart();
          
          ++adapt_window_counter_;
          return true;
          
        }
        
        ++adapt_window_counter_;
        return false;
        
      }
      
    protected:
      
      prob::welford_covar_estimator estimator_;
      
    };
    
  } // mcmc
  
} // stan

#endif
