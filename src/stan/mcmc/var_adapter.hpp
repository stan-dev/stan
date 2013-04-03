#ifndef __STAN__MCMC__STATIC__ADAPTER__VAR__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__VAR__BETA__

#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
  
  namespace mcmc {
        
    class var_adapter: public stepsize_adapter {
      
    public:
      
      var_adapter(int n): _sum_x(Eigen::VectorXd::Zero(n)),
                          _sum_x2(Eigen::VectorXd::Zero(n)) 
      { init(); }
      
      void init() {
        
        stepsize_adapter::init();
        
        _adapt_var_counter = 0;
        _adapt_var_next = 1;
        
        _sum_n = 0;
        _sum_x.setZero();
        _sum_x2.setZero();
        
      }
      
    protected:
      
      double _adapt_var_counter;
      double _adapt_var_next;
      
      double _sum_n;
      Eigen::VectorXd _sum_x;
      Eigen::VectorXd _sum_x2;
      
      void _learn_variance(Eigen::VectorXd& var, std::vector<double>& q) {
        
        ++_adapt_var_counter;
        
        Eigen::Map<Eigen::VectorXd> x(&q[0], q.size());
        
        ++_sum_n;
        _sum_x += x;
        _sum_x2 += x.cwiseAbs2();
        
        if (_adapt_var_counter == _adapt_var_next) {
          
          _adapt_var_next *= 2;
          
          _sum_x /= _sum_n;
          _sum_x2 /= _sum_n;
          
          var = _sum_x2 - _sum_x.cwiseAbs2();
          
          const double norm = var.squaredNorm() / static_cast<double>(var.size());
          
          if (norm) {
            var /= norm;
          }
          else
            var.setOnes();
          
          _sum_n = 0;
          _sum_x.setZero();
          _sum_x2.setZero();
         
        }
        
      }
      
    };
    
  } // mcmc
  
} // stan

#endif