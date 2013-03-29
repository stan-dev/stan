#ifndef __STAN__MCMC__STATIC__ADAPTER__COVAR__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__COVAR__BETA__

#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
  
  namespace mcmc {
        
    class covar_adapter: public stepsize_adapter {
      
    public:
      
      covar_adapter(int n): _sum_x(Eigen::VectorXd::Zero(n)),
                            _sum_xxt(Eigen::MatrixXd::Zero(n, n)) 
        { init(); }
      
      void init() {
        
        stepsize_adapter::init();
        
        _adapt_covar_counter = 0;
        _adapt_covar_next = 1;
        
        _sum_n = 0;
        _sum_x.setZero();
        _sum_xxt.setZero();
        
      }
      
    protected:
      
      double _adapt_covar_counter;
      double _adapt_covar_next;
      
      double _sum_n;
      Eigen::VectorXd _sum_x;
      Eigen::MatrixXd _sum_xxt;
      
      void _learn_covariance(Eigen::MatrixXd& covar, std::vector<double>& q) {
        
        ++_adapt_covar_counter;
        
        Eigen::Map<Eigen::VectorXd> x(&q[0], q.size());
        
        ++_sum_n;
        _sum_x += x;
        _sum_xxt += x * x.transpose();
        
        if (_adapt_covar_counter == _adapt_covar_next) {
          
          _adapt_covar_next *= 2;
          
          _sum_x /= _sum_n;
          _sum_xxt /= _sum_n;
          
          covar = _sum_xxt - _sum_x * _sum_x.transpose();
          
          const double norm = covar.trace() / covar.rows();
          if(norm) {
            
            covar *= _sum_n / (norm * (_sum_n + 5)) ;
            for (size_t i = 0; i < covar.rows(); i++)
              covar(i, i) =  ( (_sum_n + 2) / _sum_n ) * covar(i, i) + ( 3.0 / (_sum_n + 5) );
            
          }
          else
            covar = Eigen::MatrixXd::Identity(covar.rows(), covar.cols());
          
          _sum_n = 0;
          _sum_x.setZero();
          _sum_xxt.setZero();
          
        }
        
      }
      
    };
    
    
  } // mcmc
  
} // stan

#endif