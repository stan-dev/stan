#ifndef __STAN__MCMC__STATIC__ADAPTER__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__BETA__

#include <Eigen/Dense>

namespace stan {
  
  namespace mcmc {
    
    class adapter {
      
    public:
      
      adapter(): _adapt_mu(0.5), _adapt_delta(0.651), _adapt_gamma(0.05),
                 _adapt_kappa(0.75), _adapt_t0(10) { init(); }
      
      void set_mu(double m) { _adapt_mu = m; }
      void set_delta(double d) { _adapt_delta = d; }
      void set_gamma(double g) { _adapt_gamma = g; }
      void set_kappa(double k) { _adapt_kappa = k; }
      void set_t0(double t) { _adapt_t0 = t; }
      
      void init();
      
    protected:
      
      double _adapt_counter; // Adaptation iteration
      double _adapt_s_bar;   // Moving average statistic
      double _adapt_x_bar;   // Moving average parameter
      double _adapt_mu;      // Asymptotic mean of parameter
      double _adapt_delta;   // Target value of statistic
      double _adapt_gamma;   // Adaptation scaling
      double _adapt_kappa;   // Adaptation shrinkage
      double _adapt_t0;      // Effective starting iteration
      
      double _adapt_var_counter;
      double _adapt_var_next;
      
      // Eigen::Vector sum_x;
      
      double _adapt_covar_counter;
      double _adapt_covar_next;
      
      // Eigen::Vector sum_x;
      // Eigen::Vector sum_x2;
      
      void _learn_stepsize(double& epsilon, double adapt_stat);
      //void _learn_variance(Eigen::VectorXd& var, ps_point& z);
      //void _learn_covariance(Eigen::MatrixXd& covar, ps_point& z);
      
    };
    
    void adapter::init() {
      _adapt_counter = 0;
      _adapt_s_bar = 0;
      _adapt_x_bar = 0;
      
      _adapt_var_counter = 0;
      _adapt_var_next = 1;
      _adapt_covar_counter = 0;
      _adapt_covar_next = 1;
    }
    
    void adapter::_learn_stepsize(double& epsilon, double adapt_stat) {
       
      ++_adapt_counter;
      
      adapt_stat = adapt_stat > 1 ? 1 : adapt_stat;

      // Nesterov Dual-Averaging of log(epsilon)
      const double eta = 1.0 / (_adapt_counter + _adapt_t0);
      
      _adapt_s_bar = (1.0 - eta) * _adapt_s_bar + eta * (_adapt_delta - adapt_stat);
      
      const double x = _adapt_mu - _adapt_s_bar * sqrt(_adapt_counter) / _adapt_gamma;
      const double x_eta = pow(_adapt_counter, _adapt_kappa);
      
      _adapt_x_bar = (1.0 - x_eta) * _adapt_x_bar + x_eta * x;
      
      epsilon = exp(_adapt_x_bar);
      
    }
    
    /*
    void adapter::_learn_variance(Eigen::VectorXd& var, ps_point& z) {
      
      ++_adapt_var_counter;
      
      if (_adapt_var_counter == _adapt_var_next) {
        
        _adapt_var_next *= 2;
        
        double step_size_sq_sum = 0;
        
        for (size_t i = 0; i < var.size(); i++) {
          
          double Ex = _x_sum[i] / _x_sum_n;
          double Exsq = _xsq_sum[i] / _x_sum_n;
          
          _x_sum[i] = 0;
          _xsq_sum[i] = 0;
          
          _step_sizes[i] = sqrt(Exsq - Ex*Ex);
          step_size_sq_sum += _step_sizes[i] * _step_sizes[i];
        }
        
        if (step_size_sq_sum > 0.0) {
          
          _x_sum_n = 0;
          double normalizer = sqrt((double)_step_sizes.size()) / sqrt(step_size_sq_sum);
          
          for (size_t i = 0; i < _step_sizes.size(); i++)
            _step_sizes[i] *= normalizer;
        } 
        else {
          for (size_t i = 0; i < _step_sizes.size(); i++)
            _step_sizes[i] = 1.0;
        }
      
      }
      
    }
    
    void adapter::_learn_covariance(Eigen::MatrixXd& covar, ps_point& z) {
     
      ++_adapt_covar_counter;
      
      std::vector<double>& q = z.q;
      Eigen::Map<Eigen::VectorXd> x(&q[0], q.size());
      
      ++sum_n;
      sum_x += x;
      sum_x2 += x * x.transpose();
      
      if (_adapt_covar_counter == _adapt_covar_next) {
        
        _adapt_covar_next *= 2;

        sum_x /= sum_n;
        sum_x2 /= sum_n;
        
        covar = sum_x2 - sum_x * sum_x.transpose();
        
        const double norm = _covar.trace() / _covar.rows();
        if(norm) {
          
          covar *= sum_n / (norm * (sum_n + 5)) ;
          for(size_t i = 0; i < covar.rows(); i++)
            covar(i, i) =  ( (sum_x + 2) / sum_n ) * covar(i, i) + ( 3.0 / (sum_n + 5) );
          
        }
        else
          covar = Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

        //_cov_L = _cov_mat.selfadjointView<Eigen::Upper>().llt().matrixL();
        sum_n *= 0;
        sum_x *= 0;
        sum_x2 *= 0;
        
      }
       
    }
    */
    
  } // mcmc
  
} // stan

#endif