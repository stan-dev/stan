#ifndef __STAN__MCMC__STATIC__ADAPTER__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__BETA__

#include <stan/mcmc/hmc_base.hpp>
#include <Eigen/Dense>

namespace stan {
  
  namespace mcmc {
    
    class base_adapter {
      
    public:
      
      virtual void init() = 0;
      
    };
    
    class stepsize_adapter: public base_adapter {
      
    public:
      
      stepsize_adapter(): _adapt_flag(true), _adapt_mu(0.5), _adapt_delta(0.651),
                          _adapt_gamma(0.05), _adapt_kappa(0.75), _adapt_t0(10)
                          { init(); }
      
      void engage_adaptation() { _adapt_flag = true; }
      void disengage_adaptation() { _adapt_flag = false; }
      
      bool adapting() { return _adapt_flag; }
      
      void set_adapt_mu(double m) { _adapt_mu = m; }
      void set_adapt_delta(double d) { _adapt_delta = d; }
      void set_adapt_gamma(double g) { _adapt_gamma = g; }
      void set_adapt_kappa(double k) { _adapt_kappa = k; }
      void set_adapt_t0(double t) { _adapt_t0 = t; }
      
      void init();
      
    protected:
      
      bool _adapt_flag;
      
      double _adapt_counter; // Adaptation iteration
      double _adapt_s_bar;   // Moving average statistic
      double _adapt_x_bar;   // Moving average parameter
      double _adapt_mu;      // Asymptotic mean of parameter
      double _adapt_delta;   // Target value of statistic
      double _adapt_gamma;   // Adaptation scaling
      double _adapt_kappa;   // Adaptation shrinkage
      double _adapt_t0;      // Effective starting iteration
      
      void _learn_stepsize(double& epsilon, double adapt_stat);
      
    };
    
    void stepsize_adapter::init() {
      _adapt_counter = 0;
      _adapt_s_bar = 0;
      _adapt_x_bar = 0;
    }
    
    void stepsize_adapter::_learn_stepsize(double& epsilon, double adapt_stat) {
       
      ++_adapt_counter;
      
      adapt_stat = adapt_stat > 1 ? 1 : adapt_stat;

      // Nesterov Dual-Averaging of log(epsilon)
      const double eta = 1.0 / (_adapt_counter + _adapt_t0);
      
      _adapt_s_bar = (1.0 - eta) * _adapt_s_bar + eta * (_adapt_delta - adapt_stat);
      
      const double x = _adapt_mu - _adapt_s_bar * sqrt(_adapt_counter) / _adapt_gamma;
      const double x_eta = pow(_adapt_counter, - _adapt_kappa);
      
      _adapt_x_bar = (1.0 - x_eta) * _adapt_x_bar + x_eta * x;
      
      epsilon = exp(_adapt_x_bar);
      
    }
    
    class var_adapter: public stepsize_adapter {
      
    public:
      
      var_adapter(int n): _sum_x(Eigen::VectorXd::Zero(n)),
                          _sum_x2(Eigen::VectorXd::Zero(n)) { init(); }
      
      void init();
      
    protected:
      
      double _adapt_var_counter;
      double _adapt_var_next;
      
      double _sum_n;
      Eigen::VectorXd _sum_x;
      Eigen::VectorXd _sum_x2;

      void _learn_variance(Eigen::VectorXd& var, std::vector<double>& q);
      
    };
    
    void var_adapter::init() {
      
      stepsize_adapter::init();
      
      _adapt_var_counter = 0;
      _adapt_var_next = 1;
      
      _sum_n = 0;
      _sum_x.setZero();
      _sum_x2.setZero();
      
    }
    
    void var_adapter::_learn_variance(Eigen::VectorXd& var, std::vector<double>& q) {

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

    class covar_adapter: public stepsize_adapter {
      
    public:
      
      covar_adapter(int n): _sum_x(Eigen::VectorXd::Zero(n)),
                            _sum_xxt(Eigen::MatrixXd::Zero(n, n)) { init(); }
      
      void init();
      
    protected:

      double _adapt_covar_counter;
      double _adapt_covar_next;
      
      double _sum_n;
      Eigen::VectorXd _sum_x;
      Eigen::MatrixXd _sum_xxt;
      
      void _learn_covariance(Eigen::MatrixXd& covar, std::vector<double>& q);
      
    };
    
    void covar_adapter::init() {
      
      stepsize_adapter::init();
      
      _adapt_covar_counter = 0;
      _adapt_covar_next = 1;
      
      _sum_n = 0;
      _sum_x.setZero();
      _sum_xxt.setZero();
      
    }
    
    void covar_adapter::_learn_covariance(Eigen::MatrixXd& covar, std::vector<double>& q) {

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

    
  } // mcmc
  
} // stan

#endif