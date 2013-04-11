#ifndef __STAN__MCMC__STATIC__ADAPTER__STEPSIZE__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__STEPSIZE__BETA__

#include <stan/mcmc/base_adapter.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_adapter: public base_adapter {
      
    public:
      
      stepsize_adapter(): _adapt_flag(true), _adapt_mu(0.5), _adapt_delta(0.5),
                          _adapt_gamma(0.05), _adapt_kappa(0.75), _adapt_t0(10)
      { init(); }
      
      void engage_adaptation()    { _adapt_flag = true; }
      void disengage_adaptation() { _adapt_flag = false; }
      
      bool adapting() { return _adapt_flag; }
      
      void set_adapt_mu(double m)    { _adapt_mu = m; }
      void set_adapt_delta(double d) { if(d > 0 && d < 1) _adapt_delta = d; }
      void set_adapt_gamma(double g) { if(g > 0)          _adapt_gamma = g; }
      void set_adapt_kappa(double k) { if(k > 0)          _adapt_kappa = k; }
      void set_adapt_t0(double t)    { if(t > 0)          _adapt_t0 = t; }

      double get_adapt_mu()    { return _adapt_mu; }
      double get_adapt_delta() { return _adapt_delta; }
      double get_adapt_gamma() { return _adapt_gamma; }
      double get_adapt_kappa() { return _adapt_kappa; }
      double get_adapt_t0()    { return _adapt_t0; }
      
      void init() {
        _adapt_counter = 0;
        _adapt_s_bar = 0;
        _adapt_x_bar = 0;
      }
      
      void learn_stepsize(double& epsilon, double adapt_stat) {
        
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
      
    };
    
  } // mcmc
  
} // stan

#endif