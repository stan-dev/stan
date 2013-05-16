#ifndef __STAN__MCMC__STEPSIZE__ADAPTATION__BETA__
#define __STAN__MCMC__STEPSIZE__ADAPTATION__BETA__

#include <cmath>
#include <stan/mcmc/base_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_adaptation: public base_adaptation {
      
    public:
      
      stepsize_adaptation(): _mu(0.5), _delta(0.5), _gamma(0.05),
                             _kappa(0.75), _t0(10)
      { restart(); }
      
      void set_mu(double m)    { _mu = m; }
      void set_delta(double d) { if(d > 0 && d < 1) _delta = d; }
      void set_gamma(double g) { if(g > 0)          _gamma = g; }
      void set_kappa(double k) { if(k > 0)          _kappa = k; }
      void set_t0(double t)    { if(t > 0)          _t0 = t; }
      
      double get_mu()    { return _mu; }
      double get_delta() { return _delta; }
      double get_gamma() { return _gamma; }
      double get_kappa() { return _kappa; }
      double get_t0()    { return _t0; }
      
      void restart() {
        _counter = 0;
        _s_bar = 0;
        _x_bar = 0;
      }
      
      void learn_stepsize(double& epsilon, double adapt_stat) {
        
        ++_counter;
        
        adapt_stat = adapt_stat > 1 ? 1 : adapt_stat;
        
        // Nesterov Dual-Averaging of log(epsilon)
        const double eta = 1.0 / (_counter + _t0);
        
        _s_bar = (1.0 - eta) * _s_bar + eta * (_delta - adapt_stat);
        
        const double x = _mu - _s_bar * std::sqrt(_counter) / _gamma;
        const double x_eta = std::pow(_counter, - _kappa);
        
        _x_bar = (1.0 - x_eta) * _x_bar + x_eta * x;
        
        epsilon = std::exp(x);
        
      }
      
      void complete_adaptation(double& epsilon) {
        epsilon = std::exp(_x_bar);
      }
      
    protected:
      
      double _counter; // Adaptation iteration
      double _s_bar;   // Moving average statistic
      double _x_bar;   // Moving average parameter
      double _mu;      // Asymptotic mean of parameter
      double _delta;   // Target value of statistic
      double _gamma;   // Adaptation scaling
      double _kappa;   // Adaptation shrinkage
      double _t0;      // Effective starting iteration
      
    };
    
  } // mcmc
  
} // stan

#endif
