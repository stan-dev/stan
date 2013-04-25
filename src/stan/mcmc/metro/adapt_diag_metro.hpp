#ifndef __STAN__MCMC__ADAPT__DIAG__METRO__HPP__
#define __STAN__MCMC__ADAPT__DIAG__METRO__HPP__

#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/metro/diag_metro.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename M, class BaseRNG>
    class adapt_diag_metro: public diag_metro<M, BaseRNG>,
                            public stepsize_var_adapter {
      
    public:
      
      adapt_diag_metro(M &m, BaseRNG& rng, std::ostream* error_msg)
        : diag_metro<M, BaseRNG>(m, rng, this->_propCovDiag, error_msg),
          stepsize_var_adapter(m.num_params_r())
      {
        this->_propCovDiag.resize(m.num_params_r());
        this->_propCovDiag.setIdentity();
};
      
      ~adapt_diag_metro() {};
      
      sample transition(sample& init_sample) {
        sample s = diag_metro<M, BaseRNG>::transition(init_sample);
        
        if (this->_adapt_flag) {
          
          this->_stepsize_adaptation.learn_stepsize(this->_nom_epsilon, 
                                                    s.accept_stat());
          bool update = this->_var_adaptation.learn_variance(this->_propCovDiag, 
                                                             this->_params_r);
          
          if(update) {
            this->init_stepsize();
            
            this->_stepsize_adaptation.set_mu(log(10 * this->_nom_epsilon));
            this->_stepsize_adaptation.restart();
          }
          
        }
        
        return s;
        
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
