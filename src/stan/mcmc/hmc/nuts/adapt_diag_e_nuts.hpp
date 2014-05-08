#ifndef __STAN__MCMC__ADAPT__DIAG__E__NUTS__BETA__
#define __STAN__MCMC__ADAPT__DIAG__E__NUTS__BETA__

#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

namespace stan {
  
  namespace mcmc {
    
    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with diagonal metric
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_diag_e_nuts: public diag_e_nuts<M, BaseRNG>,
                             public stepsize_var_adapter {
      
    public:
      
        adapt_diag_e_nuts(M &m, BaseRNG& rng,
                          std::ostream* o = &std::cout, std::ostream* e = 0):
        diag_e_nuts<M, BaseRNG>(m, rng, o, e),
        stepsize_var_adapter(m.num_params_r())
      {};
      
      ~adapt_diag_e_nuts() {};
      
      sample transition(sample& init_sample) {
        
        sample s = diag_e_nuts<M, BaseRNG>::transition(init_sample);
        
        if (this->_adapt_flag) {
        
          this->_stepsize_adaptation.learn_stepsize(this->_nom_epsilon, s.accept_stat());
          
          bool update = this->_var_adaptation.learn_variance(this->_z.mInv, this->_z.q);
          
          if(update) {
            this->init_stepsize();
            
            this->_stepsize_adaptation.set_mu(log(10 * this->_nom_epsilon));
            this->_stepsize_adaptation.restart();
          }
          
        }
        
        return s;
        
      }
                               
      void disengage_adaptation() {
        base_adapter::disengage_adaptation();
        this->_stepsize_adaptation.complete_adaptation(this->_nom_epsilon);
      }

    };
    
  } // mcmc
  
} // stan


#endif
