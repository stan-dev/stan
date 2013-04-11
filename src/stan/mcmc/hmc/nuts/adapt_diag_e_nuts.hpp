#ifndef __STAN__MCMC__ADAPT__DIAG__E__NUTS__BETA__
#define __STAN__MCMC__ADAPT__DIAG__E__NUTS__BETA__

#include <stan/mcmc/var_adapter.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

namespace stan {
  
  namespace mcmc {
    
    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with diagonal metric
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_diag_e_nuts: public diag_e_nuts<M, BaseRNG>, public var_adapter {
      
    public:
      
      adapt_diag_e_nuts(M &m, BaseRNG& rng): diag_e_nuts<M, BaseRNG>(m, rng),
                                             var_adapter(m.num_params_r())
      {};
      
      ~adapt_diag_e_nuts() {};
      
      sample transition(sample& init_sample) {
        
        sample s = diag_e_nuts<M, BaseRNG>::transition(init_sample);
        
        if (this->_adapt_flag) {
        
          this->learn_stepsize(this->_epsilon, s.accept_stat());
          
          bool update = this->learn_variance(this->_z.mInv, this->_z.q);
          
          if(update) {
            this->init_stepsize();
            
            this->set_adapt_mu(log(10 * this->_epsilon));
            this->stepsize_adapter::init();
          }
          
        }
        
        return s;
        
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
