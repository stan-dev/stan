#ifndef __STAN__MCMC__ADAPT__DENSE__E__STATIC__HMC__BETA__
#define __STAN__MCMC__ADAPT__DENSE__E__STATIC__HMC__BETA__

#include <stan/mcmc/covar_adapter.hpp>
#include <stan/mcmc/hmc/static/dense_e_static_hmc.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with dense metric,
    // static integration time,
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_dense_e_static_hmc: public dense_e_static_hmc<M, BaseRNG>, public covar_adapter {
      
    public:
      
      adapt_dense_e_static_hmc(M &m, BaseRNG& rng): dense_e_static_hmc<M, BaseRNG>(m, rng),
                                                    covar_adapter(m.num_params_r())
      {};
      
      ~adapt_dense_e_static_hmc() {};
      
      sample transition(sample& init_sample) {
        
        sample s = dense_e_static_hmc<M, BaseRNG>::transition(init_sample);
        
        if (this->_adapt_flag) {
          
          this->learn_stepsize(this->_nom_epsilon, s.accept_stat());
          this->_update_L();
          
          bool update = this->learn_covariance(this->_z.mInv, this->_z.q);
          
          if(update) {
            this->init_stepsize();
            this->_update_L();
            
            this->set_adapt_mu(log(10 * this->_nom_epsilon));
            this->stepsize_adapter::init();
          }
          
        }
        
        return s;
        
      }
      
    };
    
  } // mcmc
  
} // stan


#endif
