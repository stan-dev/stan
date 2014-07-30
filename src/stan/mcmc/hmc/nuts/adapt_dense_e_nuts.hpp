#ifndef STAN__MCMC__ADAPT__DENSE__E__NUTS__BETA
#define STAN__MCMC__ADAPT__DENSE__E__NUTS__BETA

#include <stan/mcmc/stepsize_covar_adapter.hpp>
#include <stan/mcmc/hmc/nuts/dense_e_nuts.hpp>

namespace stan {
  
  namespace mcmc {
    
    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with dense metric
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_dense_e_nuts: public dense_e_nuts<M, BaseRNG>,
                              public stepsize_covar_adapter {
      
    public:
      
        adapt_dense_e_nuts(M &m, BaseRNG& rng,
                           std::ostream* o = &std::cout, std::ostream* e = 0):
        dense_e_nuts<M, BaseRNG>(m, rng, o, e),
        stepsize_covar_adapter(m.num_params_r())
      {};
      
      ~adapt_dense_e_nuts() {};
      
      sample transition(sample& init_sample) {
        
        sample s = dense_e_nuts<M, BaseRNG>::transition(init_sample);
        
        if (this->adapt_flag_) {
          
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_, s.accept_stat());
          
          bool update = this->covar_adaptation_.learn_covariance(this->z_.mInv, this->z_.q);
          
          if(update) {
            this->init_stepsize();
            
            this->stepsize_adaptation_.set_mu(log(10 * this->nom_epsilon_));
            this->stepsize_adaptation_.restart();
          }
          
        }
        
        return s;
        
      }

      void disengage_adaptation() {
        base_adapter::disengage_adaptation();
        this->stepsize_adaptation_.complete_adaptation(this->nom_epsilon_);
      }
                                
    };
    
  } // mcmc
  
} // stan


#endif
