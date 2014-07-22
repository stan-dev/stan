#ifndef STAN__MCMC__ADAPT__DIAG__E__STATIC__HMC__BETA
#define STAN__MCMC__ADAPT__DIAG__E__STATIC__HMC__BETA

#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/hmc/static/diag_e_static_hmc.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with diagonal metric,
    // static integration time,
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_diag_e_static_hmc: public diag_e_static_hmc<M, BaseRNG>,
                                   public stepsize_var_adapter {
      
    public:
      
        adapt_diag_e_static_hmc(M &m, BaseRNG& rng,
                                std::ostream* o = &std::cout, std::ostream* e = 0):
        diag_e_static_hmc<M, BaseRNG>(m, rng, o, e),
        stepsize_var_adapter(m.num_params_r())
      {};
      
      ~adapt_diag_e_static_hmc() {};
      
      sample transition(sample& init_sample) {
        
        sample s = diag_e_static_hmc<M, BaseRNG>::transition(init_sample);
        
        if (this->adapt_flag_) {
          
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_, s.accept_stat());
          this->update_L_();
          
          bool update = this->var_adaptation_.learn_variance(this->z_.mInv, this->z_.q);
          
          if(update) {
            this->init_stepsize();
            this->update_L_();
            
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
