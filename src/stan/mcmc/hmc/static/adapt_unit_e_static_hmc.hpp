#ifndef STAN__MCMC__ADAPT__UNIT__E__STATIC__HMC__BETA
#define STAN__MCMC__ADAPT__UNIT__E__STATIC__HMC__BETA

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/hmc/static/unit_e_static_hmc.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Hamiltonian Monte Carlo on a 
    // Euclidean manifold with unit metric,
    // static integration time,
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_unit_e_static_hmc: public unit_e_static_hmc<M, BaseRNG>,
                                   public stepsize_adapter {
      
    public:
      
      adapt_unit_e_static_hmc(M &m, BaseRNG& rng,
                              std::ostream* o = &std::cout, std::ostream* e = 0):
      unit_e_static_hmc<M, BaseRNG>(m, rng, o, e) {};
      
      ~adapt_unit_e_static_hmc() {};
      
      sample transition(sample& init_sample) {
        
        sample s = unit_e_static_hmc<M, BaseRNG>::transition(init_sample);
        
        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_, s.accept_stat());
          this->update_L_();
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
