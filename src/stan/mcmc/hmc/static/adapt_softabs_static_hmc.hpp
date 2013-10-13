#ifndef __STAN__MCMC__ADAPT__SOFTABS__STATIC__HMC__BETA__
#define __STAN__MCMC__ADAPT__SOFTABS__STATIC__HMC__BETA__

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/hmc/static/softabs_static_hmc.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a 
    // Riemannian metric with the SoftAbs metric
    // and static integration time
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_softabs_static_hmc: public softabs_static_hmc<M, BaseRNG>,
                                    public stepsize_adapter {
      
    public:
      
      adapt_softabs_static_hmc(M &m, BaseRNG& rng,
                               std::ostream* o = &std::cout, std::ostream* e = 0):
      softabs_static_hmc<M, BaseRNG>(m, rng, o, e) {};

      ~adapt_softabs_static_hmc() {};
      
      sample transition(sample& init_sample) {
        
        sample s = softabs_static_hmc<M, BaseRNG>::transition(init_sample);
        
        if (this->_adapt_flag) {
          this->_stepsize_adaptation.learn_stepsize(this->_nom_epsilon, s.accept_stat());
          this->_update_L();
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
