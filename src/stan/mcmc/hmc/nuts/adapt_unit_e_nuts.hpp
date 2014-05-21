#ifndef __STAN__MCMC__ADAPT__UNIT__E__NUTS__BETA__
#define __STAN__MCMC__ADAPT__UNIT__E__NUTS__BETA__

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>

namespace stan {
  
  namespace mcmc {
    
    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with unit metric
    // and adaptive stepsize
    
    template <typename M, class BaseRNG>
    class adapt_unit_e_nuts: public unit_e_nuts<M, BaseRNG>,
                             public stepsize_adapter {
      
    public:
      
      adapt_unit_e_nuts(M &m, BaseRNG& rng,
                        std::ostream* o = &std::cout, std::ostream* e = 0):
      unit_e_nuts<M, BaseRNG>(m, rng, o, e) {};
      
      ~adapt_unit_e_nuts() {};
      
      sample transition(sample& init_sample) {
        
        sample s = unit_e_nuts<M, BaseRNG>::transition(init_sample);
        
        if (this->_adapt_flag)
          this->_stepsize_adaptation.learn_stepsize(this->_nom_epsilon, s.accept_stat());
        
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
