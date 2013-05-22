#ifndef __STAN__MCMC__ADAPT__UNIT__METRO__HPP
#define __STAN__MCMC__ADAPT__UNIT__METRO__HPP

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/metro/unit_metro.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename M, class BaseRNG>
    class adapt_unit_metro: public unit_metro<M, BaseRNG>,
                            public stepsize_adapter {
      
    public:
      
      adapt_unit_metro(M &m, BaseRNG& rng,
                       std::ostream* o = &std::cout, 
                       std::ostream* e = 0)
        : unit_metro<M, BaseRNG>(m, rng, o, e) {};
      
      ~adapt_unit_metro() {};
      
      sample transition(sample& init_sample) {
        
        sample s = unit_metro<M, BaseRNG>::transition(init_sample);
        
        if (_adapt_flag) {      
          this->_stepsize_adaptation.learn_stepsize(this->_nom_epsilon, 
                                                    s.accept_stat());
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
