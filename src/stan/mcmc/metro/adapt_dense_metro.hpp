#ifndef __STAN__MCMC__ADAPT__DENSE__METRO__HPP
#define __STAN__MCMC__ADAPT__DENSE__METRO__HPP

#include <stan/mcmc/stepsize_covar_adapter.hpp>
#include <stan/mcmc/metro/dense_metro.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename M, class BaseRNG>
    class adapt_dense_metro: public dense_metro<M, BaseRNG>,
                             public stepsize_covar_adapter {
      
    public:
      
      adapt_dense_metro(M &m, BaseRNG& rng, int max_adapt,
                        std::ostream* o = &std::cout, 
                        std::ostream* e = 0)
        : dense_metro<M, BaseRNG>(m, rng, o, e),
          stepsize_covar_adapter(m.num_params_r(), max_adapt)
      {};
      
      ~adapt_dense_metro() {};
      
      sample transition(sample& init_sample) {

        sample s = dense_metro<M, BaseRNG>::transition(init_sample);
        
        if (_adapt_flag) {
          
          this->_stepsize_adaptation.learn_stepsize(this->_nom_epsilon, 
                                                    s.accept_stat());
          
          bool update = this->_covar_adaptation.learn_covariance(this->_prop_cov, 
                                                                 this->_params_r);

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
