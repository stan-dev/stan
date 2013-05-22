#ifndef __STAN__MCMC__UNIT__METRO__HPP
#define __STAN__MCMC__UNIT__METRO__HPP

#include <stan/mcmc/metro/base_metro.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

namespace stan {

  namespace mcmc {
    
    template <class M, class BaseRNG>
    class unit_metro: public base_metro<M, BaseRNG> {
      
    public:
      
      unit_metro(M& m, BaseRNG& rng,
                 std::ostream* o = &std::cout, 
                 std::ostream* e = 0)
        : base_metro<M, BaseRNG>(m, rng, o, e)
      { this->_name = "Metropolis with a unit metric"; }

      void propose(std::vector<double>& q,
                    BaseRNG& rng) {
        for (size_t i = 0; i < q.size(); ++i) 
          q[i] = stan::prob::normal_rng(q[i],this->_nom_epsilon,this->_rand_int);

        try {
          this->_log_prob = this->log_prob(q, this->_params_i);
        } catch (std::domain_error e) {
          this->_write_error_msg(this->_err_stream, e);
          this->_log_prob = std::numeric_limits<double>::infinity();
        }

        std::cout<<"step_size:"<<this->_nom_epsilon<<std::endl;
      }

      void write_metric(std::ostream& o) {
        o << "# No free parameters for unit metric" << std::endl;
      };
                        
    };

  } // mcmc

} // stan
          

#endif
