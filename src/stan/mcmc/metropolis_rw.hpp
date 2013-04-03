#ifndef __STAN__MCMC__METROPOLIS_RW__
#define __STAN__MCMC__METROPOLIS_RW__

#include <iostream>
#include <vector>
#include <algorithm>

#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace stan {
  namespace mcmc {

    template <class BaseRNG = boost::mt19937>
    class metropolis : public adaptive_sampler {
    private:

      std::vector<double> params_r;
      std::vector<int> params_i;
      double log_prob;
      stan::model::prob_grad *_model;
      BaseRNG base_rng;
      double epsilon;

    public:
    metropolis(stan::model::prob_grad& model,
	       const std::vector<double>& par_r,
	       const std::vector<int>& par_i,
	       bool adapt,
	       double eps = 1,
	       std::ostream* error_msgs = 0,
	       std::ostream* output_msgs = 0)
      : adaptive_sampler(adapt, error_msgs, output_msgs),
	params_r(par_r.size()), params_i(par_i.size()), _model(&model), base_rng(BaseRNG(std::time(0))), epsilon(eps) {
      set_params(par_r, par_i);
    }

      /**
       * Destructor.
       *
       * The implementation for this class is a no-op.
       */
      ~metropolis() { }
  
      void set_params( const std::vector<double>& par_r,
		       const std::vector<int>& par_i) {
	if(par_r.size() != params_r.size())
	  throw std::invalid_argument("metropolis::set_params double params must match in size");
	if(par_i.size() != params_i.size())
	  throw std::invalid_argument("metropolis::set_params int params must match in size");
	params_r = par_r;
	params_i = par_i;

	log_prob = _model->log_prob(params_r, params_i, _error_msgs);
      }

      void write_adaptation_params(std::ostream& /*o*/) {
      }

      /**
       * Return the next sample.
       *
       * @return The next sample.
       */
      sample next_impl() {
      
      std::vector<double> new_params_r(params_r.size());

      for(size_t i = 0; i < new_params_r.size(); i++)
	new_params_r[i] = stan::prob::normal_rng(params_r[i], epsilon, base_rng);

      if (!this->adapting()) { 
	epsilon = this->mean_stat();
      }

      double new_log_prob = _model->log_prob(new_params_r, params_i, _error_msgs);

      double adapt_stat = std::min(std::exp(new_log_prob - log_prob), 1.0);

      double u = stan::prob::uniform_rng(0,1,base_rng);
    	
      if(u < adapt_stat)
	{
	  params_r = new_params_r;
	  log_prob = new_log_prob;  
	}

        double avg_eta = 1.0 / this->n_steps();
        this->update_mean_stat(avg_eta,adapt_stat);

	return mcmc::sample (params_r, params_i, log_prob);
    }
    };
  }
}
#endif
