#ifndef __STAN__MCMC__PROB_GRAD_AD_H__
#define __STAN__MCMC__PROB_GRAD_AD_H__

#include <vector>
#include "stan/agrad/agrad.hpp"
#include "stan/mcmc/prob_grad.hpp"

namespace stan {

  namespace mcmc {

    class prob_grad_ad : public prob_grad {
    public:

      prob_grad_ad(unsigned int num_params_r)
	: prob_grad::prob_grad(num_params_r) { 
      }

      prob_grad_ad(unsigned int num_params_r,
		   std::vector<int>& param_ranges_i)
	: prob_grad::prob_grad(num_params_r,
			       param_ranges_i) {
      }

      virtual ~prob_grad_ad() {
      }

      virtual agrad::var log_prob(std::vector<agrad::var>& params_r, 
				  std::vector<int>& params_i) = 0;

      virtual double grad_log_prob(std::vector<double>& params_r, 
				   std::vector<int>& params_i, 
				   std::vector<double>& gradient) {
	std::vector<agrad::var> ad_params_r;
	for (unsigned int i = 0; i < num_params_r(); ++i) {
	  agrad::var var_i(params_r[i]);
	  ad_params_r.push_back(var_i);
	}
	agrad::var adLogProb = log_prob(ad_params_r,params_i);
	double val = adLogProb.val();
	adLogProb.grad(ad_params_r,gradient);
	return val;
      }

      virtual double log_prob(std::vector<double>& params_r,
			      std::vector<int>& params_i) {
	std::vector<agrad::var> ad_params_r;
	for (unsigned int i = 0; i < num_params_r(); ++i) {
	  agrad::var var_i(params_r[i]);
	  ad_params_r.push_back(var_i);
	}
	agrad::var adLogProb = log_prob(ad_params_r,params_i);
	double val = adLogProb.val();
	agrad::vari::recover_memory();
	return val;
      }
    
    };
    
  }
}

#endif

