#ifndef __STAN__MODEL__PROB_GRAD_AD_HPP__
#define __STAN__MODEL__PROB_GRAD_AD_HPP__

#include <cstddef>
#include <utility>
#include <vector>

#include <stan/agrad/agrad.hpp>
#include <stan/model/prob_grad.hpp>

namespace stan {

  namespace model {

    class prob_grad_ad : public prob_grad {
    public:

      prob_grad_ad(size_t num_params_r)
        : prob_grad::prob_grad(num_params_r) { 
      }

      prob_grad_ad(size_t num_params_r,
                   std::vector<std::pair<int,int> >& param_ranges_i)
        : prob_grad::prob_grad(num_params_r,
                               param_ranges_i) {
      }

      virtual ~prob_grad_ad() {
      }

      virtual agrad::var log_prob(std::vector<agrad::var>& params_r, 
                                  std::vector<int>& params_i,
                                  std::ostream* output_stream = 0) = 0;

      virtual double grad_log_prob(std::vector<double>& params_r, 
                                   std::vector<int>& params_i, 
                                   std::vector<double>& gradient,
                                   std::ostream* output_stream = 0) {
        std::vector<agrad::var> ad_params_r(num_params_r());
        for (size_t i = 0; i < num_params_r(); ++i) {
          agrad::var var_i(params_r[i]);
          ad_params_r[i] = var_i;
        }
        agrad::var adLogProb;
        try {
          adLogProb = log_prob(ad_params_r,params_i,output_stream);
        }
        catch (std::exception &ex) {
          agrad::recover_memory();
          throw;
        }
        double val = adLogProb.val();
        adLogProb.grad(ad_params_r,gradient);
        return val;
      }

      virtual double log_prob(std::vector<double>& params_r,
                              std::vector<int>& params_i,
                              std::ostream* output_stream = 0) {
        std::vector<agrad::var> ad_params_r;
        for (size_t i = 0; i < num_params_r(); ++i) {
          agrad::var var_i(params_r[i]);
          ad_params_r.push_back(var_i);
        }
        agrad::var adLogProb = log_prob(ad_params_r,params_i,output_stream);
        double val = adLogProb.val();
        agrad::recover_memory();
        return val;
      }
    
    };
    
  }
}

#endif

