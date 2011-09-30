#ifndef __STAN__PROB__RHAT_H__
#define __STAN__PROB__RHAT_H__

#include "stan/prob/online_avg.hpp"

namespace stan {

  namespace prob {

    class rhat {
    public:
      rhat(unsigned int num_chains, unsigned int num_params) 
	: _avgs(num_chains) {
	assert(num_chains > 0);
	assert(num_params > 0);
	for (unsigned int m = 0; m < num_chains; ++m) {
	  online_avg avgs(num_params);  // could move this out if next fully copies
	  _avgs[m] = avgs;
	}
      }

      ~rhat() { }	

      int num_params() {
	return _avgs[0].num_dimensions();
      }

      int num_chains() {
	return _avgs.size();
      }

      void add(unsigned int j, std::vector<double>& theta) {
	assert((int)j < num_chains());
	_avgs[j].add(theta);
	//if (_auto_corr_prod_avgs[j].num_samples() > 0) {
      
	//}
      }

      void compute(std::vector<double>& rhats) {
	int m = num_chains();
	for (int k = 0; k < num_params(); ++k) {
	  double W = 0.0;
	  for (int j = 0; j < m; ++j)
	    W += _avgs[j].sample_variance(k);
	  W /= m;

	  long total_count = 0L;
	  double psi_bar = 0.0;
	  for (int j = 0; j < m; ++j) {
	    total_count += _avgs[j].num_samples();
	    psi_bar += _avgs[j].num_samples() * _avgs[j].avg(k);
	  }
	  psi_bar /= total_count;
    
	  double B = 0.0;
	  for (int j = 0; j < m; ++j) {
	    double diff = _avgs[j].avg(k) - psi_bar;
	    B +=  diff * diff;
	  }
	  // n_avg = n in BDA defn. if all chains same length
	  double n_avg = total_count / (double) num_chains(); 
	  B *= n_avg;
	  B /= m - 1;

	  double var_hat_plus_j = ((n_avg - 1.0) * W + B) / n_avg;
	  rhats[k] = sqrt(var_hat_plus_j / W);
	}
      }

    private: 
      std::vector<online_avg> _avgs;
      //std::vector<online_avg> _auto_corr_prod_avgs;
    };
  

  }

}

#endif
