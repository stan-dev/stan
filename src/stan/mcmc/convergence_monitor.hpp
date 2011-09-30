#ifndef MCMC_CONVERGENCE_MONITOR_H
#define MCMC_CONVERGENCE_MONITOR_H

import "mcmc/sample.h"

#include <queue>
#include <vector>

namespace stan {

  namespace mcmc {

    /**
     * Monitors convergence of multiple Markov chains using R-hat and
     * sample autocorrelation statistics.  Updates are online and only
     * the second half of each chain is saved.
     */
    class convergence_monitor {

    private:
      std::vector<unsigned int> _total_sample_counts;
      std::vector<std::queue<sample> > _samples;
      std::vector<online_avg> _avgs;
      std::vector<online_avg> _lag_avgs;

    public:

      convergence_monitor(unsigned int num_chains, 
			  unsigned int num_params)
	: _total_sample_counts(num_chains,0),
	  _samples(num_chains),
	  _avgs(num_chains),
	  _lag_avgs(num_chains) {
	for (int m = 0; m < num_chains; ++m) {

	  online_avg avgs(num_params); // move out of loop? into constructor?
	  _avgs[m] = avgs;

	  online_avg lag_avgs(num_params);
	  _lag_avgs[m] = lag_avgs;

	  queue<sample> q(); // redundant? 
	  _samples[m] = q;
	}
      }

      int num_params() {
	return _avgs[0].num_dimensions();
      }

      int num_chains() {
	return _avgs.size();
      }
  
      void add(int chain, sample sample) {
	std::vector<double> params_r = sample.params_r();

	// remove stale sample
	if ((_total_sample_counts[chain] % 2) == 1) {
	  std::vector<double> removed_params_r = _samples[chain].first();
	  _samples[chain].pop();                 // remove from samples
	  _avgs[chain].remove(removed_params_r); // remove from avg
	  if (_samples[chain].size() > 0) {
	    std::vector<double> prod = _samples[chain].first();
	    for (int k = 0; k < num_params(); ++k)
	      prod[k] *= removed_params_r[k];
	    _lag_avgs.remove(prod);               // remove lag prod
	  }
	}

	// add lag prod
	if (_samples[chain].size() > 0) {
	  std::vector<double> prod = _samples[chain].last();
	  for (int k = 0; k < num_params(); ++k)
	    prod[k] *= params_r[k];
	  _lag_avgs.add(prod);
	}

	// add sample
	_avgs[chain].add(params_r);

	++_total_samples[chain];
      }

      double auto_correlation(int chain, int k) {
	double sigma_hat_squared = _avgs[chain].sample_variance(k);
	double mu = _avgs[chain].avg(k);
	double prod_avg = _auto_corr_prod_avgs[chain].avg(k);
	return (prod_avg - mu * mu) / sigma_hat_squared;
      }

      double rhat(int k) {
	int m = num_chains();
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
	return sqrt(var_hat_plus_j / W);
      }

    }

  }

}
#endif
