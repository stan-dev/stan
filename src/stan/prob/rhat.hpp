#ifndef __STAN__PROB__RHAT_HPP__
#define __STAN__PROB__RHAT_HPP__

#include <stdexcept>
#include <boost/throw_exception.hpp>
#include "stan/prob/online_avg.hpp"

namespace stan {

  namespace prob {

    class rhat {
    public:
      /**
       *
       * @throw std::invalid_argument if num_chains or num_params are equal to or less
       *   than 0.
       */
      rhat(unsigned int num_chains, unsigned int num_params) 
	: _avgs(num_chains) {
	if (num_chains <= 0) 
	  BOOST_THROW_EXCEPTION(std::invalid_argument ("num_chains must be greater than 0"));
	if (num_params <= 0) 
	  BOOST_THROW_EXCEPTION(std::invalid_argument ("num_params must be greater than 0"));
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
      
      /**
       *
       * @throw std::out_of_range if j is greater than or equal to num_chains()
       */
      void add(unsigned int j, std::vector<double>& theta) {
	if ((int)j >= num_chains()) 
	  BOOST_THROW_EXCEPTION(std::out_of_range ("j must be less than num_chains()"));
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
