#ifndef __STAN__MCMC__PROB_GRAD_H__
#define __STAN__MCMC__PROB_GRAD_H__

#include <assert.h>
#include <vector>
#include <limits>

namespace stan {

  namespace mcmc {

    /**
     * The <code>prob_grad</code> class represents densities with
     * fixed numbers of discrete and scalar parameters and the
     * gradient with respet to the scalar parameters.
     *
     */
    class prob_grad {
    protected:
      unsigned int num_params_r__;
      std::vector<int> param_ranges_i__;

    public:

      prob_grad(unsigned int num_params_r)
	: num_params_r__(num_params_r),
	  param_ranges_i__(std::vector<int>(0)) {
      }

      prob_grad(unsigned int num_params_r,
		std::vector<int>& param_ranges_i)
	: num_params_r__(num_params_r),
	  param_ranges_i__(param_ranges_i) { 
      }

      virtual ~prob_grad() { }

      void set_num_params_r(unsigned int num_params_r) {
	num_params_r__ = num_params_r;
      }

      void setparam_ranges_i__(std::vector<int> param_ranges_i) {
	param_ranges_i__ = param_ranges_i;
      }

      virtual unsigned int num_params_r() {
	return num_params_r__;
      }

      virtual unsigned int num_params_i() {
	return param_ranges_i__.size();
      }

      inline int param_range_i(unsigned int idx) {
	return param_ranges_i__[idx];
      }

      virtual void init(std::vector<double>& params_r, 
			std::vector<int>& params_i) {
	for (unsigned int i = 0; i < num_params_r(); ++i)
	  params_r[i] = 0.0;
	for (unsigned int j = 0; j < num_params_i(); ++j)
	  params_i[j] = 0;
      }

      virtual double grad_log_prob(std::vector<double>& params_r, 
				   std::vector<int>& params_i,
				   std::vector<double>& gradient) = 0;

      virtual double log_prob(std::vector<double>& params_r, 
			      std::vector<int>& params_i) = 0;

      virtual double log_prob_star(unsigned int idx, 
				   int val,
				   std::vector<double>& params_r,
				   std::vector<int>& params_i) {
	// assert(idx >= 0);
	assert(idx < num_params_i());
	// assert(val >= 0);
	assert(val < param_range_i(idx));

	int original_val = params_i[idx];
	params_i[idx] = val;
	double result = log_prob(params_r,params_i);
	params_i[idx] = original_val;
	return result;
      }


      // A couple of methods for optimizing params_r to approach a MAP estimate.
      double gradientMethod(std::vector<double>& params_r, 
			    std::vector<int>& params_i, 
			    double eta) {
	std::vector<double> gradient;
	double logp = grad_log_prob(params_r, params_i, gradient);
	for (unsigned int i = 0; i < params_r.size(); i++)
	  params_r[i] += eta * gradient[i];
	return logp;
      }

      double FISTA(std::vector<double>& params_r, 
		   std::vector<int>& params_i, 
		   double eta,
		   int niterations) {
	double lasttk = 1;
	double tk = 1;
	double lastlogp = -1e100;
	double logp = -1e100;
	std::vector<double> y(params_r);
	for (int iteration = 0; iteration < niterations; iteration++) {
	  std::vector<double> lastx(params_r);
	  std::vector<double> gradient;
	  lastlogp = logp;
	  logp = grad_log_prob(params_r, params_i, gradient);
	  for (unsigned int i = 0; i < params_r.size(); i++)
	    params_r[i] = y[i] + eta * gradient[i];

	  if (logp < lastlogp) {
	    y = params_r;
	    iteration = 0;
	    continue;
	  }

	  lasttk = tk;
	  tk = 0.5 * (1 + sqrt(1 + 4 * tk * tk));
	  for (unsigned int i = 0; i < params_r.size(); i++)
	    y[i] = params_r[i] + (tk / (lasttk+1)) * (params_r[i] - lastx[i]);
	  fprintf(stderr, "%d:  logp = %f\n", iteration, logp);
	}
	return logp;
      }

      double nesterov(std::vector<double>& params_r, 
		      std::vector<int>& params_i, double eta,
		      int niterations) {
	std::vector<double> y(params_r);
	std::vector<double> gradient;
	double logp;
	for (int i = 0; i < niterations; i++) {
	  logp = grad_log_prob(y, params_i, gradient);
	  double rho = double(i) / (i + 3.0);
	  for (unsigned int k = 0; k < params_r.size(); k++) {
	    double lastxk = params_r[k];
	    params_r[k] = y[k] + eta * gradient[k];
	    y[k] = params_r[k] + rho * (params_r[k] - lastxk);
	  }
	  fprintf(stderr, "Nesterov %d:  %f\n", i, logp);
	}

	return logp;
      }


      // This shouldn't be necessary, but may help one sleep at night.
      void testGradients(std::vector<double>& params_r,
			 std::vector<int>& params_i,
			 double epsilon = 1e-6) {
	std::vector<double> gradient;
	double logp = grad_log_prob(params_r, params_i, gradient);
	std::vector<double> perturbed = params_r;
	for (unsigned int k = 0; k < params_r.size(); k++) {
	  perturbed[k] += epsilon;
	  double logp2 = log_prob(perturbed, params_i);
	  double gradest = (logp2 - logp) / epsilon;
	  fprintf(stderr, "testing gradient[%d]:  %f computed vs. %f estimated (off by %e)\n",
		  k, gradient[k], gradest, gradient[k] - gradest);
	  perturbed[k] = params_r[k];
	}
      }

    };   

  }
  
}

#endif
