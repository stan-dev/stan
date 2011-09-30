#ifndef __STAN__OPTIMIZE__GRADIENT_BASED__H__
#define __STAN__OPTIMIZE__GRADIENT_BASED__H__

#include "stan/mcmc/prob_grad.hpp"

namespace stan {

  namespace optimize {

    namespace {

      double dot(const std::vector<double>& x, const std::vector<double>& y) {
	double result = 0;
	for (unsigned int i = 0; i < x.size(); i++)
	  result += x[i] * y[i];
	return result;
      }

    }


    double gradient_step(std::vector<double>& x,
                         std::vector<unsigned int>& z,
                         stan::mcmc::prob_grad& model,
			 double step_size) {
      std::vector<double> gradient;
      double f_x = model.grad_log_prob(x, z, gradient);
      for (unsigned int k = 0; k < x.size(); ++k)
	x[k] += step_size * gradient[k];
      return f_x;
    }

    double gradient(std::vector<double>& x,
                    std::vector<unsigned int>& z,
                    stan::mcmc::prob_grad& model,
		    double initial_learn_rate,
		    double annealing_rate,
		    int max_steps) {
      double f_x = -1e100;
      for (int m = 0; m < max_steps; ++m) {
	double learn_rate = initial_learn_rate / (1.0 + m / annealing_rate);
	f_x = gradient_step(x,z,model,learn_rate);
      }
      return f_x;
    }

    double line_search(std::vector<double>& x,
                       std::vector<unsigned int>& z,
                       stan::mcmc::prob_grad& model,
		       const std::vector<double>& direction,
		       double alpha=10.0) {
	double searchfactor = 2.0;
	alpha *= searchfactor;
	std::vector<double> last_x = x;
	double last_f_x = -std::numeric_limits<double>::infinity();
	double f_x = -std::numeric_limits<double>::infinity();
	int movement = -1;
	while (!(f_x < last_f_x)) {
	  alpha *= 1.0 / searchfactor;
	  for (unsigned int i = 0; i < x.size(); i++)
	    x[i] = last_x[i] + alpha * direction[i];
	  last_f_x = f_x;
	  f_x = model.log_prob(x, z);
	  //       fprintf(stderr, "linesearch %e --- %f\n", alpha, f_x);
	  movement++;
	}
	alpha *= searchfactor;
	double temp = f_x;
	f_x = last_f_x;
	last_f_x = temp;
	if (movement > 1)
	  return alpha;

	while (!(f_x < last_f_x)) {
	  alpha *= searchfactor;
	  for (unsigned int i = 0; i < x.size(); i++)
	    x[i] = last_x[i] + alpha * direction[i];
	  last_f_x = f_x;
	  f_x = model.log_prob(x, z);
	  //       fprintf(stderr, "linesearch %e --- %f\n", alpha, f_x);
	}
	alpha *= 1.0 / searchfactor;
	for (unsigned int i = 0; i < x.size(); i++)
	  x[i] = last_x[i] + alpha * direction[i];

	return alpha;
      }

    double conjugate_gradient(std::vector<double>& x,
                              std::vector<unsigned int>& z,
                              stan::mcmc::prob_grad& model,
			      int niterations) {
	std::vector<double> gradient;
	std::vector<double> lastgradient;
	std::vector<double> direction;
	std::vector<double> lastx;
	double logp = model.grad_log_prob(x, z, gradient);
	double alpha = 10;
	direction = gradient;
	lastgradient = gradient;
	for (int i = 0; i < niterations; i++) {
	  lastx = x;
	  alpha = line_search(x, z, model, direction, alpha*2);
	  lastgradient = gradient;
	  logp = model.grad_log_prob(x, z, gradient);

	  // Fletcher-Reeves
	  // double beta = dot(gradient, gradient) / dot(lastgradient, lastgradient);

	  // Polak-Ribere+
	  double beta = 0;
	  for (unsigned int j = 0; j < gradient.size(); j++)
	    beta += gradient[j] * (gradient[j] - lastgradient[j]);
	  beta /= dot(lastgradient, lastgradient);
	  beta = beta < 0 ? 0 : beta;

	  //       fprintf(stderr, "beta = %f\n", beta);
	  for (unsigned int j = 0; j < gradient.size(); j++)
	    direction[j] = gradient[j] + beta * direction[j];

	  fprintf(stderr, "%d:  logp = %f\n", i, logp);

// 	  if (i % 50 == 0) {
// 	    fprintf(stderr, "gradient:\n");
// 	    for (unsigned int j = 0; j < gradient.size(); j++)
// 	      fprintf(stderr, "%f ", gradient[j]);
// 	    fprintf(stderr, "\n");
// 	    fprintf(stderr, "x:\n");
// 	    for (unsigned int j = 0; j < gradient.size(); j++)
// 	      fprintf(stderr, "%f ", x[j]);
// 	    fprintf(stderr, "\n");
// 	    testGradients(x, z, 1e-7);
// 	  }
	}
	return logp;
      }



  }
}
 
#endif
