#ifndef STAN_MODEL_LOG_PROB_GRAD_HPP
#define STAN_MODEL_LOG_PROB_GRAD_HPP

#include <stan/math/fwd/mat.hpp>
#include <iostream>
#include <vector>

namespace stan {
  namespace model {

    /**
     * Compute the gradient using reverse-mode automatic
     * differentiation, writing the result into the specified
     * gradient, using the specified perturbation.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[in] params_i Integer-valued parameters.
     * @param[out] gradient Vector into which gradient is written.
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    double log_prob_grad(const M& model,
                         std::vector<double>& params_r,
                         std::vector<int>& params_i,
                         std::vector<double>& gradient,
                         std::ostream* msgs = 0) {
      using std::vector;
      using stan::math::fvar;
      double lp;
      gradient.resize(model.num_params_r());
      try {
	for (size_t i = 0; i < model.num_params_r(); ++i) {
	  std::vector<stan::math::fvar<double>> theta(model.num_params_r());
	  for (size_t j = 0; j < model.num_params_r(); ++j)
	    theta[i] = fvar<double>(params_r[i], i == j ? 1.0 : 0.0);
	  fvar<double> log_prob
	    = model.template log_prob<propto, jacobian_adjust_transform>
	    (theta, params_i, msgs);
	  lp = log_prob.val_;
	  gradient[i] = log_prob.d_;
	}
      } catch (const std::exception &ex) {
        throw;
      }
      return lp;
    }

    /**
     * Compute the gradient using reverse-mode automatic
     * differentiation, writing the result into the specified
     * gradient, using the specified perturbation.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[out] gradient Vector into which gradient is written.
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    double log_prob_grad(const M& model,
                         Eigen::VectorXd& params_r,
                         Eigen::VectorXd& gradient,
                         std::ostream* msgs = 0) {
      using std::vector;
      using stan::math::fvar;
      double lp;
      gradient.resize(model.num_params_r());
      try {
	for (size_t i = 0; i < model.num_params_r(); ++i) {
	  std::vector<stan::math::fvar<double>> theta(model.num_params_r());
	  for (size_t j = 0; j < model.num_params_r(); ++j)
	    theta[i] = fvar<double>(params_r[i], i == j ? 1.0 : 0.0);
	  fvar<double> log_prob
	    = model.template log_prob<propto, jacobian_adjust_transform>
	    (theta, msgs);
	  lp = log_prob.val_;
	  gradient[i] = log_prob.d_;
	}
      } catch (const std::exception &ex) {
        throw;
      }
      return lp;
    }

  }
}
#endif
