#ifndef STAN_MODEL_LOG_PROB_GRAD_HPP
#define STAN_MODEL_LOG_PROB_GRAD_HPP

#include <stan/math/rev/mat.hpp>
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
template <bool propto, bool jacobian_adjust_transform, class M, typename VecR,
          typename VecI, typename VecGrad,
          require_std_vector_vt<std::is_floating_point, VecR>...,
          require_std_vector_vt<is_index, VecI>...,
          require_std_vector_vt<std::is_floating_point, VecGrad>...>
double log_prob_grad(const M& model, VecR&& params_r, VecI&& params_i,
                     VecGrad&& gradient, std::ostream* msgs = 0) {
  using stan::math::var;
  using std::vector;
  double lp;
  try {
    vector<var> ad_params_r(params_r.begin(), params_r.end());
    var adLogProb = model.template log_prob<propto, jacobian_adjust_transform>(
        ad_params_r, params_i, msgs);
    lp = adLogProb.val();
    adLogProb.grad(ad_params_r, gradient);
  } catch (const std::exception& ex) {
    stan::math::recover_memory();
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
template <bool propto, bool jacobian_adjust_transform, class M, typename VecR,
          typename VecGrad,
          require_eigen_vector_vt<std::is_arithmetic, VecR>...,
          require_eigen_vector_vt<std::is_arithmetic, VecGrad>...>
double log_prob_grad(const M& model, VecR&& params_r, VecGrad&& gradient,
                     std::ostream* msgs = 0) {
  using stan::math::var;
  using std::vector;

  auto ad_params_r = params_r.template cast<var>().eval();
  try {
    var adLogProb = model.template log_prob<propto, jacobian_adjust_transform>(
        ad_params_r, msgs);
    double val = adLogProb.val();
    stan::math::grad(adLogProb, ad_params_r, gradient);
    return val;
  } catch (std::exception& ex) {
    stan::math::recover_memory();
    throw;
  }
}

}  // namespace model
}  // namespace stan
#endif
