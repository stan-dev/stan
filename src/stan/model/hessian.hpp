#ifndef STAN_MODEL_HESSIAN_HPP
#define STAN_MODEL_HESSIAN_HPP

#include <stan/math/mix/mat.hpp>
#include <stan/model/model_functional.hpp>
#include <iostream>

namespace stan {
  namespace model {
    template <class M>
    void hessian(const M& model,
                 const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                 double& f,
                 Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& hess_f,
                 std::ostream* msgs = 0) {
      stan::math::hessian<model_functional<M> >(model_functional<M>(model,
                                                                    msgs),
                                                x, f, grad_f, hess_f);
    }

    /**
     * Compute the gradient and hessian using a combination of reverse-mode
     * and forward-mode automatic differentiation, writing the result into the
     * specified grad_f and hess_f variables.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[out] f Double into which funciton value is written.
     * @param[out] grad_f Vector into which gradient is written.
     * @param[out] hess_f Vector into which hessian is written.
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    void log_prob_hessian(
        const M& model,
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
        double& f,
        Eigen::Matrix<double, Eigen::Dynamic, 1>& grad_f,
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& hess_f,
        std::ostream* msgs = 0) {
      stan::math::hessian<model_functional_template<propto, jacobian_adjust_transform, M>>(
          model_functional_template<propto, jacobian_adjust_transform, M>(model, msgs),
              params_r, f, grad_f, hess_f);
    }

  }
}
#endif
