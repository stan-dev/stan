#ifndef STAN_MODEL_HESSIAN_TIMES_VECTOR_HPP
#define STAN_MODEL_HESSIAN_TIMES_VECTOR_HPP

#include <stan/model/model_functional.hpp>
#include <stan/math/mix/mat.hpp>
#include <iostream>

namespace stan {
  namespace model {

    template <class M>
    void hessian_times_vector(const M& model,
                              const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
                              const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
                              double& f,
                              Eigen::Matrix<double, Eigen::Dynamic, 1>&
                              hess_f_dot_v,
                              std::ostream* msgs = 0) {
      stan::math::hessian_times_vector(model_functional<M>(model, msgs),
                                       x, v, f, hess_f_dot_v);
    }

    /**
     * Compute the product between the Hessian and a vector using a combination
     * of forward-mode and reverse-mode automatic differentiation, writing the
     * result into the specified hess_f_dot_vec variable.
     *
     * @tparam propto True if calculation is up to proportion
     * (double-only terms dropped).
     * @tparam jacobian_adjust_transform True if the log absolute
     * Jacobian determinant of inverse parameter transforms is added to
     * the log probability.
     * @tparam M Class of model.
     * @param[in] model Model.
     * @param[in] params_r Real-valued parameters.
     * @param[in] vec Vector to be multiplied by the Hessian.
     * @param[out] f Double into which funciton value is written.
     * @param[out] hess_f_dot_vec Vector into which hessian-vector product is
     * written.
     * @param[in,out] msgs
     */
    template <bool propto, bool jacobian_adjust_transform, class M>
    void log_prob_hessian_times_vector(
        const M& model,
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& vec,
        double& f,
        Eigen::Matrix<double, Eigen::Dynamic, 1>& hess_f_dot_vec,
        std::ostream* msgs = 0) {
      stan::math::hessian_times_vector<
        model_functional_template<propto, jacobian_adjust_transform, M>>(
          model_functional_template<propto, jacobian_adjust_transform, M>(model, msgs),
              params_r, vec, f, hess_f_dot_vec);
    }

  }
}
#endif
