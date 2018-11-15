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

    template <bool propto, bool jacobian_adjust_transform, class M>
    void log_prob_hessian_times_vector(
        const M& model,
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& x,
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& v,
        double& f,
        Eigen::Matrix<double, Eigen::Dynamic, 1>& hess_f_dot_v,
        std::ostream* msgs = 0) {
      stan::math::hessian_times_vector<
        model_functional_template<propto, jacobian_adjust_transform, M>>(
          model_functional_template<propto, jacobian_adjust_transform, M>(model, msgs),
              x, v, f, hess_f_dot_v);
    }

  }
}
#endif
