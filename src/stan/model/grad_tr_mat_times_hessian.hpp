#ifndef STAN_MODEL_GRAD_TR_MAT_TIMES_HESSIAN_HPP
#define STAN_MODEL_GRAD_TR_MAT_TIMES_HESSIAN_HPP

#include <stan/math/mix.hpp>
#include <stan/model/model_functional.hpp>
#include <ostream>

namespace stan {
namespace model {

template <class M, typename VecX, typename MatX, typename VecGrad,
          require_all_vector_like_vt<std::is_arithmetic, VecX, VecGrad>...,
          require_eigen_vt<std::is_arithmetic, MatX>...>
void grad_tr_mat_times_hessian(const M& model, VecX&& x, MatX&& X,
                               VecGrad&& grad_tr_X_hess_f,
                               std::ostream* msgs = 0) {
  stan::math::grad_tr_mat_times_hessian(
      model_functional<M>(model, msgs), std::forward<VecX>(x),
      std::forward<MatX>(X), std::forward<VecGrad>(grad_tr_X_hess_f));
}

}  // namespace model
}  // namespace stan
#endif
