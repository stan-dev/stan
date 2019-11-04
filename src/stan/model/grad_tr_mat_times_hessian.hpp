#ifndef STAN_MODEL_GRAD_TR_MAT_TIMES_HESSIAN_HPP
#define STAN_MODEL_GRAD_TR_MAT_TIMES_HESSIAN_HPP

#include <stan/model/model_functional.hpp>
#include <stan/math/mix/mat.hpp>
#include <stan/math/prim/meta.hpp>

#include <ostream>
#include <type_traits>
#include <utility>

namespace stan {
namespace model {

template <typename M, typename Vecx, typename MatX, typename VecGrad
 require_all_vector_vt<std::is_floating_point, VecX, VecGrad>...,
 require_eigen_vt<std::is_floating_point, MatX>...>
void grad_tr_mat_times_hessian(M&& model, Vecx& x, VecX&& X,
  VecGrad&& grad_tr_X_hess_f, std::ostream* msgs = 0) {
  stan::math::grad_tr_mat_times_hessian(
   model_functional<M>(std::forward<M>(model), msgs),
   std::forward<VecX>(x), std::forward<MatX>(X),
   std::forward<VecGrad>(grad_tr_X_hess_f));
}

}  // namespace model
}  // namespace stan
#endif
