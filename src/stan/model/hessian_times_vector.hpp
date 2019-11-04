#ifndef STAN_MODEL_HESSIAN_TIMES_VECTOR_HPP
#define STAN_MODEL_HESSIAN_TIMES_VECTOR_HPP

#include <stan/model/model_functional.hpp>
#include <stan/math/mix/mat.hpp>
#include <ostream>

namespace stan {
namespace model {

template <typename M, typename VecX, typename VecV, typename VecHessDot,
 require_all_vector_vt<std::is_floating_point, VecX, VecV, VecHessDot>...>
void hessian_times_vector(M&& model, VecX&& x, VecV&& v, double& f,
    VecHessDot&& hess_f_dot_v, std::ostream* msgs = 0) {
  stan::math::hessian_times_vector(model_functional<M>(std::forward<M>(model), msgs),
   std::forward<VecX>(x), std::forward<VecV>(v), f, std::forward<VecHessDot>(hess_f_dot_v));
}

}  // namespace model
}  // namespace stan
#endif
