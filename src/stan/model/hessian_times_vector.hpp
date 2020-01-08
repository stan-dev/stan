#ifndef STAN_MODEL_HESSIAN_TIMES_VECTOR_HPP
#define STAN_MODEL_HESSIAN_TIMES_VECTOR_HPP

#include <stan/math/mix.hpp>
#include <stan/model/model_functional.hpp>
#include <ostream>

namespace stan {
namespace model {

template <
    class M, typename VecX, typename VecV, typename VecHess,
    require_all_vector_like_vt<std::is_arithmetic, VecX, VecV, VecHess>...>
void hessian_times_vector(const M& model, VecX&& x, VecV&& v, double& f,
                          VecHess&& hess_f_dot_v, std::ostream* msgs = 0) {
  stan::math::hessian_times_vector(model_functional<M>(model, msgs),
                                   std::forward<VecX>(x), std::forward<VecV>(v),
                                   f, std::forward<VecHess>(hess_f_dot_v));
}

}  // namespace model
}  // namespace stan
#endif
