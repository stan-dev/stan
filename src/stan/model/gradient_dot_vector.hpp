#ifndef STAN_MODEL_GRADIENT_DOT_VECTOR_HPP
#define STAN_MODEL_GRADIENT_DOT_VECTOR_HPP

#include <stan/math/mix/mat.hpp>
#include <stan/model/model_functional.hpp>
#include <iostream>

namespace stan {
namespace model {

template <class M, typename VecX, typename VecV, require_all_vector_like_vt<std::is_arithmetic, VecX, VecV>...>
void gradient_dot_vector(const M& model, VecX&& x, VecV&& v,
                         double& f, double& grad_f_dot_v,
                         std::ostream* msgs = 0) {
  stan::math::gradient_dot_vector(model_functional<M>(model, msgs),
   std::forward<VecX>(x), std::forward<VecV>(v), f, grad_f_dot_v);
}

}  // namespace model
}  // namespace stan
#endif
