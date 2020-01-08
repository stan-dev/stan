#ifndef STAN_MODEL_HESSIAN_HPP
#define STAN_MODEL_HESSIAN_HPP

#include <stan/math/mix.hpp>
#include <stan/model/model_functional.hpp>
#include <iostream>

namespace stan {
namespace model {

template <class M, typename VecX, typename VecGrad, typename MatHess,
          require_all_vector_like_vt<std::is_arithmetic, VecX, VecGrad>...,
          require_eigen_vt<std::is_arithmetic, MatHess>...>
void hessian(const M& model, VecX&& x, double& f, VecGrad&& grad_f,
             MatHess&& hess_f, std::ostream* msgs = 0) {
  stan::math::hessian<model_functional<M> >(
      model_functional<M>(model, msgs), std::forward<VecX>(x), f,
      std::forward<VecGrad>(grad_f), std::forward<MatHess>(hess_f));
}

}  // namespace model
}  // namespace stan
#endif
