#ifndef STAN_MODEL_HESSIAN_HPP
#define STAN_MODEL_HESSIAN_HPP

#include <stan/math/mix/mat.hpp>
#include <stan/model/model_functional.hpp>
#include <iostream>

namespace stan {
namespace model {

template <class M, typename Vec1, typename VecGrad, typename MatHess,
           require_all_eigen_vector_vt<std::is_floating_point, Vec1, VecGrad>...,
           require_all_eigen_vt<std::is_floating_point, MatHess>...>
void hessian(M&& model, Vec1&& x, double& f, VecGrad&& grad_f, MatHess&& hess_f,
             std::ostream* msgs = 0) {
  stan::math::hessian<model_functional<M> >(model_functional<M>(model, msgs),
                                            std::forward<Vec1>(x), f,
                                            std::forward<VecGrad>(grad_f),
                                            std::forward<MatHess>(hess_f));
}

}  // namespace model
}  // namespace stan
#endif
