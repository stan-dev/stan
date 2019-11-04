#ifndef STAN_MODEL_GRADIENT_HPP
#define STAN_MODEL_GRADIENT_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/rev/mat.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/model/model_functional.hpp>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace stan {
namespace model {

template <typename M, typename VecX, typename VecGrad, require_all_vector_vt<std::is_floating_point, VecX, VecGrad>...>
void gradient(M&& model, VecX&& x, double& f, VecGrad&& grad_f, std::ostream* msgs = 0) {
  stan::math::gradient(model_functional<M>(std::forward<M>(model), msgs),
   std::forward<VecX>(x), f, std::forward<VecGrad>(grad_f));
}

template <typename M, typename VecX, typename VecGrad, require_all_vector_vt<std::is_floating_point, VecX, VecGrad>...>
void gradient(M&& model, VecX&& x, double& f, VecGrad&& grad_f, callbacks::logger& logger) {
  std::stringstream ss;
  try {
    stan::math::gradient(model_functional<M>(std::forward<M>(model), &ss),
     std::forward<VecX>(x), f, std::forward<VecGrad>(grad_f));
  } catch (std::exception& e) {
    if (ss.str().length() > 0)
      logger.info(ss);
    throw;
  }
  if (ss.str().length() > 0)
    logger.info(ss);
}

}  // namespace model
}  // namespace stan
#endif
