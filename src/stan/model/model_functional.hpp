#ifndef STAN_MODEL_MODEL_FUNCTIONAL_HPP
#define STAN_MODEL_MODEL_FUNCTIONAL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <iostream>

namespace stan {
namespace model {

// Interface for automatic differentiation of models
template <class M>
struct model_functional {
  const M& model;
  std::ostream* o;

  template <typename Model, require_same_t<M, Model>...>
  model_functional(Model&& m, std::ostream* out) :
   model(std::forward<Model>(m)), o(out) {}

  template <typename Vec, require_vector_t<Vec>...>
  auto operator()(Vec&& x) const {
    // log_prob() requires non-const but doesn't modify its argument
    using vec_value = value_type_t<Vec>;
    return model.template log_prob<true, true, vec_value>(
        const_cast<std::decay_t<Vec>&>(x), o);
  }
};

}  // namespace model
}  // namespace stan
#endif
