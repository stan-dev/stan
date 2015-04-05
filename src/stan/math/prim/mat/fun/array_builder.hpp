#ifndef STAN_MATH_PRIM_MAT_FUN_ARRAY_BUILDER_HPP
#define STAN_MATH_PRIM_MAT_FUN_ARRAY_BUILDER_HPP

#include <stan/math/prim/mat/fun/promoter.hpp>
#include <vector>

namespace stan {
  namespace math {

    /**
     * Structure for building up arrays in an expression (rather than
     * in statements) using an argumentchaining add() method and
     * a getter method array() to return the result.
     */
    template <typename T>
    struct array_builder {
      std::vector<T> x_;
      array_builder() : x_() { }
      template <typename F>
      array_builder& add(const F& u) {
        T t;
        promoter<F, T>::promote(u, t);
        x_.push_back(t);
        return *this;
      }
      std::vector<T> array() {
        return x_;
      }
    };

  }
}
#endif
