#ifndef STAN__MATH__PRIM__MAT__FUN__ARRAY_BUILDER_HPP
#define STAN__MATH__PRIM__MAT__FUN__ARRAY_BUILDER_HPP

#include <vector>
#include <stan/math/prim/mat/fun/promoter.hpp>

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
        promoter<F,T>::promote(u,t);
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
