#ifndef STAN__AGRAD__FWD__MATRIX__DOT_SELF_HPP
#define STAN__AGRAD__FWD__MATRIX__DOT_SELF_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_vector.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/dot_product.hpp>

namespace stan {
  namespace agrad {

    template<typename T, int R, int C>
    inline fvar<T>
    dot_self(const Eigen::Matrix<fvar<T>, R, C>& v) {
      stan::error_handling::check_vector("dot_self",
                                         "v", v);
      return dot_product(v, v);
    }
  }
}
#endif
