#ifndef __STAN__DIFF__FWD__MATRIX__DOT_SELF_HPP__
#define __STAN__DIFF__FWD__MATRIX__DOT_SELF_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/matrix/dot_product.hpp>

namespace stan {
  namespace diff {

    template<typename T, int R, int C>
    inline fvar<T>
    dot_self(const Eigen::Matrix<fvar<T>, R, C>& v) {
      stan::math::validate_vector(v,"dot_self");
      return dot_product(v, v);
    }
  }
}
#endif
