#ifndef __STAN__MATH__MATRIX__SINGULAR_VALUES_HPP__
#define __STAN__MATH__MATRIX__SINGULAR_VALUES_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/fwd/functions/sqrt.hpp>
#include <stan/agrad/rev/functions/sqrt.hpp>
#include <stan/agrad/fwd/functions/abs.hpp>
#include <stan/agrad/rev/functions/abs.hpp>

namespace stan {
  namespace math {

    /**
     * Return the vector of the singular values of the specified matrix
     * in decreasing order of magnitude.
     * <p>See the documentation for <code>svd()</code> for
     * information on the signular values.
     * @param m Specified matrix.
     * @return Singular values of the matrix.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    singular_values(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      return Eigen::JacobiSVD<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >(m)
        .singularValues();
    }

  }
}
#endif
