#ifndef STAN__MATH__MATRIX__TCROSSPROD_HPP
#define STAN__MATH__MATRIX__TCROSSPROD_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <algorithm>
#include <new>

#include "Eigen/src/Core/Assign.h"
#include "Eigen/src/Core/CwiseNullaryOp.h"
#include "Eigen/src/Core/GeneralProduct.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/MatrixBase.h"
#include "Eigen/src/Core/SelfAdjointView.h"
#include "Eigen/src/Core/Transpose.h"
#include "Eigen/src/Core/products/GeneralMatrixMatrix.h"
#include "Eigen/src/Core/products/SelfadjointProduct.h"
#include "Eigen/src/Core/util/Constants.h"
#include "Eigen/src/Core/util/Memory.h"

namespace stan {
  namespace math {

    /**
     * Returns the result of post-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return M times its transpose.
     */
    inline matrix_d
    tcrossprod(const matrix_d& M) {
        if (M.rows() == 0)
          return matrix_d(0,0);
        if (M.rows() == 1)
          return M * M.transpose();
        matrix_d result(M.rows(),M.rows());
        return result
          .setZero()
          .selfadjointView<Eigen::Upper>()
          .rankUpdate(M);
    }

  }
}
#endif
