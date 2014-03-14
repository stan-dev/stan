#ifndef __STAN__MATH__MATRIX__SCALE_HPP__
#define __STAN__MATH__MATRIX__SCALE_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/validate_equal.hpp>

namespace stan {
  namespace math {
    
    using Eigen::Dynamic;
    using Eigen::Matrix;
    using boost::math::tools::promote_args;
    
    template <typename T1, typename T2, int R, int C>
    inline Matrix
    <typename promote_args<T1,T2>::type, Dynamic, Dynamic>
    scale(const Matrix<T1, Dynamic, Dynamic>& mat,
          const Matrix<T2, R, C>& vec) {
      stan::math::validate_square(mat, "scale");
      int size = vec.size();
      stan::math::validate_equal(mat.rows(), size, "matrix size",
                                 "vector size", "scale");
      Matrix<typename promote_args<T1,T2>::type, Dynamic, Dynamic>
        result(size, size);
      typename promote_args<T1,T2>::type * datap_result =
        result.data();
      const T1 * datap_mat =
        mat.data();
      for (int i = 0, ij = 0; i < size; ++i)
        for (int j = 0; j < size; ++j, ij++)
          datap_result[ij] = datap_mat[ij]*vec(i)*vec(j);
      return result;
    }

  }
}
#endif
