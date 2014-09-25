#ifndef STAN__MATH__MATRIX__PROMOTE_SCALAR_HPP
#define STAN__MATH__MATRIX__PROMOTE_SCALAR_HPP

#include <stan/math/functions/promote_scalar.hpp>
#include <stan/math/matrix/promote_scalar_type.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    /**
     * Return the matrix consisting of the recursive promotion
     * of the elements of the input matrix to the scalar type
     * specified by the return template parameter.
     *
     * @tparam T scalar return type.
     * @param S element type of input matrix.
     * @param x input standard vector.
     * @return matrix with values promoted from input vector.
     */
    template <typename T, typename S>
    struct promote_scalar_struct<T, Eigen::Matrix<S,-1,-1> > {
      static 
      Eigen::Matrix<typename promote_scalar_type<T,S>::type, -1,-1>
      apply(const Eigen::Matrix<S, -1,-1>& x) {
        Eigen::Matrix<typename promote_scalar_type<T,S>::type, -1,-1>
          y(x.rows(), x.cols());
        for (size_t i = 0; i < x.size(); ++i)
          y(i) = promote_scalar_struct<T,S>::apply(x(i));
        return y;
      }
    };


    template <typename T, typename S>
    struct promote_scalar_struct<T, Eigen::Matrix<S,1,-1> > {
      static 
      Eigen::Matrix<typename promote_scalar_type<T,S>::type, 1,-1>
      apply(const Eigen::Matrix<S, 1,-1>& x) {
        Eigen::Matrix<typename promote_scalar_type<T,S>::type, 1,-1>
          y(x.rows(), x.cols());
        for (size_t i = 0; i < x.size(); ++i)
          y(i) = promote_scalar_struct<T,S>::apply(x(i));
        return y;
      }
    };

    template <typename T, typename S>
    struct promote_scalar_struct<T, Eigen::Matrix<S,-1,1> > {
      static
      Eigen::Matrix<typename promote_scalar_type<T,S>::type, -1,1>
      apply(const Eigen::Matrix<S, -1,1>& x) {
        Eigen::Matrix<typename promote_scalar_type<T,S>::type, -1,1>
          y(x.rows(), x.cols());
        for (size_t i = 0; i < x.size(); ++i)
          y(i) = promote_scalar_struct<T,S>::apply(x(i));
        return y;
      }
    };



  }
}


#endif




