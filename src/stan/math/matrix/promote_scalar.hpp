#ifndef STAN__MATH__MATRIX__PROMOTE_SCALAR_HPP
#define STAN__MATH__MATRIX__PROMOTE_SCALAR_HPP

#include <vector>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/promote_scalar_type.hpp>

namespace stan {

  namespace math {

    /**
     * Return the value of the input argument promoted to the type
     * specified by the template parameter.  
     *
     * This is the base case for mismatching template parameter types.
     * There are also more specific overloads for matching template
     * parameter types, std::vector and Eigen::Matrix.
     *
     * This version will only work if the input type is assignable to
     * the output type.
     *
     * @tparam T return type.
     * @tparam S input type.
     * @param x input.
     * @return input promoted to return type.
     */
    template <typename T, typename S>
    inline 
    typename boost::disable_if<boost::is_same<S,T>, T>::type
    promote_scalar(S x) {
      return T(x);
    }

    /**
     * Return the value of the input argument promoted to the type
     * specified by the template parameter.  
     *
     * This is a special case which is enabled only if the two
     * template parameters are the same. 
     *
     * @tparam T return type
     * @tparam S input type
     * @param x input.
     * @return input returned without modification.
     */
    template <typename T, typename S>
    inline 
    typename boost::enable_if<boost::is_same<S,T>, T>::type
    promote_scalar(const S& x) {
      return x;
    }

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
    template <typename T, typename S, int R, int C>
    inline
    Eigen::Matrix<typename promote_scalar_type<T,S>::type, R, C>
    promote_scalar(const Eigen::Matrix<S,R,C>& x) {
      Eigen::Matrix<typename promote_scalar_type<T,S>::type, R, C> 
        y(x.rows(), x.cols());
      for (size_t i = 0; i < x.size(); ++i)
        y(i) = promote_scalar<T>(x(i));
      return y;
    }

    /**
     * Return the standard vector consisting of the recursive
     * promotion of the elements of the input standard vector to the
     * scalar type specified by the return template parameter.
     *
     * @tparam T scalar return type.
     * @param S element type of input vector.
     * @param x input standard vector.
     * @return standard vector with values promoted from input vector.
     */
    template <typename T, typename S>
    inline
    std::vector<typename promote_scalar_type<T,S>::type>
    promote_scalar(const std::vector<S>& x) {
      std::vector<typename promote_scalar_type<T,S>::type> y(x.size());
      for (size_t i = 0; i < x.size(); ++i)
        y[i] = promote_scalar<T>(x[i]);
      return y;
    }


  }
}


#endif




