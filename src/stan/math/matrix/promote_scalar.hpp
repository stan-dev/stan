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
    template <typename S, typename T>
    struct promote_scalar_struct {
      static T apply(S x) {
        return T(x);
      }
    };


    template <typename T>
    struct promote_scalar_struct<T,T> {
      static T apply(const T& x) {
        return x;
      }
    };



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
    // template <typename S, typename T>
    // struct promote_scalar_struct<S,typename boost::enable_if<boost::is_same<S,T>, T>::type> {
    //   static T apply(const S& x) {
    //     return x;
    //   }
    // };
    

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
    struct promote_scalar_struct<T, std::vector<S> > {

      static 
      std::vector<typename promote_scalar_type<T,S>::type>
      apply(const std::vector<S>& x) {
        std::vector<typename promote_scalar_type<T,S>::type> y(x.size());
        for (size_t i = 0; i < x.size(); ++i)
          y[i] = promote_scalar_struct<T,S>::apply(x[i]);
        return y;
      }

    };

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


    template <typename T, typename S>
    typename promote_scalar_type<T,S>::type
    promote_scalar(const S& x) {
      return promote_scalar_struct<T,S>::apply(x);
    }



  }
}


#endif




