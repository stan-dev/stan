#ifndef STAN__MATH__REP_MATRIX_HPP
#define STAN__MATH__REP_MATRIX_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                  Eigen::Dynamic,Eigen::Dynamic>
    rep_matrix(const T& x, int m, int n) {
      using stan::error_handling::check_nonnegative;
      check_nonnegative("rep_matrix", "rows", m);
      check_nonnegative("rep_matrix", "cols", n);
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           Eigen::Dynamic,Eigen::Dynamic>::Constant(m,n,x);
    }

    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    rep_matrix(const Eigen::Matrix<T,Eigen::Dynamic,1>& v, int n) {
      using stan::error_handling::check_nonnegative;
      check_nonnegative("rep_matrix", "rows", n);
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> result(v.size(),n);
      result.colwise() = v;
      return result;
    }

    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    rep_matrix(const Eigen::Matrix<T,1,Eigen::Dynamic>& rv, int m) {
      using stan::error_handling::check_nonnegative;
      check_nonnegative("rep_matrix", "cols", m);
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> result(m,rv.size());
      result.rowwise() = rv;
      return result;
    }
  }
}

#endif
