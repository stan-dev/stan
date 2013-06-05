#ifndef __STAN__MATH__REP_MATRIX_HPP__
#define __STAN__MATH__REP_MATRIX_HPP__

#include <boost/math/tools/promotion.hpp>
#include <stan/math/validate_non_negative_rep.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                  Eigen::Dynamic,Eigen::Dynamic>
    rep_matrix(const T& x, int m, int n) {
      validate_non_negative_rep(m,"rep_matrix rows");
      validate_non_negative_rep(n,"rep_matrix cols");
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           Eigen::Dynamic,Eigen::Dynamic>::Constant(m,n,x);
    }

    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    rep_matrix(const Eigen::Matrix<T,Eigen::Dynamic,1>& v, int n) {
      validate_non_negative_rep(n,"rep_matrix of vector, num rows");
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> result(v.size(),n);
      result.colwise() = v;
      return result;
    }

    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    rep_matrix(const Eigen::Matrix<T,1,Eigen::Dynamic>& rv, int m) {
      validate_non_negative_rep(m,"rep_matrix of row vector, num cols");
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> result(m,rv.size());
      result.rowwise() = rv;
      return result;
    }
  }
}

#endif
