#ifndef __STAN__AGRAD__FWD__MATRIX__DIVIDE_HPP__
#define __STAN__AGRAD__FWD__MATRIX__DIVIDE_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/fwd/operator_division.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    template <typename T1, typename T2>
    inline 
    typename stan::return_type<T1,T2>::type
    divide(const T1& v, const T2& c) {
      return v / c;
    }

    template <typename T1, typename T2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    divide(const fvar<T1>& v, const fvar<T2>& c) {
      return to_fvar(v) / to_fvar(c);
    }

    template <typename T1, typename T2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    divide(const T1& v, const fvar<T2>& c) {
      return to_fvar(v) / c;
    }

    template <typename T1, typename T2>
    inline 
    fvar<typename stan::return_type<T1,T2>::type>
    divide(const fvar<T1>& v, const T2& c) {
      return v / to_fvar(c);
    }

    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C>
    divide(const Eigen::Matrix<fvar<T1>, R,C>& v, const fvar<T2>& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C> res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = v(i,j) / c;
      }
      return res;
    }

    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C>
    divide(const Eigen::Matrix<fvar<T1>, R,C>& v, const T2& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C> res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = v(i,j) / c;
      }
      return res;
    }

    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C>
    divide(const Eigen::Matrix<T1, R,C>& v, const fvar<T2>& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C> res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = to_fvar(v(i,j)) / c;
      }
      return res;
    }

    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C>
    operator/(const Eigen::Matrix<fvar<T1>, R,C>& v, const fvar<T2>& c) {
      return divide(v,c);
    }

    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C>
    operator/(const Eigen::Matrix<fvar<T1>, R,C>& v, const T2& c) {
      return divide(v,c);
    }

    template <typename T1, typename T2, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R,C>
    operator/(const Eigen::Matrix<T1, R,C>& v, const fvar<T2>& c) {
      return divide(v,c);
    }
  }
}
#endif
