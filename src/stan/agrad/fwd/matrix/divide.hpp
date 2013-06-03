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

    template <typename T>
    inline 
    fvar<T>
    divide(const fvar<T>& v, const fvar<T>& c) {
      return to_fvar(v) / to_fvar(c);
    }

    template <typename T>
    inline 
    fvar<typename stan::return_type<T,double>::type>
    divide(double v, const fvar<T>& c) {
      return to_fvar(v) / c;
    }

    template <typename T>
    inline 
    fvar<typename stan::return_type<T,double>::type>
    divide(const fvar<T>& v, double c) {
      return v / to_fvar(c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>,R,C>
    divide(const Eigen::Matrix<fvar<T>, R,C>& v, const fvar<T>& c) {
      Eigen::Matrix<fvar<T>,R,C> res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = v(i,j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T,double>::type>,R,C>
    divide(const Eigen::Matrix<fvar<T>, R,C>& v, double c) {
      Eigen::Matrix<fvar<typename stan::return_type<T,double>::type>,R,C> 
        res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = v(i,j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T,double>::type>,R,C>
    divide(const Eigen::Matrix<double, R,C>& v, const fvar<T>& c) {
      Eigen::Matrix<fvar<typename stan::return_type<T,double>::type>,R,C> 
        res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = to_fvar(v(i,j)) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>,R,C>
    operator/(const Eigen::Matrix<fvar<T>, R,C>& v, const fvar<T>& c) {
      return divide(v,c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T,double>::type>,R,C>
    operator/(const Eigen::Matrix<fvar<T>, R,C>& v, double c) {
      return divide(v,c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<typename stan::return_type<T,double>::type>,R,C>
    operator/(const Eigen::Matrix<double, R,C>& v, const fvar<T>& c) {
      return divide(v,c);
    }
  }
}
#endif
