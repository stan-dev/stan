#ifndef STAN__AGRAD__FWD__MATRIX__DIVIDE_HPP
#define STAN__AGRAD__FWD__MATRIX__DIVIDE_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd.hpp>
#include <stan/agrad/fwd/operators/operator_division.hpp>
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
    inline Eigen::Matrix<fvar<T>,R,C>
    divide(const Eigen::Matrix<fvar<T>, R,C>& v, const double c) {
      Eigen::Matrix<fvar<T>,R,C> 
        res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = v(i,j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>,R,C>
    divide(const Eigen::Matrix<double, R,C>& v, const fvar<T>& c) {
      Eigen::Matrix<fvar<T>,R,C> 
        res(v.rows(),v.cols());
      for(int i = 0; i < v.rows(); i++) {
        for(int j = 0; j < v.cols(); j++)
          res(i,j) = v(i,j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>,R,C>
    operator/(const Eigen::Matrix<fvar<T>, R,C>& v, const fvar<T>& c) {
      return divide(v,c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>,R,C>
    operator/(const Eigen::Matrix<fvar<T>, R,C>& v, const double c) {
      return divide(v,c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>,R,C>
    operator/(const Eigen::Matrix<double,R,C>& v, const fvar<T>& c) {
      return divide(v,c);
    }
  }
}
#endif
