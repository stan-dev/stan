#ifndef STAN_MATH_FWD_MAT_FUN_DIVIDE_HPP
#define STAN_MATH_FWD_MAT_FUN_DIVIDE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <vector>

namespace stan {
  namespace math {

    template <typename T1, typename T2>
    inline
    typename stan::return_type<T1, T2>::type
    divide(const T1& v, const T2& c) {
      return v / c;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>, R, C>
    divide(const Eigen::Matrix<fvar<T>, R, C>& v, const fvar<T>& c) {
      Eigen::Matrix<fvar<T>, R, C> res(v.rows(), v.cols());
      for (int i = 0; i < v.rows(); i++) {
        for (int j = 0; j < v.cols(); j++)
          res(i, j) = v(i, j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>, R, C>
    divide(const Eigen::Matrix<fvar<T>, R, C>& v, const double c) {
      Eigen::Matrix<fvar<T>, R, C>
        res(v.rows(), v.cols());
      for (int i = 0; i < v.rows(); i++) {
        for (int j = 0; j < v.cols(); j++)
          res(i, j) = v(i, j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>, R, C>
    divide(const Eigen::Matrix<double, R, C>& v, const fvar<T>& c) {
      Eigen::Matrix<fvar<T>, R, C>
        res(v.rows(), v.cols());
      for (int i = 0; i < v.rows(); i++) {
        for (int j = 0; j < v.cols(); j++)
          res(i, j) = v(i, j) / c;
      }
      return res;
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>, R, C>
    operator/(const Eigen::Matrix<fvar<T>, R, C>& v, const fvar<T>& c) {
      return divide(v, c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>, R, C>
    operator/(const Eigen::Matrix<fvar<T>, R, C>& v, const double c) {
      return divide(v, c);
    }

    template <typename T, int R, int C>
    inline Eigen::Matrix<fvar<T>, R, C>
    operator/(const Eigen::Matrix<double, R, C>& v, const fvar<T>& c) {
      return divide(v, c);
    }
  }
}
#endif
