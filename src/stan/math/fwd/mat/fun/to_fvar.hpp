#ifndef STAN__MATH__FWD__MAT__FUN__TO_FVAR_HPP
#define STAN__MATH__FWD__MAT__FUN__TO_FVAR_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>
#include <stan/math/prim/scal/meta/is_constant.hpp>
#include <boost/utility/enable_if.hpp>

namespace stan {
  namespace agrad {

    template<typename T>
    inline
    fvar<T>
    to_fvar(const T& x) {
      return fvar<T>(x);
    }

    template<typename T>
    inline
    fvar<T>
    to_fvar(const fvar<T>& x) {
      return x;
    }


    template<int R, int C, typename T>
    inline
    Eigen::Matrix<T, R, C>
    to_fvar(const Eigen::Matrix<T, R, C>& m) {
      return m;
    }

    template<int R, int C>
    inline
    Eigen::Matrix<fvar<double>, R, C>
    to_fvar(const Eigen::Matrix<double, R, C>& m) {
      Eigen::Matrix<fvar<double>, R, C> m_fd(m.rows(), m.cols());
      for (int i = 0; i < m.size(); ++i)
        m_fd(i) = m(i);
      return m_fd;
    }

    template<typename T, int R, int C>
    inline
    Eigen::Matrix<fvar<T>, R, C>
    to_fvar(const Eigen::Matrix<T,R,C>& val,
            const Eigen::Matrix<T,R,C>& deriv) {

      stan::math::check_matching_dims("to_fvar",
                                      "value", val,
                                      "deriv", deriv);
      Eigen::Matrix<fvar<T>,R,C> ret(val.rows(), val.cols());
      for(int i = 0; i < val.rows(); i++) {
        for(int j = 0; j < val.cols(); j++) {
          ret(i,j).val_ = val(i,j);
          ret(i,j).d_ = deriv(i,j);
        }
      }
      return ret;
    }
  }
}
#endif
