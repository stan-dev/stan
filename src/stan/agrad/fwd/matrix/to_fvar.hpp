#ifndef STAN__AGRAD__FWD__MATRIX__TO_FVAR_HPP
#define STAN__AGRAD__FWD__MATRIX__TO_FVAR_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_matching_dims.hpp>

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

    inline 
    matrix_fd
    to_fvar(const stan::math::matrix_d& m) {
      matrix_fd m_v(m.rows(), m.cols());
      for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
          m_v(i,j) = m(i,j);
      return m_v;
    }

    inline 
    matrix_fd
    to_fvar(const matrix_fd& m) {
      return m;
    }

    inline 
    matrix_fv
    to_fvar(const matrix_fv& m) {
      return m;
    }

    inline 
    matrix_ffd
    to_fvar(const matrix_ffd& m) {
      return m;
    }

    inline 
    matrix_ffv 
    to_fvar(const matrix_ffv& m) {
      return m;
    }

    inline
    vector_fd
    to_fvar(const stan::math::vector_d& v) {
      vector_fd v_v(v.size());
      for (int i = 0; i < v.size(); ++i)
        v_v[i] = v[i];
      return v_v;
    }

    inline 
    vector_fd
    to_fvar(const vector_fd& v) {
      return v;
    }

    inline 
    vector_fv
    to_fvar(const vector_fv& v) {
      return v;
    }

    inline 
    vector_ffd
    to_fvar(const vector_ffd& v) {
      return v;
    }

    inline 
    vector_ffv 
    to_fvar(const vector_ffv& v) {
      return v;
    }

    inline 
    row_vector_fd 
    to_fvar(const stan::math::row_vector_d& rv) {
      row_vector_fd rv_v(rv.size());
      for (int i = 0; i < rv.size(); ++i)
        rv_v[i] = rv[i];
      return rv_v;
    }

    inline 
    row_vector_fd 
    to_fvar(const row_vector_fd& rv) {
      return rv;
    }

    inline 
    row_vector_fv
    to_fvar(const row_vector_fv& rv) {
      return rv;
    }

    inline 
    row_vector_ffd 
    to_fvar(const row_vector_ffd& rv) {
      return rv;
    }

    inline 
    row_vector_ffv
    to_fvar(const row_vector_ffv& rv) {
      return rv;
    }

    template<typename T, int R, int C>
    inline
    Eigen::Matrix<fvar<T>, R, C>
    to_fvar(const Eigen::Matrix<T,R,C>& val,
            const Eigen::Matrix<T,R,C>& deriv) {
      
      stan::error_handling::check_matching_dims("to_fvar",
                                                "value", val,
                                                "deriv", deriv);
      Eigen::Matrix<fvar<T>,R,C> ret(val.rows(), val.cols());
      for(size_type i = 0; i < val.rows(); i++) {
        for(size_type j = 0; j < val.cols(); j++) {
          ret(i,j).val_ = val(i,j);
          ret(i,j).d_ = deriv(i,j);
        }
      }
      return ret;
    }
  }
}
#endif
