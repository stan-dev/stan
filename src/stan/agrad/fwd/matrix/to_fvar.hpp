#ifndef __STAN__AGRAD__FWD__MATRIX__TO_FVAR_HPP__
#define __STAN__AGRAD__FWD__MATRIX__TO_FVAR_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

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
    matrix_fv 
    to_fvar(const stan::math::matrix_d& m) {
      matrix_fv m_v(m.rows(), m.cols());
      for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
          m_v(i,j) = m(i,j);
      return m_v;
    }

    inline 
    matrix_fv 
    to_fvar(const matrix_fv& m) {
      return m;
    }

    inline
    vector_fv 
    to_fvar(const stan::math::vector_d& v) {
      vector_fv v_v(v.size());
      for (int i = 0; i < v.size(); ++i)
        v_v[i] = v[i];
      return v_v;
    }

    inline 
    vector_fv 
    to_fvar(const vector_fv& v) {
      return v;
    }

    inline 
    row_vector_fv 
    to_fvar(const stan::math::row_vector_d& rv) {
      row_vector_fv rv_v(rv.size());
      for (int i = 0; i < rv.size(); ++i)
        rv_v[i] = rv[i];
      return rv_v;
    }

    inline 
    row_vector_fv 
    to_fvar(const row_vector_fv& rv) {
      return rv;
    }

  }
}
#endif
