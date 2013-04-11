#ifndef __STAN__AGRAD__FWD__MATRIX__TO_FVAR_HPP__
#define __STAN__AGRAD__FWD__MATRIX__TO_FVAR_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.
     *
     * @param[in] x A scalar value
     * @return An automatic differentiation variable with the input value.
     */
    inline fvar<double> to_fvar(const double& x) {
      return fvar<double>(x);
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.
     *
     * @param[in] x An automatic differentiation variable.
     * @return An automatic differentiation variable with the input value.
     */    
    inline fvar<double> to_fvar(const fvar<double>& x) {
      return x;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.
     *
     * @param[in] m A Matrix with scalars
     * @return A Matrix with automatic differentiation variables
     */
    inline matrix_fv to_fvar(const stan::math::matrix_d& m) {
      matrix_fv m_v(m.rows(), m.cols());
      for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
          m_v(i,j) = m(i,j);
      return m_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.
     * 
     * @param[in] m A Matrix with automatic differentiation variables.
     * @return A Matrix with automatic differentiation variables.
     */
    inline matrix_fv to_fvar(const matrix_fv& m) {
      return m;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.     
     *
     * @param[in] v A Vector of scalars
     * @return A Vector of automatic differentiation variables with
     *   values of v
     */
    inline vector_fv to_fvar(const stan::math::vector_d& v) {
      vector_fv v_v(v.size());
      for (int i = 0; i < v.size(); ++i)
        v_v[i] = v[i];
      return v_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.     
     *
     * @param[in] v A Vector of automatic differentiation variables
     * @return A Vector of automatic differentiation variables with
     *   values of v
     */
    inline vector_fv to_fvar(const vector_fv& v) {
      return v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] rv A row vector of scalars
     * @return A row vector of automatic differentation variables with 
     *   values of rv.
     */
    inline row_vector_fv to_fvar(const stan::math::row_vector_d& rv) {
      row_vector_fv rv_v(rv.size());
      for (int i = 0; i < rv.size(); ++i)
        rv_v[i] = rv[i];
      return rv_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::fvar variable with the input value.     
     *
     * @param[in] rv A row vector with automatic differentiation variables
     * @return A row vector with automatic differentiation variables
     *    with values of rv.
     */
    inline row_vector_fv to_fvar(const row_vector_fv& rv) {
      return rv;
    }

  }
}
#endif
