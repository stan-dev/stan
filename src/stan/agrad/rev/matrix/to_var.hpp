#ifndef STAN__AGRAD__REV__MATRIX__TO_VAR_HPP
#define STAN__AGRAD__REV__MATRIX__TO_VAR_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

namespace stan {
  namespace agrad {

    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     *
     * @param[in] x A scalar value
     * @return An automatic differentiation variable with the input value.
     */
    inline var to_var(const double& x) {
      return var(x);
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     *
     * @param[in] x An automatic differentiation variable.
     * @return An automatic differentiation variable with the input value.
     */    
    inline var to_var(const var& x) {
      return x;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     *
     * @param[in] m A Matrix with scalars
     * @return A Matrix with automatic differentiation variables
     */
    inline matrix_v to_var(const stan::math::matrix_d& m) {
      matrix_v m_v(m.rows(), m.cols());
      for (int j = 0; j < m.cols(); ++j)
        for (int i = 0; i < m.rows(); ++i)
          m_v(i,j) = m(i,j);
      return m_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.
     * 
     * @param[in] m A Matrix with automatic differentiation variables.
     * @return A Matrix with automatic differentiation variables.
     */
    inline matrix_v to_var(const matrix_v& m) {
      return m;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] v A Vector of scalars
     * @return A Vector of automatic differentiation variables with
     *   values of v
     */
    inline vector_v to_var(const stan::math::vector_d& v) {
      vector_v v_v(v.size());
      for (int i = 0; i < v.size(); ++i)
        v_v[i] = v[i];
      return v_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] v A Vector of automatic differentiation variables
     * @return A Vector of automatic differentiation variables with
     *   values of v
     */
    inline vector_v to_var(const vector_v& v) {
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
    inline row_vector_v to_var(const stan::math::row_vector_d& rv) {
      row_vector_v rv_v(rv.size());
      for (int i = 0; i < rv.size(); ++i)
        rv_v[i] = rv[i];
      return rv_v;
    }
    /**
     * Converts argument to an automatic differentiation variable.
     *
     * Returns a stan::agrad::var variable with the input value.     
     *
     * @param[in] rv A row vector with automatic differentiation variables
     * @return A row vector with automatic differentiation variables
     *    with values of rv.
     */
    inline row_vector_v to_var(const row_vector_v& rv) {
      return rv;
    }

  }
}
#endif
