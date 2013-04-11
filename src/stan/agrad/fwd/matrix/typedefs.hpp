#ifndef __STAN__AGRAD__FWD__MATRIX__TYPEDEFS_HPP__
#define __STAN__AGRAD__FWD__MATRIX__TYPEDEFS_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;

    /**
     * The type of a matrix holding <code>stan::agrad::fvar</code>
     * values.
     */
    typedef 
    Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic>
    matrix_fv;

    /**
     * The type of a (column) vector holding <code>stan::agrad::fvar</code>
     * values.
     */
    typedef 
    Eigen::Matrix<fvar<double>,Eigen::Dynamic,1>
    vector_fv;

    /**
     * The type of a row vector holding <code>stan::agrad::fvar</code>
     * values.
     */
    typedef 
    Eigen::Matrix<fvar<double>,1,Eigen::Dynamic>
    row_vector_fv;

  }
}
#endif
