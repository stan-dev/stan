#ifndef STAN__MATH__MIX__MAT__FUN__TYPEDEFS_HPP
#define STAN__MATH__MIX__MAT__FUN__TYPEDEFS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>

namespace stan {
  namespace agrad {
    
    typedef 
    Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic>
    matrix_fv;

    typedef 
    Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic>
    matrix_ffv;

    typedef 
    Eigen::Matrix<fvar<var>,Eigen::Dynamic,1>
    vector_fv;

    typedef 
    Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1>
    vector_ffv;

    typedef 
    Eigen::Matrix<fvar<var>,1,Eigen::Dynamic>
    row_vector_fv;

    typedef 
    Eigen::Matrix<fvar<fvar<var> >,1,Eigen::Dynamic>
    row_vector_ffv;

  }
}
#endif
