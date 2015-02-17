#ifndef STAN__MATH__PRIM__MAT__FUN__TRACE_HPP
#define STAN__MATH__PRIM__MAT__FUN__TRACE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the trace of the specified matrix.  The trace
     * is defined as the sum of the elements on the diagonal.
     * The matrix is not required to be square.  Returns 0 if
     * matrix is empty.
     *
     * @param[in] m Specified matrix.
     * @return Trace of the matrix.
     */
    template <typename T>
    inline T trace(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      return m.trace();
    }
    
    template <typename T>
    inline T
      trace(const T& m) {
      return m;
    }
  }
}
#endif
