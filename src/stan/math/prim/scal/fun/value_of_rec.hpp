#ifndef STAN__MATH__PRIM__SCAL__FUN__VALUE_OF_REC_HPP
#define STAN__MATH__PRIM__SCAL__FUN__VALUE_OF_REC_HPP

namespace stan {

  namespace math {
    
    /**
     * Return the value of the specified scalar argument
     * converted to a double value.
     *
     * <p>See the <code>stan::math::primitive_value</code> function to 
     * extract values without casting to <code>double</code>.
     *
     * @param x Scalar to convert to double.
     * @return Value of scalar cast to a double.
     */
    inline double value_of_rec(double x) {
      return x;
    }
    inline double value_of_rec(int x) {
      return static_cast<double>(x);
    }

  }
}

#endif
