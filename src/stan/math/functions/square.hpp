#ifndef STAN__MATH__FUNCTIONS__SQUARE_HPP
#define STAN__MATH__FUNCTIONS__SQUARE_HPP

namespace stan {
  namespace math {
    
    /**
     * Return the square of the specified argument.
     *
     * <p>\f$\mbox{square}(x) = x^2\f$.
     *
     * <p>The implementation of <code>square(x)</code> is just 
     * <code>x * x</code>.  Given this, this method is mainly useful 
     * in cases where <code>x</code> is not a simple primitive type, 
     * particularly when it is an auto-dif type.
     *
     * @param x Input to square.
     * @return Square of input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline T square(const T x) {
      return x * x;
    }

  }
}

#endif
