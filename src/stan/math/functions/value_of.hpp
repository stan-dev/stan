#ifndef STAN__MATH__FUNCTIONS__VALUE_OF_HPP
#define STAN__MATH__FUNCTIONS__VALUE_OF_HPP

namespace stan {

  namespace math {
    
    /**
     * Return the value of the specified scalar argument
     * converted to a double value.
     *
     * <p>See the <code>stan::math::primitive_value</code> function to 
     * extract values without casting to <code>double</code>.
     *
     * <p>This function is meant to cover the primitive types. For
     * types requiring pass-by-reference, this template function
     * should be specialized.
     *
     * @tparam T Type of scalar.
     * @param x Scalar to convert to double.
     * @return Value of scalar cast to a double.
     */
    template <typename T>
    inline double value_of(const T x) {
      return static_cast<double>(x);
    }

    /**
     * Return the specified argument. 
     *
     * <p>See <code>value_of(T)</code> for a polymorphic
     * implementation using static casts.
     * 
     * <p>This inline pass-through no-op should be compiled away.
     *
     * @param x Specified value.
     * @return Specified value.
     */
    template <>
    inline double value_of<double>(const double x) {
      return x; 
    }

  }
}

#endif
