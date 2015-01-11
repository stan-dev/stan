#ifndef STAN__MATH__MATRIX__MINUS_HPP
#define STAN__MATH__MATRIX__MINUS_HPP

namespace stan {
  namespace math {

    /**
     * Returns the negation of the specified scalar or matrix.
     *
     * @tparam T Type of subtrahend.
     * @param x Subtrahend.
     * @return Negation of subtrahend.
     */
    template <typename T>
    inline
    T minus(const T& x) {
      return -x;
    }

  }
}
#endif
