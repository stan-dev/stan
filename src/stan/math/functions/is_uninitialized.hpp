#ifndef __STAN__MATH__FUNCTIONS__IS_UNINITIALIZED_HPP__
#define __STAN__MATH__FUNCTIONS__IS_UNINITIALIZED_HPP__

namespace stan {

  namespace math {

    /**
     * Returns <code>true</code> if the specified variable is
     * uninitialized.  Arithmetic types are always initialized
     * by definition (the value is not specified).
     * 
     * @tparam T Type of object to test.
     * @param x Object to test.
     * @return <code>true</code> if the specified object is uninitialized.
     */
    template <typename T>
    inline bool is_uninitialized(T x) {
      return false;
    }

  }
}
#endif
