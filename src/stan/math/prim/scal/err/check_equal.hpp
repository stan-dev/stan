#ifndef STAN_MATH_PRIM_SCAL_ERR_CHECK_EQUAL_HPP
#define STAN_MATH_PRIM_SCAL_ERR_CHECK_EQUAL_HPP

#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/scal/err/domain_error_vec.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/get.hpp>
#include <string>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_eq,
                bool is_vec>
      struct equal {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y,
                          const T_eq& eq) {
          using stan::length;
          VectorView<const T_eq> eq_vec(eq);
          for (size_t n = 0; n < length(eq); n++) {
            if (!(y == eq_vec[n])) {
              std::stringstream msg;
              msg << ", but must be equal to ";
              msg << eq_vec[n];
              std::string msg_str(msg.str());

              domain_error(function, name, y,
                           "is ", msg_str.c_str());
            }
          }
          return true;
        }
      };

      // throws if y or eq is nan
      template <typename T_y,
                typename T_eq>
      struct equal<T_y, T_eq, true> {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y,
                          const T_eq& eq) {
          using stan::length;
          using stan::get;
          VectorView<const T_eq> eq_vec(eq);
          for (size_t n = 0; n < length(y); n++) {
            if (!(get(y, n) == eq_vec[n])) {
              std::stringstream msg;
              msg << ", but must be equal to ";
              msg << eq_vec[n];
              std::string msg_str(msg.str());
              domain_error_vec(function, name, y, n,
                               "is ", msg_str.c_str());
            }
          }
          return true;
        }
      };
    }

    /**
     * Return <code>true</code> if <code>y</code> is equal to
     * <code>eq</code>.
     *
     * This function is vectorized over both <code>y</code> and
     * <code>eq</code>. If both <code>y</code> and <code>eq</code> are
     * scalar or vector-like, then each element is compared in order.
     * If one of <code>y</code> or <code>eq</code> are vector and the
     * other is scalar, then the scalar is broadcast to the size of
     * the vector.
     *
     * @tparam T_y Type of variable
     * @tparam T_eq Type of comparison
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Variable to check equality
     * @param eq Expected value for y
     *
     * @return <code>true</code> if y is equal to eq
     * @throw <code>std::domain_error</code> if y is unequal to eq or
     *    if any element of y or eq is NaN.
     */
    template <typename T_y, typename T_eq>
    inline bool check_equal(const char* function,
                            const char* name,
                            const T_y& y,
                            const T_eq& eq) {
      return equal<T_y, T_eq, is_vector_like<T_y>::value>
        ::check(function, name, y, eq);
    }
  }
}
#endif
