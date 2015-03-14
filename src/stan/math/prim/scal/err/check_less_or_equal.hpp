#ifndef STAN__MATH__PRIM__SCAL__ERR__CHECK_LESS_OR_EQUAL_HPP
#define STAN__MATH__PRIM__SCAL__ERR__CHECK_LESS_OR_EQUAL_HPP

#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/scal/err/domain_error_vec.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/get.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/arr/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <string>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y, typename T_high, bool is_vec>
      struct less_or_equal {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y,
                          const T_high& high) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(high); n++) {
            if (!(y <= high_vec[n])) {
              std::stringstream msg;
              msg << ", but must be less than or equal to ";
              msg << high_vec[n];
              std::string msg_str(msg.str());
              domain_error(function, name, y,
                           "is ", msg_str.c_str());
            }
          }
          return true;
        }
      };

      template <typename T_y, typename T_high>
      struct less_or_equal<T_y, T_high, true> {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y,
                          const T_high& high) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(stan::get(y, n) <= high_vec[n])) {
              std::stringstream msg;
              msg << ", but must be less than or equal to ";
              msg << high_vec[n];
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
     * Return <code>true</code> if <code>y</code> is less or equal to
     * <code>high</code>.
     *
     * This function is vectorized and will check each element of
     * <code>y</code> against each element of <code>high</code>.
     *
     * @tparam T_y Type of y
     * @tparam T_high Type of upper bound
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Variable to check
     * @param high Upper bound
     *
     * @return <code>true</code> if y is less than or equal to low.
     * @throw <code>std::domain_error</code> if y is not less than or equal to low
     *   or if any element of y or high is NaN.
     */
    template <typename T_y, typename T_high>
    inline bool check_less_or_equal(const char* function,
                                    const char* name,
                                    const T_y& y,
                                    const T_high& high) {
      return less_or_equal<T_y, T_high, is_vector_like<T_y>::value>
        ::check(function, name, y, high);
    }
  }
}
#endif
