#ifndef STAN_SERVICES_UTIL_GET_UNDERLYING_HPP
#define STAN_SERVICES_UTIL_GET_UNDERLYING_HPP

#include <memory>
#include <utility>

namespace stan {
namespace services {
namespace util {
/**
 * Specializtion to get a const reference to the underlying value in a
 *  shared_ptr.
 */
template <typename T>
inline auto&& get_underlying(const std::shared_ptr<T>& x) {
  return *x;
}

/**
 * Specializtion to get a const reference to the underlying value in a
 *  unique_ptr.
 */
template <typename T>
inline auto&& get_underlying(const std::unique_ptr<T>& x) {
  return *x;
}

/**
 * Specialization to return back the input
 */
template <typename T>
inline auto&& get_underlying(T&& x) {
  return std::forward<T>(x);
}

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
