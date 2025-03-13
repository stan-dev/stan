#ifndef STAN_SERVICES_UTIL_DEFAULTS_HPP
#define STAN_SERVICES_UTIL_DEFAULTS_HPP

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace stan {
namespace services {
namespace util {

/**
 * Generic template for defining and validating configuration option values.
 * This template can be specialized for different option types.
 *
 * @tparam T Value type of the option
 * @tparam Tag Tag type used to distinguish between different options of
 *             the same value type
 */
template <typename T, typename Tag>
class option {
 public:
  using value_type = T;
  
  /**
   * Return the string description of the option.
   * Must be specialized for each option.
   *
   * @return Description string
   */
  static std::string description() {
    return "Option description not available.";
  }

  /**
   * Return the default value for the option.
   * Must be specialized for each option.
   *
   * @return Default value
   */
  static T default_value() {
    // This generic implementation provides a reasonable default 
    // for common types, but should be specialized for each option
    if constexpr (std::is_integral_v<T>) {
      return 0;
    } else if constexpr (std::is_floating_point_v<T>) {
      return 0.0;
    } else if constexpr (std::is_same_v<T, bool>) {
      return false;
    } else {
      // This will cause a compilation error for types that don't have a default
      static_assert(std::is_default_constructible_v<T>, 
                    "No default value defined for this option type");
      return T();
    }
  }

  /**
   * Validate a option value.
   * Must be specialized for each option.
   *
   * @param value Value to validate
   * @throw std::invalid_argument if the value is invalid
   */
  static void validate(const T& value) {
    // By default, all values are valid
    // Specializations should implement specific validation logic
  }
};

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
