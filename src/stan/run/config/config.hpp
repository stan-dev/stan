#ifndef STAN_RUN_CONFIG_BASE_HPP
#define STAN_RUN_CONFIG_BASE_HPP

#include <string>
#include <stdexcept>
#include <functional>

namespace stan {
namespace run {

/**
 * Generic configuration option template class.
 * Handles default values, descriptions, and validation.
 */
template <typename T>
class config {
public:
  /**
   * Constructor with default value, description, and validator.
   */
  config(T default_value, 
        std::string desc,
        std::function<void(const T&)> validator = [](const T&){})
    : value_(default_value), 
      description_(std::move(desc)), 
      validate_(validator) 
  {
    // Validate the default value at construction
    validate_(value_);
  }

  /**
   * Returns the config value.
   */
  const T& value() const { return value_; }

  /**
   * Returns the config description.
   */
  const std::string& description() const { return description_; }

  /**
   * Sets the config value with validation.
   */
  void set_value(const T& value) {
    validate_(value);
    value_ = value;
  }

private:
  T value_;
  std::string description_;
  std::function<void(const T&)> validate_;
};

}  // namespace run
}  // namespace stan

#endif
