#ifndef STAN_IO_JSON_JSON_HANDLER_HPP
#define STAN_IO_JSON_JSON_HANDLER_HPP

#include <cstdint>
#include <string_view>

namespace stan {

namespace json {

/**
 * Abstract base class for JSON handlers.  More efficient to just
 * implement directly and pass in as a template, but this version
 * is available for convenience.
 */
class json_handler {
 public:
  json_handler() {}

  ~json_handler() {}

  /**
   * Handle the start of the text.
   */
  virtual void start_text() {}

  /**
   * Handle the end of the text.
   */
  virtual void end_text() {}

  /**
   * Handle the start of an array.
   */
  virtual void start_array() {}

  /**
   * Handle the end of an array.
   */
  virtual void end_array() {}

  /**
   * Handle the start of an object.
   */
  virtual void start_object() {}

  /**
   * Handle the end of an object.
   */
  virtual void end_object() {}

  /**
   * Handle the null literal value.
   */
  virtual void null() {}

  /**
   * Handle the boolean literal value of the specified polarity.
   *
   * @param p polarity of boolean
   */
  virtual void boolean(bool p) {}

  /**
   * Handle the specified double-precision floating point
   * value.
   *
   * @param x Value to handle.
   */
  virtual void number_double(double x) {}

  /**
   * Handle the specified integer value.
   *
   * @param n Value to handle.
   */
  virtual void number_int(int n) {}

  /**
   * Handle the specified unsigned integer value.
   *
   * @param n Value to handle.
   */
  virtual void number_unsigned_int(unsigned n) {}

  /**
   * Handle the specified 64-bit integer value.
   *
   * @param n Value to handle.
   */
  virtual void number_int64(int64_t n) {}

  /**
   * Handle the specified unsigned 64-bit integer value.
   *
   * @param n Value to handle.
   */
  virtual void number_unsigned_int64(uint64_t n) {}

  /**
   * Handle the specified string value.
   *
   * The view is only valid for the duration of the call; a handler which
   * needs to retain the text must copy it.
   *
   * @param s String value to handle.
   */
  virtual void string(std::string_view s) {}

  /**
   * Handle the specified object key.
   *
   * The view is only valid for the duration of the call; a handler which
   * needs to retain the text must copy it.
   *
   * @param s String object key to handle.
   */
  virtual void key(std::string_view s) {}
};

}  // namespace json
}  // namespace stan

#endif
