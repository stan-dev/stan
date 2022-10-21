#ifndef STAN_MODEL_RETHROW_LOCATED_HPP
#define STAN_MODEL_RETHROW_LOCATED_HPP

#include <exception>
#include <ios>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace stan {

namespace lang {

/**
 * Returns true if the specified exception can be dynamically
 * cast to the template parameter type.
 *
 * @tparam E Type to test.
 * @param[in] e Exception to test.
 * @return true if exception can be dynamically cast to type.
 */
template <typename E>
bool is_type(const std::exception& e) {
  try {
    (void)dynamic_cast<const E&>(e);
    return true;
  } catch (...) {
    return false;
  }
}

/**
 * Structure for a located exception for standard library
 * exception types that have no what-based constructors.
 *
 * @param E Type of original exception.
 */
template <typename E>
struct located_exception : public E {
  std::string what_;

  /**
   * Construct a located exception with no what message.
   */
  located_exception() throw() : what_("") {}

  /**
   * Construct a located exception with the specified what
   * message and specified original type.
   *
   * @param[in] what Original what message.
   * @param[in] orig_type Original type.
   */
  located_exception(const std::string& what,
                    const std::string& orig_type) throw()
      : what_(what + " [origin: " + orig_type + "]") {}

  /**
   * Destroy a located exception.
   */
  ~located_exception() throw() {}

  /**
   * Return the character sequence describing the exception,
   * including the original what message and original type if
   * constructed with such.
   *
   * @return Description of exception.
   */
  const char* what() const throw() { return what_.c_str(); }
};

/**
 * Rethrow an exception of type specified by the dynamic type of
 * the specified exception, adding the specified source file location to
 * the specified exception's message.
 *
 * @param[in] e original exception
 * @param[in] location string representing the source file location
 */
[[noreturn]] inline void rethrow_located(const std::exception& e,
                                         std::string location) {
  using std::bad_alloc;      // -> exception
  using std::bad_cast;       // -> exception
  using std::bad_exception;  // -> exception
  using std::bad_typeid;     // -> exception
  using std::domain_error;   // -> logic_error
  using std::exception;
  using std::invalid_argument;  // -> logic_error
  using std::ios_base;          // ::failure -> exception
  using std::length_error;      // -> logic_error
  using std::logic_error;       // -> exception
  using std::out_of_range;      // -> logic_error
  using std::overflow_error;    // -> runtime_error
  using std::range_error;       // -> runtime_error
  using std::runtime_error;     // -> exception
  using std::underflow_error;   // -> runtime_error

  // create message with trace of includes and location of error
  std::stringstream o;
  o << "Exception: " << e.what() << location;
  std::string s = o.str();

  if (is_type<bad_alloc>(e))
    throw located_exception<bad_alloc>(s, "bad_alloc");
  if (is_type<bad_cast>(e))
    throw located_exception<bad_cast>(s, "bad_cast");
  if (is_type<bad_exception>(e))
    throw located_exception<bad_exception>(s, "bad_exception");
  if (is_type<bad_typeid>(e))
    throw located_exception<bad_typeid>(s, "bad_typeid");
  if (is_type<domain_error>(e))
    throw domain_error(s);
  if (is_type<invalid_argument>(e))
    throw invalid_argument(s);
  if (is_type<length_error>(e))
    throw length_error(s);
  if (is_type<out_of_range>(e))
    throw out_of_range(s);
  if (is_type<logic_error>(e))
    throw logic_error(s);
  if (is_type<overflow_error>(e))
    throw overflow_error(s);
  if (is_type<range_error>(e))
    throw range_error(s);
  if (is_type<underflow_error>(e))
    throw underflow_error(s);
  if (is_type<runtime_error>(e))
    throw runtime_error(s);

  throw located_exception<exception>(s, "unknown original type");
}

}  // namespace lang

}  // namespace stan

#endif
