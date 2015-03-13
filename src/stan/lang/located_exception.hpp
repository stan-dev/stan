#ifndef STAN__LANG__LINE_NUM_EXCEPTION_HPP
#define STAN__LANG__LINE_NUM_EXCEPTION_HPP

#include <exception>
#include <iostream>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace stan {
  
  namespace lang {

    /**
     * Class to hold a reference to a nested exception along with a
     * starting line number of the statement in the Stan program which
     * caused the exception.
     *
     * <p>The exception class is defined to extend the type of the
     * of the nested exception so that catch statements cn be
     * keyed to the original exception type (or the templated line
     * number exception type).
     *
     * @tparam E Type of nested exception.
     */
    template <class E>
    class located_exception : public E { 

    private:

      /** Nested exception. */
      const E& e_;

      /** Line number where exception arose or -1 if unknown. */
      const int line_;

      /** Message describing exception */
      const std::string what_;
      
      /**
       * Construct the message for the exception description.
       * 
       * <p>This is a nested static function so that the constructor
       * can create the message and assign it to a constant member
       * variable. 
       *
       * @param e The nested exception.
       * @param line Line number of exception or -1 if unkown.
       * @return Description of exception including original
       * description and line number.
       */
      static std::string construct_what(const E& e, int line) throw() {
        std::ostringstream o;
        o << "Exception raised at line="
          << line
          << ": "
          << e.what();
        return o.str();
      }


    public:

      /**
       * Construct a located exception from a reference to the nested
       * exception and line number in a Stan program where the
       * exception arose.
       *
       * @param e Nested exception reference.
       * @param line Line number of statement where exception arose,
       * or -1 if it is unknown.
       */
      located_exception(const E& e, int line) throw()
        : E(e),
          e_(e),
          line_(line),
          what_(construct_what(e,line)) {
      }

      /**
       * Destroy this exception.  Because the nested exception is a
       * reference, it is not destroyed.
       */
      ~located_exception() throw() { }

      /**
       * Return a description of the exception, which includes the
       * line number and the description of the nested exception.
       *
       * @return Description of exception.
       */
      const char* what() const throw() {
        return what_.c_str();
      };

      /**
       * Return a constant reference to the nested exception as
       * a reference to the class template type.
       * 
       * @return The nested exception.
       */
      const E& get_nested() const throw() {
        return e_;
      }

      /**
       * Return the line number of the exception in the underlying
       * code.  This may be -1 if the location is not known or there
       * was an exception during initialization before the first
       * statement was executed.
       *
       * @return Line number of exception or -1 if unknown.
       */
      int line() const throw() { 
        return line_; 
      }
      
      /**
       * Return true if the nested exception can be cast to 
       * the specified template type.
       *
       * @tparam E2 Template type to compare against.
       * @return true if nested type is castable to E2.
       */
      template <typename E2>
      bool is_type() const {
        try {
          dynamic_cast<const E2&>(e_);
        } catch (const std::bad_cast& e) {
          return false;
        }
        return true;
      }

      /**
       * Return the underlying exception dynamically cast to the specified
       * template parameter type.
       * 
       * @tparam E2 Type to which the underlying exception is cast.
       * @return Underlying exception dynamically cast to the
       * specified type.
       * @throw std::bad_cast if the cast cannot be performed.
       */
      template <typename E2>
      const E2& get_nested_as() const {
        return dynamic_cast<const E2&>(e_);
      }


    };

    /**
     * Throws a located exception with the specified template
     * parameter if the run-time type of the argument exception
     * matches the template parameter.  
     *
     * @tparam E type to test against.
     * @param e Nested exception type.
     * @param line Line number in program causing error.
     * @throw Located exception of type E if the argument exception's
     * type is E.
     */
    template <typename E>
    void throw_located_if_type(const std::exception& e, int line) {
      if (typeid(e) == typeid(E))
        throw located_exception<E>(dynamic_cast<const E&>(e),line);
    }

    /**
     * Use runtime type information to throw a located exception with
     * the specified exception and line instantiated to the most
     * specific type possible from among the exception types available
     * in the standard library.
     *
     * The types tested are bad_alloc, bad_exception, bad_cast,
     * bad_type_id, ios_base::failure, domain_error, invalid_argument,
     * length_error, out_of_range, overflow_error, range_error,
     * underflow_error, logic_error, runtime_error.  If none of those
     * match, an instance of of located exception is instantiated with
     * std::exception. 
     *
     * @param e Nested exception.
     * @param line Line number in source program causing error.
     * @throw Always throws located exception of a type more specific
     * than std::exception.
     */
    void throw_located_exception(const std::exception& e, int line) {
      // derived types
      throw_located_if_type<std::bad_alloc>(e,line);
      throw_located_if_type<std::bad_exception>(e,line);
      throw_located_if_type<std::bad_cast>(e,line);
      throw_located_if_type<std::bad_typeid>(e,line);
      throw_located_if_type<std::ios_base::failure>(e,line);

      // logic errors
      throw_located_if_type<std::domain_error>(e,line);
      throw_located_if_type<std::invalid_argument>(e,line);
      throw_located_if_type<std::length_error>(e,line);
      throw_located_if_type<std::out_of_range>(e,line);

      // runtime errors
      throw_located_if_type<std::overflow_error>(e,line);
      throw_located_if_type<std::range_error>(e,line);
      throw_located_if_type<std::underflow_error>(e,line);
      
      // superclass catches
      throw_located_if_type<std::logic_error>(e,line);
      throw_located_if_type<std::runtime_error>(e,line);

      // default
      throw located_exception<std::exception>(e,line);
    }

  }

}

#endif
