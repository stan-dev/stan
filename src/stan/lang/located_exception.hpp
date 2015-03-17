#ifndef STAN__LANG__LOCATED_EXCEPTION_HPP
#define STAN__LANG__LOCATED_EXCEPTION_HPP

#include <exception>
#include <string>
#include <sstream>
#include <typeinfo>

// from client side:
// throw construct_located_exception(const std::exception& e, int line);

// if (is_type<located_exception<...>) construct
// if (is_type<...>) construct

namespace stan {
  
  namespace lang {

    class located_exception : public std::exception {
    private:
      const std::exception& e_;
      const int line_;
      const std::string what_;
      static std::string construct_what(const std::exception& e, 
                                        int line) throw() {
        std::ostringstream o;
        o << "Exception raised at source line "
          << line
          << ":"
          << std::endl
          << e.what();
        return o.str();
      }
      static const 
      std::exception& get_base(const std::exception& e) throw() {
        try {
          return get_base(dynamic_cast<const located_exception&>(e)
                          .nested_exception());
        } catch (const std::bad_cast& /*e2*/) {
          return e;
        }
      }
    public:             
      located_exception(const std::exception& e, int line) throw() 
        : e_(e), 
          line_(line),
          what_(construct_what(e,line)) { 
      }
      ~located_exception() throw() { }
      const char* what() const throw() { 
        return what_.c_str();
      }
      const std::exception& nested_exception() const throw() { 
        return e_; 
      }
      int line() { 
        return line_; 
      }
      const std::exception& base_exception() const throw() {
        return get_base(*this);
      }
      template <typename E>
      bool base_exception_is() const throw() {
        try {
          (void)dynamic_cast<const E&>(base_exception());
          return true;
        } catch (const std::bad_cast& /*e*/) {
          return false;
        }
      }
    };

  }
}

#endif
