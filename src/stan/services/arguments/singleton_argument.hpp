#ifndef STAN_SERVICES_ARGUMENTS_SINGLETON_ARGUMENT_HPP
#define STAN_SERVICES_ARGUMENTS_SINGLETON_ARGUMENT_HPP

#include <stan/services/arguments/valued_argument.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace stan {
  namespace services {

    template <typename T>
    struct type_name {
      static std::string name() { return typeid(T).name(); }
    };

    // Specialize to something more readable
    template <>
    struct type_name<int> {
      static std::string name() { return "int"; }
    };

    template <>
    struct type_name<unsigned int> {
      static std::string name() { return "unsigned int"; }
    };

    template <>
    struct type_name<double> {
      static std::string name() { return "double"; }
    };

    template <>
    struct type_name<bool> {
      static std::string name() { return "boolean"; }
    };

    template <>
    struct type_name<std::string> {
      static std::string name() { return "string"; }
    };

    template<typename T>
    class singleton_argument: public valued_argument {
    public:
      singleton_argument(): _validity("All") {
        _constrained = false;
        _name = "";
        _value_type = type_name<T>::name();
      }

      explicit singleton_argument(const std::string name): _validity("All") {
        _name = name;
      }

      bool parse_args(std::vector<std::string>& args,
                      interface_callbacks::writer::base_writer& info,
                      interface_callbacks::writer::base_writer& err,
                      bool& help_flag) {
        if (args.size() == 0)
          return true;

        if ( (args.back() == "help") || (args.back() == "help-all") ) {
          print_help(info, 0);
          help_flag |= true;
          args.clear();
          return true;
        }

        std::string name;
        std::string value;
        split_arg(args.back(), name, value);

        if (_name == name) {
          args.pop_back();

          T proposed_value = boost::lexical_cast<T>(value);

          if (!set_value(proposed_value)) {
            std::stringstream message;
            message << proposed_value
                    << " is not a valid value for "
                    << "\"" << _name << "\"";
            err(message.str());
            err(std::string(indent_width, ' ')
                + "Valid values:" + print_valid());

            args.clear();
            return false;
          }
        }
        return true;
      }

      virtual void probe_args(argument* base_arg,
                            stan::interface_callbacks::writer::base_writer& w) {
        w("good");
        _value = _good_value;
        base_arg->print(w, 0, "");
        w();

        if (_constrained) {
          w("bad");
          _value = _bad_value;
          base_arg->print(w, 0, "");
          w();
        }

        _value = _default_value;
      }

      void find_arg(const std::string& name,
                    const std::string& prefix,
                    std::vector<std::string>& valid_paths) {
        if (name == _name) {
          valid_paths.push_back(prefix + _name + "=<" + _value_type + ">");
        }
      }

      T value() { return _value; }

      bool set_value(const T& value) {
        if (is_valid(value)) {
          _value = value;
          return true;
        }
        return false;
      }

      std::string print_value() {
        return boost::lexical_cast<std::string>(_value);
      }

      std::string print_valid() {
        return " " + _validity;
      }

      bool is_default() {
        return _value == _default_value;
      }

    protected:
      std::string _validity;
      virtual bool is_valid(T value) { return true; }

      T _value;
      T _default_value;

      bool _constrained;

      T _good_value;
      T _bad_value;
    };

    typedef singleton_argument<double> real_argument;
    typedef singleton_argument<int> int_argument;
    typedef singleton_argument<unsigned int> u_int_argument;
    typedef singleton_argument<bool> bool_argument;
    typedef singleton_argument<std::string> string_argument;

  }  // services
}  // stan

#endif
