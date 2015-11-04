#ifndef STAN_SERVICES_ARGUMENTS_VALUED_ARGUMENT_HPP
#define STAN_SERVICES_ARGUMENTS_VALUED_ARGUMENT_HPP

#include <stan/services/arguments/argument.hpp>
#include <string>

namespace stan {
  namespace services {

    class valued_argument: public argument {
    public:
      virtual void print(interface_callbacks::writer::base_writer& w,
                         const int depth,
                         const std::string& prefix) {
        std::string indent(compute_indent(depth), ' ');

        std::string message = prefix + indent + _name + " = " + print_value();

        if (is_default())
          message +=" (Default)";
        w(message);
      }

      virtual void print_help(interface_callbacks::writer::base_writer& w,
                              const int depth,
                              const bool recurse = false) {
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');

        w(indent + _name + "=<" + _value_type + ">");
        w(indent + subindent + _description);
        w(indent + subindent + "Valid values:" + print_valid());
        w(indent + subindent + "Defaults to " + _default);
        w();
      }

      virtual std::string print_value() = 0;
      virtual std::string print_valid() = 0;
      virtual bool is_default() = 0;

    protected:
      std::string _default;
      std::string _value_type;
    };

  }  // services
}  // stan

#endif
