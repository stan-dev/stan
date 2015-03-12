#ifndef STAN__SERVICES__ARGUMENTS__VALUED__ARGUMENT__BETA
#define STAN__SERVICES__ARGUMENTS__VALUED__ARGUMENT__BETA

#include <stan/services/arguments/argument.hpp>

namespace stan {
  
  namespace services {
    
    class valued_argument: public argument {
      
    public:
      
      template <class Writer>
      void print(Writer& writer, const int depth, const std::string prefix) {
        std::string indent(compute_indent(depth), ' ');
        
        std::string msg = prefix + indent + _name + " = " + print_value();
        if(is_default())
          msg += " (Default)";
        writer.write_message(msg);
      }
      
      template <class Writer>
      void print_help(Writer& writer, const int depth, const bool recurse = false) {
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');
        
        writer.write_message(indent + _name + "=<" + _value_type + ">");
        writer.write_message(indent + subindent + _description);
        writer.write_message(indent + subindent + "Valid values:" + print_valid());
        writer.write_message(indent + subindent + "Defaults to " + _default);
        writer.write_message("");
      }
      
      virtual std::string print_value() = 0;
      virtual std::string print_valid() = 0;
      virtual bool is_default() = 0;
      
    protected:
      
      std::string _default;
      std::string _value_type;
      
    };
    
  } // services
} // stan
#endif
