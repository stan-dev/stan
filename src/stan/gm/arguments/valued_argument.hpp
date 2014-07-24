#ifndef STAN__GM__ARGUMENTS__VALUED__ARGUMENT__BETA
#define STAN__GM__ARGUMENTS__VALUED__ARGUMENT__BETA

#include <stan/gm/arguments/argument.hpp>

namespace stan {
  
  namespace gm {
    
    class valued_argument: public argument {
      
    public:
      
      virtual void print(std::ostream* s, const int depth, const std::string prefix) {
        if (!s)
          return;
        
        std::string indent(compute_indent(depth), ' ');
        
        *s << prefix << indent << _name << " = " << print_value();
        if(is_default())
          *s << " (Default)";
        *s << std::endl;
      }
      
      virtual void print_help(std::ostream* s, const int depth, const bool recurse = false) {
        if (!s) 
          return;
        
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');
        
        *s << indent << _name << "=<" << _value_type << ">" << std::endl;
        *s << indent << subindent << _description << std::endl;
        *s << indent << subindent << "Valid values:" << print_valid() << std::endl;
        *s << indent << subindent << "Defaults to " << _default << std::endl;
        *s << std::endl;
        
      }
      
      virtual std::string print_value() = 0;
      virtual std::string print_valid() = 0;
      virtual bool is_default() = 0;
      
    protected:
      
      std::string _default;
      std::string _value_type;
      
    };
    
  } // gm
} // stan
#endif
