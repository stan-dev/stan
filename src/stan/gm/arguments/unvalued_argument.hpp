#ifndef __STAN__GM__ARGUMENTS__UNVALUED__ARGUMENT__BETA__
#define __STAN__GM__ARGUMENTS__UNVALUED__ARGUMENT__BETA__

#include <vector>
#include <stan/gm/arguments/argument.hpp>

namespace stan {
  
  namespace gm {
    
    class unvalued_argument: public argument {
      
    public:
      
      unvalued_argument(): _is_present(false) {};
      
      void print(std::ostream* s, int depth, const char prefix) {};
      
      void print_help(std::ostream* s, int depth, bool recurse = false) {
        
        if(!s) return;
        
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');
        
        *s << indent << _name << std::endl;
        *s << indent << subindent << _description << std::endl;
        *s << std::endl;
        
      }
      
      bool parse_args(std::vector<std::string>& args, std::ostream* out,
                      std::ostream* err, bool& help_flag) {
        
        if ( (args.back() == "help") || (args.back() == "help-all") ) {
          print_help(out, 0);
          help_flag |= true;
          args.clear();
          return true;
        }
        
        _is_present = true;
        return true;
      };
      
      bool is_present() { return _is_present; }
      
    protected:
      
      bool _is_present;
      
    };
    
  } // gm
  
} // stan

#endif