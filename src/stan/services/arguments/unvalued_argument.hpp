#ifndef STAN__SERVICES__ARGUMENTS__UNVALUED__ARGUMENT__BETA
#define STAN__SERVICES__ARGUMENTS__UNVALUED__ARGUMENT__BETA
#include <iostream>

#include <vector>
#include <stan/services/arguments/argument.hpp>

namespace stan {
  
  namespace services {
    
    class unvalued_argument: public argument {
      
    public:
      
      unvalued_argument()
        : _is_present(false) {}
      
      template <class Writer>
      void print(Writer& writer, const int depth, const std::string prefix) {}
      
      template <class Writer>
      void print_help(Writer& writer, const int depth, const bool recurse = false) {
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');

        writer.write_message(indent + _name);
        writer.write_message(indent + subindent + _description);
        writer.write_message();
      }
      
      template <class InfoWriter, class ErrWriter>
      bool parse_args(std::vector<std::string>& args,
                      InfoWriter& info, ErrWriter& err,
                      bool& help_flag) {
        if (args.size() == 0)
          return true;
        
        if ((args.back() == "help") || (args.back() == "help-all")) {
          print_help(info, 0);
          help_flag |= true;
          args.clear();
          return true;
        }
        
        _is_present = true;
        return true;
      };
      
      bool is_present() { 
        return _is_present; 
      }
      
    protected:
      bool _is_present;
    };

  } // services
} // stan
#endif
