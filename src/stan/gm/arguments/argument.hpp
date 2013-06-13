#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__BETA__
#define __STAN__GM__ARGUMENTS__ARGUMENT__BETA__

#include <string>
#include <fstream>

namespace stan {

  namespace gm {
    
    class argument {
      
    public:
      
      argument(): indent_width(2) {};
      virtual ~argument() {};

      std::string name() { return _name; }
      std::string description() { return _description; }

      virtual void print(std::ostream* s, int depth, const char prefix) = 0;
      virtual void print_help(std::ostream* s, int depth, bool recurse) = 0;
      
      virtual bool parse_args(std::vector<std::string>& args,
                              std::ostream* out,
                              std::ostream* err,
                              bool& help_flag) { return true; }
      
      static void split_arg(std::string arg, std::string& name, std::string& value) {
        
        size_t pos = arg.find('=');
        
        if (pos != std::string::npos) {
          name = arg.substr(0, pos);
          value = arg.substr(pos + 1, arg.size() - pos);
        }
        
      }
      
      virtual argument* arg(std::string name) { return 0; }
      
      int compute_indent(int depth) { return indent_width * depth + 1; }
      
    protected:
      
      std::string _name;
      std::string _description;
    
      int indent_width;
      
    };
    
  } // gm
  
} // stan

#endif

