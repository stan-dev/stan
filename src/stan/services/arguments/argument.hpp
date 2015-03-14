#ifndef STAN__SERVICES__ARGUMENTS__ARGUMENT__BETA
#define STAN__SERVICES__ARGUMENTS__ARGUMENT__BETA

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace stan {
  namespace services {
    
    class argument {
    public:
      
      argument()
        : indent_width(2),
          help_width(20) { }
      
      argument(const std::string name) 
        : _name(name),
          indent_width(2),
          help_width(20) { }

      virtual ~argument() { }
      
      std::string name() const { 
        return _name; 
      }
      
      std::string description() const { 
        return _description; 
      }

      template <class Writer>
      void print(Writer& writer, const int depth, const std::string prefix) {}
      
      template <class Writer>
      void print_help(Writer& writer, const int depth, const bool recurse) {
        std::cout << "Calling base method for some reason..." << std::endl;
      }
      
      template <class InfoWriter, class ErrWriter>
      bool parse_args(std::vector<std::string>& args,
                              InfoWriter& info,
                              ErrWriter& err,
                              bool& help_flag) { 
        return true; 
      }
      
      template <class Writer>
      void probe_args(argument* base_arg, Writer& writer) {};
      
      virtual void find_arg(std::string name,
                            std::string prefix,
                            std::vector<std::string>& valid_paths) {
        if (name == _name) {
          valid_paths.push_back(prefix + _name);
        }
      }
      
      static void split_arg(const std::string& arg, std::string& name, std::string& value) {
        size_t pos = arg.find('=');
        
        if (pos != std::string::npos) {
          name = arg.substr(0, pos);
          value = arg.substr(pos + 1, arg.size() - pos);
        }
        else {
          name = arg;
          value = "";
        }
      }
      
      virtual argument* arg(const std::string name) { 
        return 0; 
      }
      
      int compute_indent(const int depth) { 
        return indent_width * depth + 1; 
      }
      
    protected:
      std::string _name;
      std::string _description;
    
      int indent_width;
      int help_width;
    };

  } // services
} // stan
#endif

