#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__READER__HPP__
#define __STAN__GM__ARGUMENTS__ARGUMENT__READER__HPP__

#include <string>
#include <vector>
#include <fstream>

#include <stan/gm/arguments/argument.hpp>

namespace stan {
  
  namespace gm {
    
    class argument_reader {
      
    public:
      
      argument_reader(std::vector<argument*>& valid_args): _valid_arguments(valid_args) {};
      
      void parse_arguments(int argc, const char* argv[]) {
        
        int arg_index = -1;
        for (int i = 1; i < argc; ++i) {
          
          std::string arg(argv[i]);
          
          if (arg.at(0) == '-') {
            
            arg.erase(0, 1);
            
            // Check validity
            bool valid = false;
            
            for (int i = 0; i < _valid_arguments.size(); ++i) {
              if(arg == _valid_arguments.at(i)->get_name()) {
                arg_index = i;
                valid = true;
                break;
              }
            }
            
            if(!valid) {
              std::cout << "The argument " << arg
                        << " is not valid.  It, and all sub-arguments "
                        << "immediately following, will be ignored." << std::endl;
              arg_index = -1;
            }
            
          }
          else {
            
            if(arg_index < 0) {
              std::cout << "Expecting a new flagged argument but instead found the sub-argument "
                        << arg << ", which will be ignored." << std::endl;
              break;
            }
            
            size_t pos = arg.find('=');
            
            if (pos != std::string::npos) {
              std::string name = arg.substr(0, pos);
              double value = boost::lexical_cast<double>(arg.substr(pos + 1, arg.size() - pos));
              
              _valid_arguments.at(arg_index)->parse_subargument(name, value);
              
            }
            else {
              std::cout << "Found the sub-argument " << arg
                        << " without any value, it will be ignored." << std::endl;
            }
            
          }

        }
        
      }
      
      void print_arguments(std::ostream* s) {
        if(!s) return;
        
        for (int i = 0; i < _valid_arguments.size(); ++i) {
          _valid_arguments.at(i)->print_subarguments(s);
        }
        
      }
      
    protected:
      
      std::vector<argument*>& _valid_arguments;
      
      // We can also check for, and warn the user of, deprecated arguments
      //std::vector<argument*> deprecated_arguments;
      // check_arg_conflict // Ensure non-zero intersection of valid and deprecated argumentss
      
    };
    
  } // gm
  
} // stan

#endif

