#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__PARSER__HPP__
#define __STAN__GM__ARGUMENTS__ARGUMENT__PARSER__HPP__

#include <string>
#include <vector>
#include <fstream>

#include <stan/gm/arguments/argument.hpp>

namespace stan {
  
  namespace gm {
    
    class argument_parser {
      
    public:
      
      argument_parser(std::vector<argument*>& valid_args): _arguments(valid_args) {};

      bool parse_args(int argc, const char* argv[], std::ostream* err = 0) {
        
        if(argc == 1) return true;
        
        std::vector<std::string> args;
        
        // Fill in reverse order as parse_args pops from the back
        for (int i = argc - 1; i > 0; --i) 
          args.push_back(std::string(argv[i]));

        bool good_arg = true;
        bool valid_arg = true;
        
        while(good_arg) {
          
          if(args.size() == 0) return valid_arg;
          
          good_arg = false;
          
          std::string cat_name = args.back();
          
          std::string val_name;
          std::string val;
          argument::split_arg(cat_name, val_name, val);
          
          for (std::vector<argument*>::iterator it = _arguments.begin();
               it != _arguments.end(); ++it) {
            
            if( (*it)->name() == cat_name) {
              args.pop_back();
              valid_arg &= (*it)->parse_args(args, err);
              good_arg = true;
            }
            else if( (*it)->name() == val_name ) {
              valid_arg &= (*it)->parse_args(args, err);
              good_arg = true;
            }
            
          }
          
          if(!good_arg && err) *err << cat_name << " is either mistyped or misplaced." << std::endl;
          
        }
        
        return valid_arg && good_arg;
        
      }
      
      void print(std::ostream* s, const char prefix = '\0') {
        if(!s) return;
        
        for (int i = 0; i < _arguments.size(); ++i) {
          _arguments.at(i)->print(s, 0, prefix);
        }
        
      }
      
      void print_help(std::ostream* s) {
        if(!s) return;
        
        for (int i = 0; i < _arguments.size(); ++i) {
          _arguments.at(i)->print_help(s, 0);
        }
        
      }
      
      argument* arg(std::string name) {
        
        for (std::vector<argument*>::iterator it = _arguments.begin();
             it != _arguments.end(); ++it)
          if( name == (*it)->name() ) return (*it);
        
        return 0;
        
      }
      
    protected:
      
      std::vector<argument*>& _arguments;
      
      // We can also check for, and warn the user of, deprecated arguments
      //std::vector<argument*> deprecated_arguments;
      // check_arg_conflict // Ensure non-zero intersection of valid and deprecated argumentss
      
    };
    
  } // gm
  
} // stan

#endif

