#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__PARSER__HPP__
#define __STAN__GM__ARGUMENTS__ARGUMENT__PARSER__HPP__

#include <string>
#include <vector>
#include <fstream>

#include <stan/gm/arguments/argument.hpp>
#include <stan/gm/arguments/arg_method.hpp>

namespace stan {
  
  namespace gm {
    
    class argument_parser {
      
    public:
      
      argument_parser(std::vector<argument*>& valid_args)
        : _arguments(valid_args),
          _help_flag(false),
          _method_flag(false)
      {
        _arguments.insert(_arguments.begin(), new arg_method());
      }
      
      bool parse_args(int argc,
                      const char* argv[],
                      std::ostream* out = 0,
                      std::ostream* err = 0) {
        
        if (argc == 1) {
        
          print_help_header(out, argv[0]);
          print_help(out, false);
          print_help_footer(out, argv[0]);
          
          _help_flag |= true;
          
          return 0;
          
        }
        
        std::vector<std::string> args;
        
        // Fill in reverse order as parse_args pops from the back
        for (int i = argc - 1; i > 0; --i) 
          args.push_back(std::string(argv[i]));

        bool good_arg = true;
        bool valid_arg = true;
        _help_flag = false;
        
        std::vector<argument*> unset_args = _arguments;
        
        while(good_arg) {
          
          if (args.size() == 0)
            break;
          
          good_arg = false;
          std::string cat_name = args.back();
          
          // Check for method arguments entered without the method= prefix
          if (!_method_flag) {

            list_argument* method = dynamic_cast<list_argument*>(_arguments.front());
            
            if (method->valid_value(cat_name)) {
              cat_name = "method=" + cat_name;
              args.back() = cat_name;
            }
            
          }

          std::string val_name;
          std::string val;
          argument::split_arg(cat_name, val_name, val);
          
          if (val_name == "method")
            _method_flag = true;
          
          std::vector<argument*>::iterator arg_it;
          
          for (arg_it = unset_args.begin(); arg_it != unset_args.end(); ++arg_it) {
            if ( (*arg_it)->name() == cat_name) {
              args.pop_back();
              valid_arg &= (*arg_it)->parse_args(args, out, err, _help_flag);
              good_arg = true;
              break;
            }
            else if ( (*arg_it)->name() == val_name) {
              valid_arg &= (*arg_it)->parse_args(args, out, err, _help_flag);
              good_arg = true;
              break;
            }
            
          }
          
          if (good_arg) unset_args.erase(arg_it);
          
          if (cat_name == "help") {

            print_help_header(out, argv[0]);
            print_help(out, false);
            print_help_footer(out, argv[0]);

            _help_flag |= true;
            
            args.clear();
            
            return true;
            
          } else if (cat_name == "help-all") {
            
            print_help(out, true);
            _help_flag |= true;
            args.clear();
            
            return true;
            
          }
          
          if (!good_arg && err) 
            *err << cat_name << " is either mistyped or misplaced." << std::endl;
          
        }
        
        if (_help_flag)
          return true;
        
        if (!_method_flag)
          *err << "A method must be specified!" << std::endl;
        
        return valid_arg && good_arg && _method_flag;
      }
      
      void print(std::ostream* s, const char prefix = '\0') {
        if (!s) 
          return;

        for (int i = 0; i < _arguments.size(); ++i) {
          _arguments.at(i)->print(s, 0, prefix);
        }
        
      }
      
      void print_help_header(std::ostream* s, const char* executable) {
        if (!s)
          return;
        
        *s << "Usage: " << executable << " <arg1> <subarg1_1> ... <subarg1_m>"
           << " ... <arg_n> <subarg_n_1> ... <subarg_n_m>"
           << std::endl << std::endl;
        *s << "Valid arguments to Stan:" << std::endl << std::endl;
        
      }
      
      void print_help(std::ostream* s, bool recurse) {
        if (!s) 
          return;
        
        for (int i = 0; i < _arguments.size(); ++i) {
          _arguments.at(i)->print_help(s, 1, recurse);
        }
      }
    
      void print_help_footer(std::ostream* s, const char* executable) {
        if (!s)
          return;
        
        *s << std::endl;
        *s << "See '" << executable << "' <arg1> [ help | help-all ]"
           << " for argument specific information" << std::endl;
        *s << " or '" << executable << "' help-all for all argument information" << std::endl;
        
      }
      
      argument* arg(std::string name) {
        for (std::vector<argument*>::iterator it = _arguments.begin();
             it != _arguments.end(); ++it)
          if ( name == (*it)->name() ) 
            return (*it);
        return 0;
      }
      
      bool help_printed() { 
        return _help_flag; 
      }
      
    protected:
      
      std::vector<argument*>& _arguments;
      
      // We can also check for, and warn the user of, deprecated arguments
      //std::vector<argument*> deprecated_arguments;
      // check_arg_conflict // Ensure non-zero intersection of valid and deprecated arguments
      
      bool _help_flag;
      bool _method_flag;
      
    };
    
  } // gm
  
} // stan

#endif

