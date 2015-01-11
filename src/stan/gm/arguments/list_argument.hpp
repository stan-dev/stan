#ifndef STAN__GM__ARGUMENTS__LIST__ARGUMENT__BETA
#define STAN__GM__ARGUMENTS__LIST__ARGUMENT__BETA

#include <iostream>
#include <stan/gm/arguments/valued_argument.hpp>
#include <stan/gm/arguments/arg_fail.hpp>

namespace stan {
  
  namespace gm {
    
    class list_argument: public valued_argument {
      
    public:
      
      list_argument() {
        _value_type = "list element";
      }
      
      ~list_argument() {
        
        for (std::vector<argument*>::iterator it = _values.begin();
             it != _values.end(); ++it) {
          delete *it;
        }
        
        _values.clear();
        
      }

      void print(std::ostream* s, int depth, const std::string prefix) {
        valued_argument::print(s, depth, prefix);
        _values.at(_cursor)->print(s, depth + 1, prefix);
      }
      
      void print_help(std::ostream* s, int depth, bool recurse) {
        _default = _values.at(_default_cursor)->name();

        valued_argument::print_help(s, depth);
        
        if (recurse) {
          for (std::vector<argument*>::iterator it = _values.begin();
               it != _values.end(); ++it)
            (*it)->print_help(s, depth + 1, true);
        }
      }
      
      bool parse_args(std::vector<std::string>& args, std::ostream* out,
                      std::ostream* err, bool& help_flag) {
        
        if(args.size() == 0) return true;
        
        std::string name;
        std::string value;
        split_arg(args.back(), name, value);
        
        if(_name == "help") {
          print_help(out, 0, false);
          help_flag |= true;
          args.clear();
          return false;
        }
        else if(_name == "help-all") {
          print_help(out, 0, true);
          help_flag |= true;
          args.clear();
          return false;
        }
        else if(_name == name) {
         
          args.pop_back();
          
          bool good_arg = false;
          bool valid_arg = true;
          
          for (size_t i = 0; i < _values.size(); ++i) {
            if( _values.at(i)->name() != value) continue;
            
            _cursor = i;
            valid_arg &= _values.at(_cursor)->parse_args(args, out, err, help_flag);
            good_arg = true;
            break;
          }
          
          if(!good_arg) {
            
            if(err) {
              *err << value << " is not a valid value for \"" << _name << "\"" << std::endl;
              *err << std::string(indent_width, ' ') << "Valid values:" << print_valid() << std::endl;
            }
            
            args.clear();
          }
          
          return valid_arg && good_arg;
          
        }
        
        return true;
        
      };

      virtual void probe_args(argument* base_arg, std::stringstream& s) {

        for (size_t i = 0; i < _values.size(); ++i) {
          _cursor = i;
          
          s << "good" << std::endl;
          base_arg->print(&s, 0, "");
          s << std::endl;

          _values.at(i)->probe_args(base_arg, s);
        }
        
        _values.push_back(new arg_fail);
        _cursor = _values.size() - 1;
        s << "bad" << std::endl;
        base_arg->print(&s, 0, "");
        s << std::endl;
        
        _values.pop_back();
        _cursor = _default_cursor;
        
      }
      
      void find_arg(std::string name,
                    std::string prefix,
                    std::vector<std::string>& valid_paths) {
        
        if (name == _name) {
          valid_paths.push_back(prefix + _name + "=<list_element>");
        }
        
        prefix += _name + "=";
        for (std::vector<argument*>::iterator it = _values.begin();
             it != _values.end(); ++it) {
          std::string value_prefix = prefix + (*it)->name() + " ";
          (*it)->find_arg(name, prefix, valid_paths);
        }
        
      }
      
      bool valid_value(std::string name) {
        for (std::vector<argument*>::iterator it = _values.begin();
             it != _values.end(); ++it)
          if (name == (*it)->name())
            return true;
        return false;
      }
      
      argument* arg(std::string name) {
        if(name == _values.at(_cursor)->name())
          return _values.at(_cursor);
        else
          return 0;
      }
      
      std::vector<argument*>& values() { return _values; }
      
      std::string value() { return _values.at(_cursor)->name(); }
      
      std::string print_value() { return _values.at(_cursor)->name(); }
      
      std::string print_valid() {
        std::string valid_values;
        
        std::vector<argument*>::iterator it = _values.begin();
        valid_values += " " + (*it)->name();
        ++it;
        
        for (; it != _values.end(); ++it)
          valid_values += ", " + (*it)->name();
        
        return valid_values;
        
      }
      
      bool is_default() { return _cursor == _default_cursor; }
      
    protected:
      
      int _cursor;
      int _default_cursor;
      
      std::vector<argument*> _values;
      
    };
    
  } // gm
  
} // stan

#endif
