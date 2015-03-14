#ifndef STAN__SERVICES__ARGUMENTS__LIST__ARGUMENT__BETA
#define STAN__SERVICES__ARGUMENTS__LIST__ARGUMENT__BETA

#include <stan/services/arguments/valued_argument.hpp>
#include <stan/services/arguments/arg_fail.hpp>

namespace stan {
  
  namespace services {
    
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

      template <class Writer>
      void print(Writer& writer, int depth, const std::string prefix) {
        valued_argument::print(writer, depth, prefix);
        _values.at(_cursor)->print(writer, depth + 1, prefix);
      }
      
      template <class Writer>
      void print_help(Writer& writer, int depth, bool recurse) {
        std::cout << "list_argument: Trying to write some fucking messages!" << std::endl;
        _default = _values.at(_default_cursor)->name();

        valued_argument::print_help(writer, depth);
        
        if (recurse) {
          for (std::vector<argument*>::iterator it = _values.begin();
               it != _values.end(); ++it)
            (*it)->print_help(writer, depth + 1, true);
        }
      }
      
      template <class InfoWriter, class ErrWriter>
      bool parse_args(std::vector<std::string>& args,
                      InfoWriter& info, ErrWriter& err,
                      bool& help_flag) {
        
        if(args.size() == 0) return true;
        
        std::string name;
        std::string value;
        split_arg(args.back(), name, value);
        
        if(_name == "help") {
          print_help(info, 0, false);
          help_flag |= true;
          args.clear();
          return false;
        }
        else if(_name == "help-all") {
          print_help(info, 0, true);
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
            valid_arg &= _values.at(_cursor)->parse_args(args, info, err, help_flag);
            good_arg = true;
            break;
          }
          
          if(!good_arg) {
            err.write_message(value + " is not a valid value for \"" + _name + "\"");
            err.write_message(std::string(indent_width, ' ') + "Valid values:" + print_valid());
            args.clear();
          }
          
          return valid_arg && good_arg;
          
        }
        
        return true;
        
      };

      template <class Writer>
      void probe_args(argument* base_arg, Writer& writer) {

        for (size_t i = 0; i < _values.size(); ++i) {
          _cursor = i;
          
          writer.write_message("good");
          base_arg->print(writer, 0, "");
          writer.write_message("");

          _values.at(i)->probe_args(base_arg, writer);
        }
        
        _values.push_back(new arg_fail);
        _cursor = _values.size() - 1;
        writer.write_message("bad");
        base_arg->print(writer, 0, "");
        writer.write_message("");
        
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
    
  } // services
  
} // stan

#endif
