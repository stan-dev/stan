#ifndef STAN__SERVICES__ARGUMENTS__CATEGORY__ARGUMENT__BETA
#define STAN__SERVICES__ARGUMENTS__CATEGORY__ARGUMENT__BETA

#include <vector>
#include <stan/services/arguments/argument.hpp>

namespace stan {
  
  namespace services {
    
    class categorical_argument: public argument {
      
    public:
      
      virtual ~categorical_argument() {
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it) {
          delete *it;
        }
        
        _subarguments.clear();
      }
      
      template <class Writer>
      void print(Writer& writer, const int depth, const std::string prefix) {
        if (!s)
          return;
        std::string indent(compute_indent(depth), ' ');
        writer.write_message(prefix + indent + _name);
        
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it)
          (*it)->print(writer, depth + 1, prefix);
      }
      
      template <class Writer>
      void print_help(Writer& writer, const int depth, const bool recurse) {
        
        if (!s) 
          return;
        
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');
        
        writer.write_message(indent + _name);
        writer.write_message(indent + subindent + _description);
        if (_subarguments.size() > 0) {
          std::string msg = indent + subindent + "Valid subarguments:";
          
          std::vector<argument*>::iterator it = _subarguments.begin();
          message +=  " " + (*it)->name();
          ++it;
          
          for (; it != _subarguments.end(); ++it)
            msg +=  ", " + (*it)->name();
        
          writer.write_message(msg);
          writer.write_message("");
          
          if (recurse) {
            for (std::vector<argument*>::iterator it = _subarguments.begin();
                 it != _subarguments.end(); ++it)
              (*it)->print_help(writer, depth + 1, true);
          }
        }
        else {
          writer.write_message("");
        }
         
      }
      
      template <class InfoWriter, class ErrWriter>
      bool parse_args(std::vector<std::string>& args,
                      InfoWriter& info, ErrWriter& err,
                      bool& help_flag) {

        bool good_arg = true;
        bool valid_arg = true;
        
        while (good_arg) {
          if (args.size() == 0)
            return valid_arg;
          
          good_arg = false;
          
          std::string cat_name = args.back();
          
          if (cat_name == "help") {
            print_help(info, 0, false);
            help_flag |= true;
            args.clear();
            return true;
          } else if (cat_name == "help-all") {
            print_help(info, 0, true);
            help_flag |= true;
            args.clear();
            return true;
          }
          
          std::string val_name;
          std::string val;
          split_arg(cat_name, val_name, val);
          
          if (_subarguments.size() == 0)
            valid_arg = true;
          for (std::vector<argument*>::iterator it = _subarguments.begin();
               it != _subarguments.end(); ++it) {
            if ( (*it)->name() == cat_name) {
              args.pop_back();
              valid_arg &= (*it)->parse_args(args, info, err, help_flag);
              good_arg = true;
              break;
            } else if ( (*it)->name() == val_name ) {
              valid_arg &= (*it)->parse_args(args, info, err, help_flag);
              good_arg = true;
              break;
            } else {
              good_arg = false;
            }
          }
        }
        return valid_arg;
      };
      
      virtual void probe_args(argument* base_arg, std::stringstream& s) {
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it) {
          (*it)->probe_args(base_arg, s);
        }
      }
      
      void find_arg(std::string name,
                    std::string prefix,
                    std::vector<std::string>& valid_paths) {
        
        argument::find_arg(name, prefix, valid_paths);
        
        prefix += _name + " ";
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it)
          (*it)->find_arg(name, prefix, valid_paths);
        
      }
      
      std::vector<argument*>& subarguments() {
        return _subarguments;
      }
      
      argument* arg(const std::string name) {
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it)
          if ( name == (*it)->name() ) 
            return (*it);
        return 0;
      }
      
    protected:
      
      std::vector<argument*> _subarguments;
      
    };
    
  } // services
  
} // stan

#endif
