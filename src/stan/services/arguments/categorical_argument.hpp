#ifndef STAN_SERVICES_ARGUMENTS_CATEGORICAL_ARGUMENT_HPP
#define STAN_SERVICES_ARGUMENTS_CATEGORICAL_ARGUMENT_HPP

#include <stan/services/arguments/argument.hpp>
#include <string>
#include <vector>

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

      void print(interface_callbacks::writer::base_writer& w,
                 const int depth,
                 const std::string& prefix) {
        std::string indent(compute_indent(depth), ' ');
        w(prefix + indent + _name);

        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it)
          (*it)->print(w, depth + 1, prefix);
      }

      void print_help(interface_callbacks::writer::base_writer& w,
                      const int depth,
                      const bool recurse) {
        std::string indent(indent_width * depth, ' ');
        std::string subindent(indent_width, ' ');

        w(indent + _name);
        w(indent + subindent + _description);
        if (_subarguments.size() > 0) {
          std::stringstream ss;
          ss << indent << subindent << "Valid subarguments:";

          std::vector<argument*>::iterator it = _subarguments.begin();
          ss << " " << (*it)->name();
          ++it;

          for (; it != _subarguments.end(); ++it)
            ss << ", " << (*it)->name();
          w(ss.str());
          w();

          if (recurse) {
            for (std::vector<argument*>::iterator it = _subarguments.begin();
                 it != _subarguments.end(); ++it)
              (*it)->print_help(w, depth + 1, true);
          }
        } else {
          w();
        }
      }

      bool parse_args(std::vector<std::string>& args,
                      interface_callbacks::writer::base_writer& info,
                      interface_callbacks::writer::base_writer& err,
                      bool& help_flag) {
        bool good_arg = true;
        bool valid_arg = true;

        while (good_arg && valid_arg) {
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

          if (val_name == this->name())
            return false;

          if (_subarguments.size() == 0)
            valid_arg = true;
          for (std::vector<argument*>::iterator it = _subarguments.begin();
               it != _subarguments.end(); ++it) {
            if ((*it)->name() == cat_name) {
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
      }

      virtual void probe_args(argument* base_arg,
                              interface_callbacks::writer::base_writer& w) {
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it) {
          (*it)->probe_args(base_arg, w);
        }
      }

      void find_arg(const std::string& name,
                    const std::string& prefix,
                    std::vector<std::string>& valid_paths) {
        argument::find_arg(name, prefix, valid_paths);

        std::string new_prefix = prefix + _name + " ";
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it)
          (*it)->find_arg(name, new_prefix, valid_paths);
      }

      std::vector<argument*>& subarguments() {
        return _subarguments;
      }

      argument* arg(const std::string& name) {
        for (std::vector<argument*>::iterator it = _subarguments.begin();
             it != _subarguments.end(); ++it)
          if ( name == (*it)->name() )
            return (*it);
        return 0;
      }

    protected:
      std::vector<argument*> _subarguments;
    };

  }  // services
}  // stan

#endif
