#ifndef STAN_SERVICES_ARGUMENTS_ARGUMENT_PARSER_HPP
#define STAN_SERVICES_ARGUMENTS_ARGUMENT_PARSER_HPP

#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/arg_method.hpp>
#include <stan/services/error_codes.hpp>
#include <cstring>
#include <string>
#include <vector>

namespace stan {
  namespace services {

    class argument_parser {
    public:
      explicit argument_parser(std::vector<argument*>& valid_args)
        : _arguments(valid_args),
          _help_flag(false),
          _method_flag(false) {
        _arguments.insert(_arguments.begin(), new arg_method());
      }

      int parse_args(int argc,
                     const char* argv[],
                     interface_callbacks::writer::base_writer& info,
                     interface_callbacks::writer::base_writer& err) {
        if (argc == 1) {
          print_usage(info, argv[0]);
          return error_codes::USAGE;
        }

        std::vector<std::string> args;

        // Fill in reverse order as parse_args pops from the back
        for (int i = argc - 1; i > 0; --i)
          args.push_back(std::string(argv[i]));

        bool good_arg = true;
        bool valid_arg = true;
        _help_flag = false;

        std::vector<argument*> unset_args = _arguments;

        while (good_arg) {
          if (args.size() == 0)
            break;

          good_arg = false;
          std::string cat_name = args.back();

          // Check for method arguments entered without the method= prefix
          if (!_method_flag) {
            list_argument* method
              = dynamic_cast<list_argument*>(_arguments.front());

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

          for (arg_it = unset_args.begin();
               arg_it != unset_args.end(); ++arg_it) {
            if ((*arg_it)->name() == cat_name) {
              args.pop_back();
              valid_arg &= (*arg_it)->parse_args(args, info, err, _help_flag);
              good_arg = true;
              break;
            } else if ((*arg_it)->name() == val_name) {
              valid_arg &= (*arg_it)->parse_args(args, info, err, _help_flag);
              good_arg = true;
              break;
            }
          }

          if (good_arg) unset_args.erase(arg_it);

          if (cat_name == "help") {
            _help_flag |= true;
            args.clear();
          } else if (cat_name == "help-all") {
            print_help(info, true);
            _help_flag |= true;
            args.clear();
          }

          if (_help_flag) {
            print_usage(info, argv[0]);
            return error_codes::OK;
          }

          if (!good_arg) {
            err(cat_name + " is either mistyped or misplaced.");

            std::vector<std::string> valid_paths;

            for (size_t i = 0; i < _arguments.size(); ++i) {
              _arguments.at(i)->find_arg(val_name, "", valid_paths);
            }

            if (valid_paths.size()) {
              err("Perhaps you meant one of the following "
                  "valid configurations?");
              for (size_t i = 0; i < valid_paths.size(); ++i)
                err("  " + valid_paths.at(i));
            }
          }
        }

        if (_help_flag)
          return error_codes::OK;

        if (!_method_flag)
          err("A method must be specified!");

        return (valid_arg && good_arg && _method_flag)
          ? error_codes::OK : error_codes::USAGE;
      }

      void print(interface_callbacks::writer::base_writer& w,
                 const std::string& prefix = "") {
        for (size_t i = 0; i < _arguments.size(); ++i) {
          _arguments.at(i)->print(w, 0, prefix);
        }
      }

      void print_help(interface_callbacks::writer::base_writer& w,
                      bool recurse) {
        for (size_t i = 0; i < _arguments.size(); ++i) {
          _arguments.at(i)->print_help(w, 1, recurse);
        }
      }

      void print_usage(interface_callbacks::writer::base_writer& w,
                       const char* executable) {
        std::string indent(2, ' ');
        int width = 12;

        w(std::string("Usage: ") + executable
          + " <arg1> <subarg1_1> ... <subarg1_m>"
          + " ... <arg_n> <subarg_n_1> ... <subarg_n_m>");
        w();

        w("Begin by selecting amongst the following inference methods"
          " and diagnostics,");

        std::vector<argument*>::iterator arg_it = _arguments.begin();
        list_argument* method = dynamic_cast<list_argument*>(*arg_it);

        std::stringstream ss;
        ss << std::left;
        for (std::vector<argument*>::iterator value_it
               = method->values().begin();
             value_it != method->values().end(); ++value_it) {
          ss.str("");
          ss << std::setw(width)
             << indent + (*value_it)->name()
             << indent + (*value_it)->description();
          w(ss.str());
        }
        w();

        w("Or see help information with");
        ss.str("");
        ss << std::setw(width)
           << indent + "help"
           << indent + "Prints help";
        w(ss.str());
        ss.str("");
        ss << std::setw(width)
           << indent + "help-all"
           << indent + "Prints entire argument tree";
        w(ss.str());
        w();

        w("Additional configuration available by specifying");

        ++arg_it;
        for (; arg_it != _arguments.end(); ++arg_it) {
          ss.str("");
          ss << std::setw(width)
             << indent + (*arg_it)->name()
             << indent + (*arg_it)->description();
          w(ss.str());
        }

        w();
        w(std::string("See ") + executable + " <arg1> [ help | help-all ] "
          + "for details on individual arguments.");
        w();
      }

      argument* arg(const std::string& name) {
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
      // std::vector<argument*> deprecated_arguments;
      // check_arg_conflict
      // Ensure non-zero intersection of valid and deprecated arguments

      bool _help_flag;
      bool _method_flag;
    };

  }  // services
}  // stan

#endif
