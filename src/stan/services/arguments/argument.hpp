#ifndef STAN_SERVICES_ARGUMENTS_ARGUMENT_HPP
#define STAN_SERVICES_ARGUMENTS_ARGUMENT_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
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

      explicit argument(const std::string& name)
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

      virtual void print(interface_callbacks::writer::base_writer& w,
                         const int depth,
                         const std::string& prefix) = 0;

      virtual void print_help(interface_callbacks::writer::base_writer& w,
                              const int depth,
                              const bool recurse) = 0;

      virtual bool parse_args(std::vector<std::string>& args,
                              interface_callbacks::writer::base_writer& info,
                              interface_callbacks::writer::base_writer& err,
                              bool& help_flag) {
        return true;
      }

      virtual void probe_args(argument* base_arg,
                              interface_callbacks::writer::base_writer& w) {}

      virtual void find_arg(const std::string& name,
                            const std::string& prefix,
                            std::vector<std::string>& valid_paths) {
        if (name == _name) {
          valid_paths.push_back(prefix + _name);
        }
      }

      static void split_arg(const std::string& arg,
                            std::string& name,
                            std::string& value) {
        size_t pos = arg.find('=');

        if (pos != std::string::npos) {
          name = arg.substr(0, pos);
          value = arg.substr(pos + 1, arg.size() - pos);
        } else {
          name = arg;
          value = "";
        }
      }

      virtual argument* arg(const std::string& name) {
        return 0;
      }

      int compute_indent(const int depth) {
        return indent_width * depth;
      }

    protected:
      std::string _name;
      std::string _description;

      int indent_width;
      int help_width;
    };

  }  // services
}  // stan
#endif
