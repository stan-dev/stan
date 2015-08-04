#ifndef STAN_INTERFACE_CALLBACKS_VAR_CONTEXT_FACTORY_DUMP_FACTORY_HPP
#define STAN_INTERFACE_CALLBACKS_VAR_CONTEXT_FACTORY_DUMP_FACTORY_HPP

#include <stan/interface_callbacks/var_context_factory/var_context_factory.hpp>
#include <stan/io/dump.hpp>
#include <fstream>
#include <string>

namespace stan {
  namespace interface_callbacks {
    namespace var_context_factory {

      // FIXME: Move to CmdStan
      class dump_factory: public var_context_factory<stan::io::dump> {
      public:
        stan::io::dump operator()(const std::string source) {
          std::fstream source_stream(source.c_str(),
                                     std::fstream::in);

          if (source_stream.fail()) {
            std::string message("dump_factory Error: the file "
                                + source + " does not exist.");
            throw std::runtime_error(message);
          }

          stan::io::dump dump(source_stream);
          source_stream.close();

          return dump;
        }
      };

    }
  }
}

#endif
