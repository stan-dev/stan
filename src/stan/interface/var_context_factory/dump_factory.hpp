#ifndef STAN__INTERFACE__VAR_CONTEXT_FACTORY__DUMP_FACTORY_HPP
#define STAN__INTERFACE__VAR_CONTEXT_FACTORY__DUMP_FACTORY_HPP

#include <fstream>
#include <stan/io/dump.hpp>
#include <stan/interface/var_context_factory.hpp>

namespace stan {
  namespace interface {
    namespace var_context_factory {

      // FIXME: Move to CmdStan
      class dump_factory: public var_context_factory<stan::io::dump> {
      public:
        stan::io::dump operator()(const std::string source) {
          std::fstream source_stream(source.c_str(),
                                     std::fstream::in);
          
          if (source_stream.fail()) {
            std::string message("dump_factory Error: the file " + source + " does not exist.");
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
