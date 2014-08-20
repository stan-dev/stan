#ifndef STAN__COMMON__CONTEXT_FACTORY_HPP
#define STAN__COMMON__CONTEXT_FACTORY_HPP

#include <fstream>
#include <stan/io/dump.hpp>

namespace stan {
  namespace common {

    class var_context_factory {
    public:
      var_context_factory() {}
      virtual stan::io::var_context* operator()(const std::string source) = 0;
    };

    // FIXME: Move to CmdStan
    class dump_factory: public var_context_factory {
    public:
      stan::io::var_context* operator()(const std::string source) {
        std::fstream source_stream(source.c_str(),
                                   std::fstream::in);
        if (source_stream.fail()) {
          std::string message("ERROR: specified initialization file does not exist: ");
          message += source;
          throw std::runtime_error(message);
        }

        stan::io::var_context* dump = new stan::io::dump(source_stream);
        source_stream.close();

        return dump;
      }
    };
    
  }
}

#endif
