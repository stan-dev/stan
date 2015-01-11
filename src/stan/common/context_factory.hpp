#ifndef STAN__COMMON__CONTEXT_FACTORY_HPP
#define STAN__COMMON__CONTEXT_FACTORY_HPP

#include <fstream>
#include <stan/io/dump.hpp>

namespace stan {
  namespace common {

    template <typename VARCON>
    class var_context_factory {
    public:
      var_context_factory() {}
      virtual VARCON operator()(const std::string source) = 0;
      typedef VARCON var_context_t;
    };

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

#endif
