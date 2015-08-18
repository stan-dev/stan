#ifndef STAN_SERVICES_IO_WRITE_STAN_HPP
#define STAN_SERVICES_IO_WRITE_STAN_HPP

#include <stan/version.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace services {
    namespace io {

      /**
       * @tparam Writer An implementation of
       *                src/stan/interface_callbacks/writer/base_writer.hpp
       * @param writer Writer callback
       * @param prefix Message Prefix
       */
      template <class Writer>
      void write_stan(Writer& writer, const std::string& prefix = "") {
        writer(prefix + " stan_version_major = " + stan::MAJOR_VERSION);
        writer(prefix + " stan_version_minor = " + stan::MINOR_VERSION);
        writer(prefix + " stan_version_patch = " + stan::PATCH_VERSION);
      }

    }
  }
}

#endif
