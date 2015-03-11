#ifndef STAN__SERVICES__IO__WRITE_STAN_HPP
#define STAN__SERVICES__IO__WRITE_STAN_HPP

#include <string>
#include <stan/version.hpp>

namespace stan {
  namespace services {
    namespace io {

      template <class Writer>
      void write_stan(Writer& writer, const std::string& prefix = "") {
        writer.write_message(prefix + " stan_version_major = " + stan::MAJOR_VERSION);
        writer.write_message(prefix + " stan_version_minor = " + stan::MINOR_VERSION);
        writer.write_message(prefix + " stan_version_patch = " + stan::PATCH_VERSION);
      }

    }
  }
}

#endif
