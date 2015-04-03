#ifndef STAN_SERVICES_IO_WRITE_STAN_HPP
#define STAN_SERVICES_IO_WRITE_STAN_HPP

#include <string>
#include <stan/version.hpp>

namespace stan {
  namespace services {
    namespace io {

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
