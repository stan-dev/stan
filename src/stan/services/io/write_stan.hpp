#ifndef STAN__SERVICES__IO__WRITE_STAN_HPP
#define STAN__SERVICES__IO__WRITE_STAN_HPP

#include <ostream>
#include <string>
#include <stan/version.hpp>

namespace stan {
  namespace services {
    namespace io {

      void write_stan(std::ostream* s, const std::string prefix = "") {
        if (!s) return;

        *s << prefix << " stan_version_major = " << stan::MAJOR_VERSION << std::endl;
        *s << prefix << " stan_version_minor = " << stan::MINOR_VERSION << std::endl;
        *s << prefix << " stan_version_patch = " << stan::PATCH_VERSION << std::endl;
      }

    }
  }
}

#endif
