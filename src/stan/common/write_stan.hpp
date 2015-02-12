#ifndef STAN__COMMON__WRITE_STAN_HPP
#define STAN__COMMON__WRITE_STAN_HPP

#include <stan/version.hpp>
#include <ostream>
#include <string>

namespace stan {

  namespace common {

    void write_stan(std::ostream* s, const std::string prefix = "") {
      if (!s)
        return;

      *s << prefix << " stan_version_major = "
         << stan::MAJOR_VERSION << std::endl;
      *s << prefix << " stan_version_minor = "
         << stan::MINOR_VERSION << std::endl;
      *s << prefix << " stan_version_patch = "
         << stan::PATCH_VERSION << std::endl;
    }

  }  // namespace common

}  // namespace stan

#endif
