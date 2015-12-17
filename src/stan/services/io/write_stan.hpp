#ifndef STAN_SERVICES_IO_WRITE_STAN_HPP
#define STAN_SERVICES_IO_WRITE_STAN_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/version.hpp>
#include <string>

namespace stan {
  namespace services {
    namespace io {

      void write_stan(interface_callbacks::writer::base_writer& writer,
                      const std::string prefix = "") {
        writer(prefix + " stan_version_major = " + stan::MAJOR_VERSION);
        writer(prefix + " stan_version_minor = " + stan::MINOR_VERSION);
        writer(prefix + " stan_version_patch = " + stan::PATCH_VERSION);
      }

    }
  }
}
#endif
