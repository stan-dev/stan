#ifndef STAN_SERVICES_IO_WRITE_MODEL_HPP
#define STAN_SERVICES_IO_WRITE_MODEL_HPP

#include <ostream>
#include <string>

namespace stan {
  namespace services {
    namespace io {

      void write_model(std::ostream* s,
                       const std::string model_name,
                       const std::string prefix = "") {
        if (!s) return;

        *s << prefix << " model = " << model_name << std::endl;
      }

    }
  }
}

#endif
