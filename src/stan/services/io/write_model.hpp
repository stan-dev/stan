#ifndef STAN_SERVICES_IO_WRITE_MODEL_HPP
#define STAN_SERVICES_IO_WRITE_MODEL_HPP

#include <string>

namespace stan {
  namespace services {
    namespace io {

      template <class Writer>
      void write_model(Writer& writer,
                       const std::string& model_name,
                       const std::string& prefix = "") {
        writer(prefix + " model = " + model_name);
      }

    }
  }
}

#endif
