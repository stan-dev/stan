#ifndef STAN_SERVICES_IO_WRITE_MODEL_HPP
#define STAN_SERVICES_IO_WRITE_MODEL_HPP

#include <string>

namespace stan {
  namespace services {
    namespace io {

      /**
       * @tparam Writer An implementation of
       *                src/stan/interface_callbacks/writer/base_writer.hpp
       * @param writer Writer callback
       * @param model_name Name of model
       * @param prefix Prefix
       */
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
