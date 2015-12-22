#ifndef STAN_SERVICES_IO_WRITE_ITERATION_HPP
#define STAN_SERVICES_IO_WRITE_ITERATION_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace services {
    namespace io {

      template <class Model, class RNG>
      void write_iteration(Model& model,
                           RNG& base_rng,
                           double lp,
                           std::vector<double>& cont_vector,
                           std::vector<int>& disc_vector,
                   interface_callbacks::writer::base_writer& message_writer,
                   interface_callbacks::writer::base_writer& parameter_writer) {
        std::vector<double> values;
        std::stringstream ss;
        model.write_array(base_rng, cont_vector, disc_vector, values,
                          true, true, &ss);
        if (ss.str().length() > 0)
          message_writer(ss.str());
        values.insert(values.begin(), lp);
        parameter_writer(values);
      }

    }
  }
}

#endif
