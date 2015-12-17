#ifndef STAN_SERVICES_IO_WRITE_ITERATION_HPP
#define STAN_SERVICES_IO_WRITE_ITERATION_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace services {
    namespace io {

      template <class Model, class RNG>
      void write_iteration(interface_callbacks::writer::base_writer& writer,
                           Model& model,
                           RNG& base_rng,
                           double lp,
                           std::vector<double>& cont_vector,
                           std::vector<int>& disc_vector) {
        std::vector<double> values;
        model.write_array(base_rng, cont_vector, disc_vector, values,
                          true, true);
        values.insert(values.begin(), lp);
        writer(values);
      }

    }
  }
}

#endif
