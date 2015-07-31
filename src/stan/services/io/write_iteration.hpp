#ifndef STAN_SERVICES_IO_WRITE_ITERATION_HPP
#define STAN_SERVICES_IO_WRITE_ITERATION_HPP

#include <stan/services/io/write_iteration_csv.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace services {
    namespace io {

      template <class Writer, class Model, class RNG>
      void write_iteration(Writer& writer,
                           Model& model,
                           RNG& base_rng,
                           double lp,
                           std::vector<double>& cont_vector,
                           std::vector<int>& disc_vector) {
        std::vector<double> values;
        model.write_array(base_rng, cont_vector, disc_vector,
                          values, true, true);
        values.insert(values.begin(), lp);
        writer(values);
      }

    }
  }
}

#endif
