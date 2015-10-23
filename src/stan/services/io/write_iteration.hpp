#ifndef STAN_SERVICES_IO_WRITE_ITERATION_HPP
#define STAN_SERVICES_IO_WRITE_ITERATION_HPP

#include <stan/services/io/write_iteration_csv.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace services {
    namespace io {

      template <class Model, class RNG>
      void write_iteration(std::ostream& output_stream,
                           Model& model,
                           RNG& base_rng,
                           double lp,
                           std::vector<double>& cont_vector,
                           std::vector<int>& disc_vector,
                           std::ostream* o) {
        std::vector<double> model_values;
        model.write_array(base_rng, cont_vector, disc_vector, model_values,
                          true, true, o);
        write_iteration_csv(output_stream, lp, model_values);
      }

    }
  }
}

#endif
