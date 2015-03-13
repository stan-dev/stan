#ifndef STAN__SERVICES__IO__WRITE_ITERATION_HPP
#define STAN__SERVICES__IO__WRITE_ITERATION_HPP

#include <vector>

// FIXME: write_iteration calls std::cout directly.
//   once removed, remove this include
#include <iostream>

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
        model.write_array(base_rng, cont_vector, disc_vector, values,
                          true, true, &std::cout); /////***** FIXME NOW *****//////
        values.insert(values.begin(), lp);
        writer.write_state(values);
      }

    }
  }
}

#endif
