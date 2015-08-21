#ifndef STAN_SERVICES_IO_WRITE_ITERATION_HPP
#define STAN_SERVICES_IO_WRITE_ITERATION_HPP

#include <ostream>
#include <vector>

namespace stan {
  namespace services {
    namespace io {

      /**
       * @tparam Writer An implementation of
       *                src/stan/interface_callbacks/writer/base_writer.hpp
       * @tparam Model Model implementation
       * @tparam RNG Random number generator implementation
       * @param writer Writer callback
       * @param model Model
       * @param base_rng Random number generator
       * @param lp Log posterior density
       * @param cont_vector Continous state values
       * @param disc_vector Discrete state values
       */
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
