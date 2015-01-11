#ifndef STAN__COMMON__WRITE_ITERATION_HPP
#define STAN__COMMON__WRITE_ITERATION_HPP

#include <ostream>
#include <vector>
#include <stan/common/write_iteration_csv.hpp>

// FIXME: write_iteration calls std::cout directly.
//   once removed, remove this include
#include <iostream>


namespace stan {
  
  namespace common {
    

    template <class Model, class RNG>
    void write_iteration(std::ostream& output_stream,
                         Model& model,
                         RNG& base_rng,
                         double lp,
                         std::vector<double>& cont_vector,
                         std::vector<int>& disc_vector) {
      std::vector<double> model_values;
      model.write_array(base_rng, cont_vector, disc_vector, model_values,
                        true, true, &std::cout);  
      write_iteration_csv(output_stream, lp, model_values);
    }

  } // namespace common

} // namespace stan

#endif
