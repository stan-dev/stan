#ifndef STAN_SERVICES_UTIL_INITIALIZE_HPP
#define STAN_SERVICES_UTIL_INITIALIZE_HPP

#include <stan/io/var_context.hpp>
#include <stan/io/random_var_context.hpp>
#include <stan/io/chained_var_context.hpp>
#include <sstream>
#include <vector>

namespace stan {
  namespace services {
    namespace util {


      template <class Model, class RNG>
      std::vector<double> initialize(Model& model,
                                     stan::io::var_context& init,
                                     RNG& rng,
                                     double init_radius,
                                     interface_callbacks::writer::base_writer& message_writer) {
        std::vector<double> unconstrained;
        stan::io::random_var_context random_context(model, rng, init_radius);
        stan::io::chained_var_context context(init, random_context);

        std::stringstream msg;
        std::vector<int> disc_vector;
        model.transform_inits(context,
                              disc_vector,
                              unconstrained,
                              &msg);
        message_writer(msg.str());
                              
        return unconstrained;
      }
      
    }
  }
}

#endif
