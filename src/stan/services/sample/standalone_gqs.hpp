#ifndef STAN_SERVICES_SAMPLE_STANDALONE_GQS_HPP
#define STAN_SERVICES_SAMPLE_STANDALONE_GQS_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/gq_writer.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace services {

    // TODO(carpenter): see if params_i can be static
    // w.r.t. Google style:
    // https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables

    // TODO(carpenter): add refresh

    /**
     * Return the number of constrained parameters for the specified
     * model.
     *
     * @tparam M type of model
     * @param[in] model model to query
     * @return number of constrained parameters for the model
     */
    template <class Model>
    int num_constrained_params(const Model& model) {
        std::vector<std::string> param_names;
        static const bool include_tparams = false;
        static const bool include_gqs = false;
        model.constrained_param_names(param_names, include_tparams,
                                      include_gqs);
        return param_names.size();
    }

    /**
     *
     * @tparam M model class
     * @param[in] model instantiated model
     * @param[in] draws sequence of draws of unconstrained parameters
     * @param[in] seed seed to use for randomization
     * @param[in, out] interrupt called every iteration
     * @param[in, out] logger logger to which to write warning and error messages
     * @param[in, out] sample_writer writer to which draws are written
     * @return OK error code (always)
     */
    template <class Model>
    int standalone_generate(const Model& model,
                            const std::vector<std::vector<double> >& draws,
                            unsigned int seed,
                            callbacks::interrupt& interrupt,
                            callbacks::logger& logger,
                            callbacks::writer& sample_writer) {
      const std::vector<int> params_i;
      boost::ecuyer1988 rng = util::create_rng(seed, 1);
      int num_params = num_constrained_params(model);
      util::gq_writer writer(sample_writer, logger, num_params);
      writer.write_gq_names(model);
      for (const std::vector<double>& draw : draws) {
        interrupt();  // call out to interrupt and fail
        writer.write_gq_values(model, rng, draw);
      }
      return error_codes::OK;
    }


  }
}
#endif
