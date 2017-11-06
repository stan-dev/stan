#ifndef STAN_SERVICES_SAMPLE_STANDALONE_GQS_HPP
#define STAN_SERVICES_SAMPLE_STANDALONE_GQS_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/gq_writer.hpp>
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
     * @param[in] m model to query
     * @return number of constrained parameters for the model
     */
    template <class M>
    int num_constrained_params(const M& m) {
        std::vector<std::string> param_names;
        const static bool include_tparams = false;
        const static bool include_gqs = false;
        model.constrained_param_names(param_names, include_tparams, include_gqs);
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
    template <class M>
    int standalone_generate(const M& model,
                            const std::vector<std::vector<double> >& draws,  // pre-parameters
                            unsigned int seed,
                            callbacks::interrupt& interrupt,
                            callbacks::logger& logger,
                            callbacks::writer& sample_writer) {
      static const bool include_tparams = false;
      static const bool include_gqs = true;
      const std::vector<int> params_i;
      boost::ecuyer1988 rng = util::create_rng(seed, 1);
      int num_constrained_params = num_constrained_params(model);
      qq_writer writer(sample_writer, logger, num_constrained_params);
      writer.write_gq_names(model);
      for (const std::vector<double>& draw : draws) {
        interrupt();  // call out to interrupt and fail
        writer.write_gq_values(model, draw);
      }
      return return error_codes::OK;
    }


  }
}
#endif
