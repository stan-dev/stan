#ifndef STAN_SERVICES_SAMPLE_SAMPLE_HPP
#define STAN_SERVICES_SAMPLE_SAMPLE_HPP

#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/services/arguments/arg_adapt.hpp>
#include <stan/services/arguments/arg_adapt_delta.hpp>
#include <stan/services/arguments/arg_adapt_engaged.hpp>
#include <stan/services/arguments/arg_adapt_gamma.hpp>
#include <stan/services/arguments/arg_adapt_init_buffer.hpp>
#include <stan/services/arguments/arg_adapt_kappa.hpp>
#include <stan/services/arguments/arg_adapt_t0.hpp>
#include <stan/services/arguments/arg_adapt_term_buffer.hpp>
#include <stan/services/arguments/arg_adapt_window.hpp>
#include <stan/services/arguments/arg_bfgs.hpp>
#include <stan/services/arguments/arg_data.hpp>
#include <stan/services/arguments/arg_data_file.hpp>
#include <stan/services/arguments/arg_dense_e.hpp>
#include <stan/services/arguments/arg_diag_e.hpp>
#include <stan/services/arguments/arg_diagnose.hpp>
#include <stan/services/arguments/arg_diagnostic_file.hpp>
#include <stan/services/arguments/arg_engine.hpp>
#include <stan/services/arguments/arg_fail.hpp>
#include <stan/services/arguments/arg_fixed_param.hpp>
#include <stan/services/arguments/arg_history_size.hpp>
#include <stan/services/arguments/arg_hmc.hpp>
#include <stan/services/arguments/arg_id.hpp>
#include <stan/services/arguments/arg_init.hpp>
#include <stan/services/arguments/arg_init_alpha.hpp>
#include <stan/services/arguments/arg_int_time.hpp>
#include <stan/services/arguments/arg_iter.hpp>
#include <stan/services/arguments/arg_lbfgs.hpp>
#include <stan/services/arguments/arg_max_depth.hpp>
#include <stan/services/arguments/arg_method.hpp>
#include <stan/services/arguments/arg_metric.hpp>
#include <stan/services/arguments/arg_newton.hpp>
#include <stan/services/arguments/arg_num_samples.hpp>
#include <stan/services/arguments/arg_num_warmup.hpp>
#include <stan/services/arguments/arg_nuts.hpp>
#include <stan/services/arguments/arg_optimize.hpp>
#include <stan/services/arguments/arg_optimize_algo.hpp>
#include <stan/services/arguments/arg_output.hpp>
#include <stan/services/arguments/arg_output_file.hpp>
#include <stan/services/arguments/arg_random.hpp>
#include <stan/services/arguments/arg_refresh.hpp>
#include <stan/services/arguments/arg_rwm.hpp>
#include <stan/services/arguments/arg_sample.hpp>
#include <stan/services/arguments/arg_sample_algo.hpp>
#include <stan/services/arguments/arg_save_iterations.hpp>
#include <stan/services/arguments/arg_save_warmup.hpp>
#include <stan/services/arguments/arg_seed.hpp>
#include <stan/services/arguments/arg_static.hpp>
#include <stan/services/arguments/arg_stepsize.hpp>
#include <stan/services/arguments/arg_stepsize_jitter.hpp>
#include <stan/services/arguments/arg_test.hpp>
#include <stan/services/arguments/arg_test_grad_eps.hpp>
#include <stan/services/arguments/arg_test_grad_err.hpp>
#include <stan/services/arguments/arg_test_gradient.hpp>
#include <stan/services/arguments/arg_thin.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>
#include <stan/services/arguments/arg_unit_e.hpp>
#include <stan/services/arguments/argument.hpp>
#include <stan/services/arguments/argument_parser.hpp>
#include <stan/services/arguments/argument_probe.hpp>
#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/list_argument.hpp>
#include <stan/services/arguments/singleton_argument.hpp>
#include <stan/services/arguments/unvalued_argument.hpp>
#include <stan/services/arguments/valued_argument.hpp>

#include <stan/services/error_codes.hpp>

#include <stan/services/sample/init_adapt.hpp>
#include <stan/services/sample/init_nuts.hpp>
#include <stan/services/sample/init_static_hmc.hpp>
#include <stan/services/sample/init_windowed_adapt.hpp>
#include <stan/services/sample/mcmc_writer.hpp>
#include <stan/services/sample/progress.hpp>
#include <stan/services/sample/run_adaptive_sampler.hpp>
#include <stan/services/sample/run_sampler.hpp>
#include <stan/services/sample/generate_transitions.hpp>

#include <cmath>
#include <sstream>
#include <iomanip>
#include <string>

namespace stan {
  namespace services {
    namespace sample {

      /**
       * @tparam Model Model implementation
       * @tparam RNG Random number generator implementation
       * @tparam InfoWriter An implementation of
       *                    src/stan/interface_callbacks/writer/base_writer.hpp
       * @tparam ErrWriter An implementation of
       *                   src/stan/interface_callbacks/writer/base_writer.hpp
       * @tparam OutputWriter An implementation of
       *                    src/stan/interface_callbacks/writer/base_writer.hpp
       * @tparam Diagnostic Writer An implementation of
       *                           src/stan/interface_callbacks/writer/base_writer.hpp
       * @tparam Interrupt Interrupt callback implementation
       * @param cont_params Continuous state values
       * @param model Model
       * @param base_rng Random number generator
       * @param sample_args Sampling configuration
       * @param refresh Progress update rate
       
       * @param lp Log posterior density
       * @param cont_vector Continuous state values
       * @param disc_vector Discrete state values
       * @param info Writer callback for displaying informative messages
       * @param err Writer callback for displaying error messages
       * @param output Writer callback for storing sampling history
       * @param dianostic Writer callback for storing sampling diagnostic history
       * @param iteration_interrupt Interrupt callback called at the beginning
                                    of each iteration
       */
      template <class Model, class RNG,
                class InfoWriter, class ErrWriter,
                class OutputWriter, class DiagnosticWriter,
                class Interrupt>
      int sample(Eigen::VectorXd& cont_params,
                 Model& model,
                 RNG& base_rng,
                 stan::services::categorical_argument* sample_args,
                 int refresh,
                 InfoWriter& info,
                 ErrWriter& err,
                 OutputWriter& output,
                 DiagnosticWriter& diagnostic,
                 Interrupt& iteration_interrupt) {
        // Check timing
        clock_t start_check = clock();

        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());

        stan::model::gradient(model, cont_params, init_log_prob, init_grad);

        clock_t end_check = clock();
        double deltaT
          = static_cast<double>(end_check - start_check) / CLOCKS_PER_SEC;

        std::stringstream msg;
        msg << "Gradient evaluation took " << deltaT << " seconds";
        info(msg.str());

        msg.str(std::string());
        msg.clear();
        msg << "1000 transitions using 10 leapfrog steps "
            << "per transition would take " << 1e4 * deltaT << " seconds.";
        info(msg.str());

        info("Adjust your expectations accordingly!");
        info();

        // Sampling parameters
        int num_warmup = dynamic_cast<stan::services::int_argument*>
                         (sample_args->arg("num_warmup"))->value();
        int num_samples = dynamic_cast<stan::services::int_argument*>
                          (sample_args->arg("num_samples"))->value();
        int num_thin = dynamic_cast<stan::services::int_argument*>
                       (sample_args->arg("thin"))->value();
        bool save_warmup = dynamic_cast<stan::services::bool_argument*>
                           (sample_args->arg("save_warmup"))->value();

        // Sample!
        stan::services::list_argument* algo
          = dynamic_cast<stan::services::list_argument*>
          (sample_args->arg("algorithm"));
        stan::services::categorical_argument* adapt
          = dynamic_cast<stan::services::categorical_argument*>
            (sample_args->arg("adapt"));
        bool adapt_engaged
          = dynamic_cast<stan::services::bool_argument*>
            (adapt->arg("engaged"))->value();

        mcmc_writer<Model, RNG,
                    OutputWriter, DiagnosticWriter,
                    InfoWriter, ErrWriter>
          writer(model, base_rng, output, diagnostic, info, err);

        stan::mcmc::sample s(cont_params, 0, 0);

        if (algo->value() == "fixed_param") {
          if (model.num_params_r() == 0 && algo->value() != "fixed_param") {
            err(std::string("Must use algorithm=fixed_param for ")
                + "model that has no parameters.");
            return stan::services::error_codes::USAGE;
          }

          if (num_warmup != 0) {
            info(std::string("Warning: warmup will be skipped ")
                               + "for the fixed parameter sampler!");
            num_warmup = 0;
          }

          stan::mcmc::fixed_param_sampler sampler;

          sample::run_sampler(sampler, s, num_warmup, num_samples,
                              num_thin, refresh, save_warmup,
                              writer, iteration_interrupt);

          return stan::services::error_codes::OK;

        } else if (algo->value() == "rwm") {
          info(algo->arg("rwm")->description());
          return stan::services::error_codes::OK;

        } else if (algo->value() == "hmc") {
          stan::services::list_argument* engine
            = dynamic_cast<stan::services::list_argument*>
              (algo->arg("hmc")->arg("engine"));

          stan::services::list_argument* metric
            = dynamic_cast<stan::services::list_argument*>
              (algo->arg("hmc")->arg("metric"));

          if (   engine->value() == "static"
              && metric->value() == "unit_e"
              && adapt_engaged == false) {
            stan::mcmc::unit_e_static_hmc<Model, RNG>
            sampler(model, base_rng);

            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;

            sample::run_sampler(sampler, s, num_warmup, num_samples,
                                num_thin, refresh, save_warmup,
                                writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "nuts"
              && metric->value() == "unit_e"
              && adapt_engaged == false) {
            stan::mcmc::unit_e_nuts<Model, RNG>
            sampler(model, base_rng);

            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;

            sample::run_sampler(sampler, s, num_warmup, num_samples,
                                num_thin, refresh, save_warmup,
                                writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "static"
              && metric->value() == "diag_e"
              && adapt_engaged == false) {
            stan::mcmc::diag_e_static_hmc<Model, RNG>
            sampler(model, base_rng);

            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;

            sample::run_sampler(sampler, s, num_warmup, num_samples,
                                num_thin, refresh, save_warmup,
                                writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "nuts"
              && metric->value() == "diag_e"
              && adapt_engaged == false) {
            stan::mcmc::diag_e_nuts<Model, RNG>
            sampler(model, base_rng);

            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;

            sample::run_sampler(sampler, s, num_warmup, num_samples,
                                num_thin, refresh, save_warmup,
                                writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "static"
              && metric->value() == "dense_e"
              && adapt_engaged == false) {
            stan::mcmc::dense_e_static_hmc<Model, RNG>
            sampler(model, base_rng);

            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;

            sample::run_sampler(sampler, s, num_warmup, num_samples,
                                num_thin, refresh, save_warmup,
                                writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "nuts"
              && metric->value() == "dense_e"
              && adapt_engaged == false) {
            stan::mcmc::dense_e_nuts<Model, RNG>
            sampler(model, base_rng);

            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;

            sample::run_sampler(sampler, s, num_warmup, num_samples,
                                num_thin, refresh, save_warmup,
                                writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "static"
              && metric->value() == "unit_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_unit_e_static_hmc<Model, RNG>
            sampler(model, base_rng);

            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_adapt(sampler, adapt, cont_params, err))
              return stan::services::error_codes::SOFTWARE;

            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "nuts"
              && metric->value() == "unit_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_unit_e_nuts<Model, RNG>
            sampler(model, base_rng);

            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_adapt(sampler, adapt, cont_params, err))
              return stan::services::error_codes::SOFTWARE;

            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "static"
              && metric->value() == "diag_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_diag_e_static_hmc<Model, RNG>
            sampler(model, base_rng);

            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt,
                                     num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;

            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "nuts"
              && metric->value() == "diag_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_diag_e_nuts<Model, RNG>
            sampler(model, base_rng);

            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt,
                                     num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;

            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "static"
              && metric->value() == "dense_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_dense_e_static_hmc<Model, RNG>
            sampler(model, base_rng);

            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt,
                                     num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;

            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   engine->value() == "nuts"
              && metric->value() == "dense_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_dense_e_nuts<Model, RNG>
            sampler(model, base_rng);

            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt,
                                     num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;

            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);

            return stan::services::error_codes::OK;
          }

          if (   !(engine->value() == "static"
                  || engine->value() == "nuts")
              || !(metric->value() == "unit_e"
                   || metric->value() == "diag_e"
                   || metric->value() == "dense_e")) {
              err("No sampler matching HMC specification!");
               return stan::services::error_codes::USAGE;
          }
        }

        return stan::services::error_codes::USAGE;
      }

    }  // sample
  }  // services
}  // stan

#endif
