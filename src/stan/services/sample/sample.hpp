#ifndef STAN__SERVICES__SAMPLE__SAMPLE_HPP
#define STAN__SERVICES__SAMPLE__SAMPLE_HPP

#include <cmath>
#include <sstream>
#include <iomanip>

#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/services/arguments.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/sample.hpp>

namespace stan {
  namespace services {
    namespace sample {
    
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
        
        stan::model::gradient(model, cont_params, init_log_prob,
                              init_grad, &std::cout); // FIXME
        
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
                    OutputWriter, DiagnosticWriter, InfoWriter>
          writer(model, base_rng, output, diagnostic, info);
        
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
            stan::mcmc::unit_e_static_hmc<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
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
            stan::mcmc::unit_e_nuts<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
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
            stan::mcmc::diag_e_static_hmc<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
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
            stan::mcmc::diag_e_nuts<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
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
            stan::mcmc::dense_e_static_hmc<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
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
            stan::mcmc::dense_e_nuts<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
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
            stan::mcmc::adapt_unit_e_static_hmc<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_adapt(sampler, adapt, cont_params))
              return stan::services::error_codes::SOFTWARE;
            
            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);
            
            return stan::services::error_codes::OK;
          }
          
          if (   engine->value() == "nuts"
              && metric->value() == "unit_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_unit_e_nuts<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_adapt(sampler, adapt, cont_params))
              return stan::services::error_codes::SOFTWARE;
            
            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);
                
            return stan::services::error_codes::OK;
          }
          
          if (   engine->value() == "static"
              && metric->value() == "diag_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_diag_e_static_hmc<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt, num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;
            
            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);
            
            return stan::services::error_codes::OK;
          }
          
          if (   engine->value() == "nuts"
              && metric->value() == "diag_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_diag_e_nuts<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt, num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;
            
            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);
            
            return stan::services::error_codes::OK;
          }
          
          if (   engine->value() == "static"
              && metric->value() == "dense_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_dense_e_static_hmc<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
            if (!init_static_hmc(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt, num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;
            
            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);
            
            return stan::services::error_codes::OK;
          }
          
          if (   engine->value() == "nuts"
              && metric->value() == "dense_e"
              && adapt_engaged == true) {
            stan::mcmc::adapt_dense_e_nuts<Model, RNG, ErrWriter>
            sampler(model, base_rng, err);
            
            if (!init_nuts(sampler, algo))
              return stan::services::error_codes::SOFTWARE;
            if (!init_windowed_adapt(sampler, adapt, num_warmup, cont_params, err))
              return stan::services::error_codes::SOFTWARE;
            
            sample::run_adaptive_sampler(sampler, s, num_warmup, num_samples,
                                         num_thin, refresh, save_warmup,
                                         writer, iteration_interrupt);
            
            return stan::services::error_codes::OK;
          }
          
          if(   !(   engine->value() == "static"
                  || engine->value() == "nuts")
             || !(   metric->value() == "unit_e"
                  || metric->value() == "diag_e"
                  || metric->value() == "dense_e")) {
               err("No sampler matching HMC specification!");
               return stan::services::error_codes::USAGE;
             }
          
        }

        return stan::services::error_codes::USAGE;
        
      }

    } // sample
  } // services
} // stan

#endif
