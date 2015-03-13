#ifndef STAN__SERVICES__COMMAND_HPP
#define STAN__SERVICES__COMMAND_HPP

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/json.hpp>

#include <stan/services/arguments.hpp>

#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/model/util.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/bfgs.hpp>

#include <stan/services/diagnose.hpp>
#include <stan/services/init.hpp>
#include <stan/services/io.hpp>
#include <stan/services/mcmc.hpp>
#include <stan/services/optimization.hpp>

// FIXME: These belong to the interfaces and should be templated out here
#include <stan/interface_callbacks/interrupt/noop.hpp>
#include <stan/interface_callbacks/var_context_factory/dump_factory.hpp>
#include <stan/interface_callbacks/writer.hpp>

#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
  namespace services {

    template <class Model>
    int command(int argc, const char* argv[]) {
      
      // BEGIN TEMP CALLBACKS
      // FIXME: The below should all be callbacks created externally
      
      stan::interface_callbacks::interrupt::noop iteration_interrupt;
      
      stan::interface_callbacks::writer::cout info; // Informative messages
      stan::interface_callbacks::writer::cerr err;  // Error messages

      // END TEMP CALLBACKS
      
      std::vector<argument*> valid_arguments;
      valid_arguments.push_back(new arg_id());
      valid_arguments.push_back(new stan::services::arg_data());
      valid_arguments.push_back(new stan::services::arg_init());
      valid_arguments.push_back(new stan::services::arg_random());
      valid_arguments.push_back(new stan::services::arg_output());
      
      argument_parser parser(valid_arguments);

      int err_code = parser.parse_args(argc, argv, info, err);

      if (err_code != 0) {
        info.write_message("Failed to parse arguments, terminating Stan");
        return err_code;
      }

      if (parser.help_printed())
        return err_code;

      // BEGIN TEMP CALLBACKS
      // FIXME: The below should all be callbacks created externally
      
      // Sample output
      std::string output_file =
        dynamic_cast<string_argument*>(parser.arg("output")->arg("file"))->value();
      stan::interface_callbacks::writer::fstream_csv output_stream(output_file);
      
      // Diagnostic output
      std::string diagnostic_file =
        dynamic_cast<string_argument*>(parser.arg("output")->arg("diagnostic_file"))->value();
      stan::interface_callbacks::writer::fstream_csv diagnostic_stream(diagnostic_file);
      
      // END TEMP CALLBACKS
      
      // Identification
      unsigned int id = dynamic_cast<stan::services::int_argument*>
        (parser.arg("id"))->value();

      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////

      unsigned int random_seed = 0;
      
      stan::services::u_int_argument* random_arg 
        = dynamic_cast<stan::services::u_int_argument*>
        (parser.arg("random")->arg("seed"));

      if (random_arg->is_default()) {
        random_seed
          = (boost::posix_time::microsec_clock::universal_time() -
             boost::posix_time::ptime(boost::posix_time::min_date_time))
          .total_milliseconds();

        random_arg->set_value(random_seed);

      } else {
        random_seed = random_arg->value();
      }

      typedef boost::ecuyer1988 rng_t;  // (2**50 = 1T samples, 1000 chains)
      rng_t base_rng(random_seed);

      // Advance generator to avoid process conflicts
      static boost::uintmax_t DISCARD_STRIDE
        = static_cast<boost::uintmax_t>(1) << 50;
      base_rng.discard(DISCARD_STRIDE * (id - 1));

      //////////////////////////////////////////////////
      //                  Input/Output                //
      //////////////////////////////////////////////////

      // Data input
      std::string data_file
        = dynamic_cast<stan::services::string_argument*>
        (parser.arg("data")->arg("file"))->value();
      
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      // Refresh rate
      int refresh = dynamic_cast<stan::services::int_argument*>(
                    parser.arg("output")->arg("refresh"))->value();

      //////////////////////////////////////////////////
      //                Initialize Model              //
      //////////////////////////////////////////////////

      Model model(data_var_context, &std::cout); /////***** FIXME NOW *****//////

      Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model.num_params_r());

      parser.print(info); /////***** FIXME NOW *****//////
      info.write_message("");

      services::io::write_stan(output_stream, "#");
      services::io::write_model(output_stream, model.model_name(), "#");
      parser.print(output_stream, "#");

      services::io::write_stan(diagnostic_stream, "#");
      services::io::write_model(diagnostic_stream, model.model_name(), "#");
      parser.print(diagnostic_stream, "#");

      std::string init = dynamic_cast<stan::services::string_argument*>(
                         parser.arg("init"))->value();
      
      interface_callbacks::var_context_factory::dump_factory var_context_factory;
      if (!init::initialize_state<interface_callbacks::var_context_factory::dump_factory>
          (init, cont_params, model, base_rng, info, var_context_factory))
        return stan::services::error_codes::SOFTWARE;
      
      //////////////////////////////////////////////////
      //               Model Diagnostics              //
      //////////////////////////////////////////////////

      if (parser.arg("method")->arg("diagnose")) {
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;
        
        stan::services::list_argument* test = dynamic_cast<stan::services::list_argument*>
                              (parser.arg("method")->arg("diagnose")->arg("test"));
        
        if (test->value() == "gradient") {

          info.write_message("");
          info.write_message("TEST GRADIENT MODE");

          double epsilon = dynamic_cast<stan::services::real_argument*>
                           (test->arg("gradient")->arg("epsilon"))->value();
          
          double error = dynamic_cast<stan::services::real_argument*>
                         (test->arg("gradient")->arg("error"))->value();
          
          int num_failed =
            stan::model::test_gradients<true, true>(model, cont_vector,
                                                    disc_vector, info,
                                                    epsilon, error);

          num_failed
            = stan::model::test_gradients<true, true>(model, cont_vector,
                                                      disc_vector, output_stream,
                                                      epsilon, error);
          
          (void) num_failed; // FIXME: do something with the number failed
          
          return stan::services::error_codes::OK;

        }
      }

      //////////////////////////////////////////////////
      //           Optimization Algorithms            //
      //////////////////////////////////////////////////

      if (parser.arg("method")->arg("optimize")) {
        
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;
        
        stan::services::list_argument* algo = dynamic_cast<stan::services::list_argument*>
                              (parser.arg("method")->arg("optimize")->arg("algorithm"));

        int num_iterations = dynamic_cast<stan::services::int_argument*>(
                             parser.arg("method")->arg("optimize")->arg("iter"))->value();

        bool save_iterations 
          = dynamic_cast<stan::services::bool_argument*>(parser.arg("method")
                                         ->arg("optimize")
                                         ->arg("save_iterations"))->value();

        std::vector<std::string> names;
        names.push_back("lp__");
        model.constrained_param_names(names, true, true);
        output_stream.write_state_names(names);

        double lp(0);
        int return_code = stan::services::error_codes::CONFIG;
        if (algo->value() == "newton") {
          std::vector<double> gradient;
          try {
            lp = model.template log_prob<false, false>
              (cont_vector, disc_vector, &std::cout); /////***** FIXME NOW *****//////
          } catch (const std::exception& e) {
            services::io::write_error_msg(err, e);
            lp = -std::numeric_limits<double>::infinity();
          }

          info.write_message("initial log joint probability = " + stan::interface_callbacks::writer::writer::to_string(lp));
          if (save_iterations) {
            services::io::write_iteration(output_stream, model, base_rng,
                                          lp, cont_vector, disc_vector);
          }

          double lastlp = lp * 1.1;
          int m = 0;
          info.write_message("(lp - lastlp) / lp > 1e-8: "
                             + stan::interface_callbacks::writer::writer::to_string((lp - lastlp) / fabs(lp)));
          while ((lp - lastlp) / fabs(lp) > 1e-8) {
            lastlp = lp;
            lp = stan::optimization::newton_step
              (model, cont_vector, disc_vector);
            info.write_message("Iteration " + stan::interface_callbacks::writer::writer::to_string(m + 1, 2) + ". "
                               + "Log joint probability = " + stan::interface_callbacks::writer::writer::to_string(lp, 10)
                               + ". Improved by " + stan::interface_callbacks::writer::writer::to_string(lp - lastlp) + ".");
            m++;

            if (save_iterations) {
              io::write_iteration(output_stream, model, base_rng,
                                  lp, cont_vector, disc_vector);
            }
          }
          return_code = stan::services::error_codes::OK;
        } else if (algo->value() == "bfgs") {
          
          typedef stan::optimization::BFGSLineSearch
            <Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer;
          Optimizer bfgs(model, cont_vector, disc_vector, &std::cout); /////***** FIXME NOW *****//////

          bfgs._ls_opts.alpha0 = dynamic_cast<stan::services::real_argument*>(
                         algo->arg("bfgs")->arg("init_alpha"))->value();
          bfgs._conv_opts.tolAbsF = dynamic_cast<stan::services::real_argument*>(
                         algo->arg("bfgs")->arg("tol_obj"))->value();
          bfgs._conv_opts.tolRelF = dynamic_cast<stan::services::real_argument*>(
                         algo->arg("bfgs")->arg("tol_rel_obj"))->value();
          bfgs._conv_opts.tolAbsGrad = dynamic_cast<stan::services::real_argument*>(
                         algo->arg("bfgs")->arg("tol_grad"))->value();
          bfgs._conv_opts.tolRelGrad = dynamic_cast<stan::services::real_argument*>(
                         algo->arg("bfgs")->arg("tol_rel_grad"))->value();
          bfgs._conv_opts.tolAbsX = dynamic_cast<stan::services::real_argument*>(
                         algo->arg("bfgs")->arg("tol_param"))->value();
          bfgs._conv_opts.maxIts = num_iterations;
          
          // FIXME: first cout should be output_stream
          // FIXME: second cout should be info
          return_code = optimization::do_bfgs_optimize(model,bfgs, base_rng,
                                                       lp, cont_vector, disc_vector,
                                                       &std::cout, &std::cout,
                                                       save_iterations, refresh,
                                                       iteration_interrupt);
        } else if (algo->value() == "lbfgs") {
          
          typedef stan::optimization::BFGSLineSearch
            <Model,stan::optimization::LBFGSUpdate<> > Optimizer;
          
          Optimizer bfgs(model, cont_vector, disc_vector, &std::cout);

          bfgs.get_qnupdate().set_history_size(dynamic_cast<services::int_argument*>(
                         algo->arg("lbfgs")->arg("history_size"))->value());
          bfgs._ls_opts.alpha0 = dynamic_cast<services::real_argument*>(
                         algo->arg("lbfgs")->arg("init_alpha"))->value();
          bfgs._conv_opts.tolAbsF = dynamic_cast<services::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_obj"))->value();
          bfgs._conv_opts.tolRelF = dynamic_cast<services::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_rel_obj"))->value();
          bfgs._conv_opts.tolAbsGrad = dynamic_cast<services::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_grad"))->value();
          bfgs._conv_opts.tolRelGrad = dynamic_cast<services::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_rel_grad"))->value();
          bfgs._conv_opts.tolAbsX = dynamic_cast<services::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_param"))->value();
          bfgs._conv_opts.maxIts = num_iterations;

          // FIXME: first cout should be output_stream
          // FIXME: second cout should be info
          return_code = optimization::do_bfgs_optimize(model,bfgs, base_rng,
                                                       lp, cont_vector, disc_vector,
                                                       &std::cout, &std::cout,
                                                       save_iterations, refresh,
                                                       iteration_interrupt);
        } else {
          return_code = stan::services::error_codes::CONFIG;
        }

        services::io::write_iteration(output_stream, model, base_rng,
                                      lp, cont_vector, disc_vector);

        return return_code;
      }

      //////////////////////////////////////////////////
      //              Sampling Algorithms             //
      //////////////////////////////////////////////////

      if (parser.arg("method")->arg("sample")) {
        // Check timing
        clock_t start_check = clock();

        double init_log_prob;
        Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model.num_params_r());

        stan::model::gradient(model, cont_params, init_log_prob,
                              init_grad, &std::cout);

        clock_t end_check = clock();
        double deltaT
          = static_cast<double>(end_check - start_check) / CLOCKS_PER_SEC;

        std::cout << std::endl;
        std::cout << "Gradient evaluation took " << deltaT
                  << " seconds" << std::endl;
        std::cout << "1000 transitions using 10 leapfrog steps "
                  << "per transition would take "
                  << 1e4 * deltaT << " seconds." << std::endl;
        std::cout << "Adjust your expectations accordingly!"
                  << std::endl << std::endl;
        std::cout << std::endl;
        
        // Sampling parameters
        int num_warmup = dynamic_cast<stan::services::int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_warmup"))->value();
        
        int num_samples = dynamic_cast<stan::services::int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_samples"))->value();
        
        int num_thin = dynamic_cast<stan::services::int_argument*>(
                       parser.arg("method")->arg("sample")->arg("thin"))->value();
        
        bool save_warmup = dynamic_cast<stan::services::bool_argument*>(
                           parser.arg("method")->arg("sample")->arg("save_warmup"))->value();

        // Sampler
        stan::mcmc::base_mcmc* sampler_ptr = 0;

        stan::services::list_argument* algo
          = dynamic_cast<stan::services::list_argument*>
            (parser.arg("method")->arg("sample")->arg("algorithm"));
        
        stan::services::categorical_argument* adapt
          = dynamic_cast<stan::services::categorical_argument*>
            (parser.arg("method")->arg("sample")->arg("adapt"));
        bool adapt_engaged
          = dynamic_cast<stan::services::bool_argument*>(adapt->arg("engaged"))
            ->value();

        if (model.num_params_r() == 0 && algo->value() != "fixed_param") {
          std::cout
            << "Must use algorithm=fixed_param for "
            << "model that has no parameters."
            << std::endl;
          return -1;
        }

        if (algo->value() == "fixed_param") {
          sampler_ptr = new stan::mcmc::fixed_param_sampler();

          adapt_engaged = false;

          if (num_warmup != 0) {
            std::cout << "Warning: warmup will be skipped "
                      << "for the fixed parameter sampler!"
                      << std::endl;
            num_warmup = 0;
          }

        } else if (algo->value() == "rwm") {
          std::cout << algo->arg("rwm")->description() << std::endl;
          return 0;

        } else if (algo->value() == "hmc") {
          int engine_index = 0;
          
          stan::services::list_argument* engine 
            = dynamic_cast<stan::services::list_argument*>
              (algo->arg("hmc")->arg("engine"));

          if (engine->value() == "static") {
            engine_index = 0;
          } else if (engine->value() == "nuts") {
            engine_index = 1;
          }

          int metric_index = 0;
          stan::services::list_argument* metric 
            = dynamic_cast<stan::services::list_argument*>
              (algo->arg("hmc")->arg("metric"));
          if (metric->value() == "unit_e") {
            metric_index = 0;
          } else if (metric->value() == "diag_e") {
            metric_index = 1;
          } else if (metric->value() == "dense_e") {
            metric_index = 2;
          }

          int sampler_select = engine_index
            + 10 * metric_index
            + 100 * static_cast<int>(adapt_engaged);

          switch (sampler_select) {
            case 0: {
              typedef stan::mcmc::unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_static_hmc<sampler>(sampler_ptr, algo))
                return 0;
              break;
            }

            case 1: {
              typedef stan::mcmc::unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_nuts<sampler>(sampler_ptr, algo))
                return 0;
              break;
            }

            case 10: {
              typedef stan::mcmc::diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_static_hmc<sampler>(sampler_ptr, algo))
                return 0;
              break;
            }

            case 11: {
              typedef stan::mcmc::diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_nuts<sampler>(sampler_ptr, algo))
                return 0;
              break;
            }

            case 20: {
              typedef stan::mcmc::dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_static_hmc<sampler>(sampler_ptr, algo))
                return 0;
              break;
            }

            case 21: {
              typedef stan::mcmc::dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_nuts<sampler>(sampler_ptr, algo))
                return 0;
              break;
            }

            case 100: {
              typedef stan::mcmc::adapt_unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_static_hmc<sampler>(sampler_ptr, algo))
                return 0;
              if (!init::init_adapt<sampler>(sampler_ptr, adapt, cont_params))
                return 0;
              break;
            }

            case 101: {
              typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_nuts<sampler>(sampler_ptr, algo))
                return 0;
              if (!init::init_adapt<sampler>(sampler_ptr, adapt, cont_params))
                return 0;
              break;
            }

            case 110: {
              typedef stan::mcmc::adapt_diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_static_hmc<sampler>(sampler_ptr, algo))
                return 0;
              if (!init::init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params))
                return 0;
              break;
            }

            case 111: {
              typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_nuts<sampler>(sampler_ptr, algo))
                return 0;
              if (!init::init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params))
                return 0;
              break;
            }

            case 120: {
              typedef stan::mcmc::adapt_dense_e_static_hmc<Model, rng_t>
                sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_static_hmc<sampler>(sampler_ptr, algo))
                return 0;
              if (!init::init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params))
                return 0;
              break;
            }

            case 121: {
              typedef stan::mcmc::adapt_dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng,
                                        &std::cout, &std::cout);
              if (!init::init_nuts<sampler>(sampler_ptr, algo))
                return 0;
              if (!init::init_windowed_adapt<sampler>
                  (sampler_ptr, adapt, num_warmup, cont_params))
                return 0;
              break;
            }

            default:
              std::cout << "No sampler matching HMC specification!"
                        << std::endl;
              return 0;
          }
        }

        
        mcmc::mcmc_writer<Model, rng_t,
                          stan::interface_callbacks::writer::fstream_csv,
                          stan::interface_callbacks::writer::fstream_csv,
                          stan::interface_callbacks::writer::cout>
          writer(model, base_rng, output_stream, diagnostic_stream, info);
        
        stan::mcmc::sample s(cont_params, 0, 0);
        
        mcmc::sample(*sampler_ptr, s, num_warmup, num_samples,
                     num_thin, refresh, save_warmup, adapt_engaged,
                     writer, iteration_interrupt);

        if (sampler_ptr)
          delete sampler_ptr;
      }

      for (size_t i = 0; i < valid_arguments.size(); ++i)
        delete valid_arguments.at(i);

      return 0;
    }

  }  // namespace services
}  // namespace stan

#endif
