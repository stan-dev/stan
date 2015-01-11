#ifndef STAN__COMMON__COMMAND_HPP
#define STAN__COMMON__COMMAND_HPP

#include <fstream>
#include <stdexcept>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/json.hpp>
#include <stan/io/mcmc_writer.hpp>

#include <stan/gm/arguments/argument_parser.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>

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

#include <stan/common/write_iteration_csv.hpp>
#include <stan/common/write_iteration.hpp>
#include <stan/common/write_stan.hpp>
#include <stan/common/write_model.hpp>
#include <stan/common/write_error_msg.hpp>
#include <stan/common/do_print.hpp>
#include <stan/common/do_bfgs_optimize.hpp>
#include <stan/common/print_progress.hpp>
#include <stan/common/run_markov_chain.hpp>
#include <stan/common/warmup.hpp>
#include <stan/common/sample.hpp>
#include <stan/common/init_static_hmc.hpp>
#include <stan/common/init_nuts.hpp>
#include <stan/common/init_adapt.hpp>
#include <stan/common/init_windowed_adapt.hpp>
#include <stan/common/recorder/csv.hpp>
#include <stan/common/recorder/messages.hpp>
#include <stan/common/initialize_state.hpp>
#include <stan/common/context_factory.hpp>

namespace stan {

  namespace common {

    struct NoOpFunctor {
      void operator()() { }
    };

    template <class Model>
    int command(int argc, const char* argv[]) {

      std::vector<stan::gm::argument*> valid_arguments;
      valid_arguments.push_back(new stan::gm::arg_id());
      valid_arguments.push_back(new stan::gm::arg_data());
      valid_arguments.push_back(new stan::gm::arg_init());
      valid_arguments.push_back(new stan::gm::arg_random());
      valid_arguments.push_back(new stan::gm::arg_output());
      
      stan::gm::argument_parser parser(valid_arguments);
      int err_code = parser.parse_args(argc, argv, &std::cout, &std::cout);

      if (err_code != 0) {
        std::cout << "Failed to parse arguments, terminating Stan" << std::endl;
        return err_code;
      }
      
      if (parser.help_printed())
        return err_code;
      
      // Identification
      unsigned int id = dynamic_cast<stan::gm::int_argument*>(parser.arg("id"))->value();
      
      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////
      
      unsigned int random_seed = 0;
      stan::gm::u_int_argument* random_arg 
        = dynamic_cast<stan::gm::u_int_argument*>(parser.arg("random")->arg("seed"));
      
      if (random_arg->is_default()) {
        random_seed = (boost::posix_time::microsec_clock::universal_time() -
                       boost::posix_time::ptime(boost::posix_time::min_date_time))
                      .total_milliseconds();
        
        random_arg->set_value(random_seed);
        
      } else {
        random_seed = random_arg->value();
      }
      
      typedef boost::ecuyer1988 rng_t; // (2**50 = 1T samples, 1000 chains)
      rng_t base_rng(random_seed);
      
      // Advance generator to avoid process conflicts
      static boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50;
      base_rng.discard(DISCARD_STRIDE * (id - 1));
      
      //////////////////////////////////////////////////
      //                  Input/Output                //
      //////////////////////////////////////////////////
      
      // Data input
      std::string data_file 
        = dynamic_cast<stan::gm::string_argument*>(parser.arg("data")->arg("file"))->value();
      
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();
      
      // Sample output
      std::string output_file = dynamic_cast<stan::gm::string_argument*>(
                                parser.arg("output")->arg("file"))->value();
      std::fstream* output_stream = 0;
      if (output_file != "") {
        output_stream = new std::fstream(output_file.c_str(),
                                         std::fstream::out);
      }
      
      // Diagnostic output
      std::string diagnostic_file = dynamic_cast<stan::gm::string_argument*>(
                                    parser.arg("output")->arg("diagnostic_file"))->value();
      
      std::fstream* diagnostic_stream = 0;
      if (diagnostic_file != "") {
        diagnostic_stream = new std::fstream(diagnostic_file.c_str(),
                                             std::fstream::out);
      }
      
      // Refresh rate
      int refresh = dynamic_cast<stan::gm::int_argument*>(
                    parser.arg("output")->arg("refresh"))->value();
      
      //////////////////////////////////////////////////
      //                Initialize Model              //
      //////////////////////////////////////////////////
      
      Model model(data_var_context, &std::cout);

      Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model.num_params_r());

      parser.print(&std::cout);
      std::cout << std::endl;
      
      if (output_stream) {
        write_stan(output_stream, "#");
        write_model(output_stream, model.model_name(), "#");
        parser.print(output_stream, "#");
      }
      
      if (diagnostic_stream) {
        write_stan(diagnostic_stream, "#");
        write_model(diagnostic_stream, model.model_name(), "#");
        parser.print(diagnostic_stream, "#");
      }
      
      std::string init = dynamic_cast<stan::gm::string_argument*>(
                         parser.arg("init"))->value();
      
      dump_factory var_context_factory;
      if (!initialize_state<dump_factory>
          (init, cont_params, model, base_rng, &std::cout,
           var_context_factory))
        return stan::gm::error_codes::SOFTWARE;
      
      //////////////////////////////////////////////////
      //               Model Diagnostics              //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("diagnose")) {
      
        std::vector<double> cont_vector(cont_params.size());
        for (int i = 0; i < cont_params.size(); ++i)
          cont_vector.at(i) = cont_params(i);
        std::vector<int> disc_vector;
        
        stan::gm::list_argument* test = dynamic_cast<stan::gm::list_argument*>
                              (parser.arg("method")->arg("diagnose")->arg("test"));
        
        if (test->value() == "gradient") {
          std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;

          double epsilon = dynamic_cast<stan::gm::real_argument*>
                           (test->arg("gradient")->arg("epsilon"))->value();
          
          double error = dynamic_cast<stan::gm::real_argument*>
                         (test->arg("gradient")->arg("error"))->value();
          
          int num_failed
            = stan::model::test_gradients<true,true>(model,cont_vector, disc_vector, 
                                                     epsilon, error, std::cout);
          
          if (output_stream) {
            num_failed
              = stan::model::test_gradients<true,true>(model,cont_vector, disc_vector,
                                                       epsilon, error, *output_stream);
          }
          
          if (diagnostic_stream) {
            num_failed
              = stan::model::test_gradients<true,true>(model,cont_vector, disc_vector, 
                                                       epsilon, error, *diagnostic_stream);
          }
          
          (void) num_failed; // FIXME: do something with the number failed
          
          return stan::gm::error_codes::OK;
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
        
        stan::gm::list_argument* algo = dynamic_cast<stan::gm::list_argument*>
                              (parser.arg("method")->arg("optimize")->arg("algorithm"));

        int num_iterations = dynamic_cast<stan::gm::int_argument*>(
                             parser.arg("method")->arg("optimize")->arg("iter"))->value();

        bool save_iterations 
          = dynamic_cast<stan::gm::bool_argument*>(parser.arg("method")
                                         ->arg("optimize")
                                         ->arg("save_iterations"))->value();
        if (output_stream) {
          std::vector<std::string> names;
          names.push_back("lp__");
          model.constrained_param_names(names,true,true);

          (*output_stream) << names.at(0);
          for (size_t i = 1; i < names.size(); ++i) {
            (*output_stream) << "," << names.at(i);
          }
          (*output_stream) << std::endl;
        }

        double lp(0);
        int return_code = stan::gm::error_codes::CONFIG;
        if (algo->value() == "newton") {
          std::vector<double> gradient;
          try {
            lp = model.template log_prob<false, false>(cont_vector, disc_vector, &std::cout);
          } catch (const std::exception& e) {
            write_error_msg(&std::cout, e);
            lp = -std::numeric_limits<double>::infinity();
          }
          
          std::cout << "initial log joint probability = " << lp << std::endl;
          if (output_stream && save_iterations) {
            write_iteration(*output_stream, model, base_rng,
                            lp, cont_vector, disc_vector);
          }

          double lastlp = lp * 1.1;
          int m = 0;
          std::cout << "(lp - lastlp) / lp > 1e-8: " << ((lp - lastlp) / fabs(lp)) << std::endl;
          while ((lp - lastlp) / fabs(lp) > 1e-8) {
            
            lastlp = lp;
            lp = stan::optimization::newton_step(model, cont_vector, disc_vector);
            std::cout << "Iteration ";
            std::cout << std::setw(2) << (m + 1) << ". ";
            std::cout << "Log joint probability = " << std::setw(10) << lp;
            std::cout << ". Improved by " << (lp - lastlp) << ".";
            std::cout << std::endl;
            std::cout.flush();
            m++;

            if (output_stream && save_iterations) {
              write_iteration(*output_stream, model, base_rng,
                              lp, cont_vector, disc_vector);
            }
            
          }
          return_code = stan::gm::error_codes::OK;
        } else if (algo->value() == "bfgs") {
          NoOpFunctor callback;
          typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::BFGSUpdate_HInv<> > Optimizer;
          Optimizer bfgs(model, cont_vector, disc_vector, &std::cout);

          bfgs._ls_opts.alpha0 = dynamic_cast<stan::gm::real_argument*>(
                         algo->arg("bfgs")->arg("init_alpha"))->value();
          bfgs._conv_opts.tolAbsF = dynamic_cast<stan::gm::real_argument*>(
                         algo->arg("bfgs")->arg("tol_obj"))->value();
          bfgs._conv_opts.tolRelF = dynamic_cast<stan::gm::real_argument*>(
                         algo->arg("bfgs")->arg("tol_rel_obj"))->value();
          bfgs._conv_opts.tolAbsGrad = dynamic_cast<stan::gm::real_argument*>(
                         algo->arg("bfgs")->arg("tol_grad"))->value();
          bfgs._conv_opts.tolRelGrad = dynamic_cast<stan::gm::real_argument*>(
                         algo->arg("bfgs")->arg("tol_rel_grad"))->value();
          bfgs._conv_opts.tolAbsX = dynamic_cast<stan::gm::real_argument*>(
                         algo->arg("bfgs")->arg("tol_param"))->value();
          bfgs._conv_opts.maxIts = num_iterations;
          
          return_code = do_bfgs_optimize(model,bfgs, base_rng,
                                         lp, cont_vector, disc_vector,
                                         output_stream, &std::cout, 
                                         save_iterations, refresh,
                                         callback);
        } else if (algo->value() == "lbfgs") {
          NoOpFunctor callback;
          typedef stan::optimization::BFGSLineSearch<Model,stan::optimization::LBFGSUpdate<> > Optimizer;
          Optimizer bfgs(model, cont_vector, disc_vector, &std::cout);

          bfgs.get_qnupdate().set_history_size(dynamic_cast<gm::int_argument*>(
                         algo->arg("lbfgs")->arg("history_size"))->value());
          bfgs._ls_opts.alpha0 = dynamic_cast<gm::real_argument*>(
                         algo->arg("lbfgs")->arg("init_alpha"))->value();
          bfgs._conv_opts.tolAbsF = dynamic_cast<gm::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_obj"))->value();
          bfgs._conv_opts.tolRelF = dynamic_cast<gm::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_rel_obj"))->value();
          bfgs._conv_opts.tolAbsGrad = dynamic_cast<gm::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_grad"))->value();
          bfgs._conv_opts.tolRelGrad = dynamic_cast<gm::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_rel_grad"))->value();
          bfgs._conv_opts.tolAbsX = dynamic_cast<gm::real_argument*>(
                         algo->arg("lbfgs")->arg("tol_param"))->value();
          bfgs._conv_opts.maxIts = num_iterations;

          return_code = do_bfgs_optimize(model,bfgs, base_rng,
                                         lp, cont_vector, disc_vector,
                                         output_stream, &std::cout, 
                                         save_iterations, refresh,
                                         callback);
        } else {
          return_code = stan::gm::error_codes::CONFIG;
        }

        if (output_stream) {
          write_iteration(*output_stream, model, base_rng,
                          lp, cont_vector, disc_vector);
          output_stream->close();
          delete output_stream;
        }
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
        
        stan::model::gradient(model, cont_params, init_log_prob, init_grad, &std::cout);
        
        clock_t end_check = clock();
        double deltaT = (double)(end_check - start_check) / CLOCKS_PER_SEC;
        
        std::cout << std::endl;
        std::cout << "Gradient evaluation took " << deltaT << " seconds" << std::endl;
        std::cout << "1000 transitions using 10 leapfrog steps per transition would take "
                  << 1e4 * deltaT << " seconds." << std::endl;
        std::cout << "Adjust your expectations accordingly!" << std::endl << std::endl;
        std::cout << std::endl;
        
        stan::common::recorder::csv sample_recorder(output_stream, "# ");
        stan::common::recorder::csv diagnostic_recorder(diagnostic_stream, "# ");
        stan::common::recorder::messages message_recorder(&std::cout, "# ");
        
        stan::io::mcmc_writer<Model, 
                              stan::common::recorder::csv, stan::common::recorder::csv,
                              stan::common::recorder::messages> 
          writer(sample_recorder, diagnostic_recorder, message_recorder, &std::cout);
        
        // Sampling parameters
        int num_warmup = dynamic_cast<stan::gm::int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_warmup"))->value();
        
        int num_samples = dynamic_cast<stan::gm::int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_samples"))->value();
        
        int num_thin = dynamic_cast<stan::gm::int_argument*>(
                       parser.arg("method")->arg("sample")->arg("thin"))->value();
        
        bool save_warmup = dynamic_cast<stan::gm::bool_argument*>(
                           parser.arg("method")->arg("sample")->arg("save_warmup"))->value();
        
        stan::mcmc::sample s(cont_params, 0, 0);
        
        double warmDeltaT;
        double sampleDeltaT;
        
        // Sampler
        stan::mcmc::base_mcmc* sampler_ptr = 0;
        
        stan::gm::list_argument* algo = dynamic_cast<stan::gm::list_argument*>
                              (parser.arg("method")->arg("sample")->arg("algorithm"));
        
        stan::gm::categorical_argument* adapt = dynamic_cast<stan::gm::categorical_argument*>(
                                      parser.arg("method")->arg("sample")->arg("adapt"));
        bool adapt_engaged = dynamic_cast<stan::gm::bool_argument*>(adapt->arg("engaged"))->value();
        
        if (model.num_params_r() == 0 && algo->value() != "fixed_param") {
          std::cout
            << "Must use algorithm=fixed_param for model that has no parameters."
            << std::endl;
          return -1;
        }      

        if (algo->value() == "fixed_param") {
          
          sampler_ptr = new stan::mcmc::fixed_param_sampler();
          
          adapt_engaged = false;
          
          if (num_warmup != 0) {
            std::cout << "Warning: warmup will be skipped for the fixed parameter sampler!" << std::endl;
            num_warmup = 0;
          }
          
        } else if (algo->value() == "rwm") {
          
          std::cout << algo->arg("rwm")->description() << std::endl;
          return 0;
        
        } else if (algo->value() == "hmc") {
          
          int engine_index = 0;
          stan::gm::list_argument* engine 
            = dynamic_cast<stan::gm::list_argument*>(algo->arg("hmc")->arg("engine"));
          if (engine->value() == "static") {
            engine_index = 0;
          } else if (engine->value() == "nuts") {
            engine_index = 1;
          }
          
          int metric_index = 0;
          stan::gm::list_argument* metric 
            = dynamic_cast<stan::gm::list_argument*>(algo->arg("hmc")->arg("metric"));
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
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
              
            case 1: {        
              typedef stan::mcmc::unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
              
            case 10: {
              typedef stan::mcmc::diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 11: {
              typedef stan::mcmc::diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 20: {
              typedef stan::mcmc::dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 21: {
              typedef stan::mcmc::dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 100: {
              typedef stan::mcmc::adapt_unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt, cont_params)) return 0;
              break;
            }
            
            case 101: {
              typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt, cont_params)) return 0;
              break;
            }
            
            case 110: {
              typedef stan::mcmc::adapt_diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params)) 
                return 0;
              break;
            }
            
            case 111: {
              typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params)) 
                return 0;
              break;
            }
            
            case 120: {
              typedef stan::mcmc::adapt_dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params)) 
                return 0;
              break;
            }
            
            case 121: {
              typedef stan::mcmc::adapt_dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, &std::cout, &std::cout);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_windowed_adapt<sampler>(sampler_ptr, adapt, num_warmup, cont_params))
                return 0;
              break;
            }
            
            default:
              std::cout << "No sampler matching HMC specification!" << std::endl;
              return 0;
          }
          
        }
        
        // Headers
        writer.write_sample_names(s, sampler_ptr, model);
        writer.write_diagnostic_names(s, sampler_ptr, model);
        
        std::string prefix = "";
        std::string suffix = "\n";
        NoOpFunctor startTransitionCallback;

        // Warm-Up
        clock_t start = clock();
        
        warmup<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                             refresh, save_warmup,
                             writer,
                             s, model, base_rng,
                             prefix, suffix, std::cout,
                             startTransitionCallback);
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        if (adapt_engaged) {
          dynamic_cast<mcmc::base_adapter*>(sampler_ptr)->disengage_adaptation();
          writer.write_adapt_finish(sampler_ptr);
        }
        
        // Sampling
        start = clock();
        
        stan::common::sample<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                             refresh, true,
                             writer,
                             s, model, base_rng,
                             prefix, suffix, std::cout,
                             startTransitionCallback);
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        writer.write_timing(warmDeltaT, sampleDeltaT);
        
        if (sampler_ptr) delete sampler_ptr;
        
      }
      
      if (output_stream) {
        output_stream->close();
        delete output_stream;
      }
        
      if (diagnostic_stream) {
        diagnostic_stream->close();
        delete diagnostic_stream;
      }
      
      for (size_t i = 0; i < valid_arguments.size(); ++i)
        delete valid_arguments.at(i);
      
      return 0;
 
    }

  } // namespace common

} // namespace stan

#endif
