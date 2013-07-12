#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <fstream>
#include <stdexcept>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/io/mcmc_writer.hpp>

#include <stan/gm/arguments/argument_parser.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_test_gradient.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_output.hpp>

#include <stan/mcmc/hmc/static/adapt_unit_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_diag_e_static_hmc.hpp>
#include <stan/mcmc/hmc/static/adapt_dense_e_static_hmc.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/nesterov_gradient.hpp>
#include <stan/optimization/bfgs.hpp>

namespace stan {

  namespace gm {

    void write_stan(std::ostream* s, const char prefix = '\0') {
      if(!s) return;
      
      *s << prefix << " stan_version_major = " << stan::MAJOR_VERSION << std::endl;
      *s << prefix << " stan_version_minor = " << stan::MINOR_VERSION << std::endl;
      *s << prefix << " stan_version_patch = " << stan::PATCH_VERSION << std::endl;
      
    }
    
    void write_error_msg(std::ostream* error_stream,
                         const std::domain_error& e) {
      
      if (!error_stream) return;
      
      *error_stream << std::endl
                    << "Informational Message: The parameter state is about to be Metropolis"
                    << " rejected due to the following underlying, non-fatal (really)"
                    << " issue (and please ignore that what comes next might say 'error'): "
                    << e.what()
                    << std::endl
                    << "If the problem persists across multiple draws, you might have"
                    << " a problem with an initial state or a gradient somewhere."
                    << std::endl
                    << " If the problem does not persist, the resulting samples will still"
                    << " be drawn from the posterior."
                    << std::endl;
      
    }
    
    bool do_print(int n, int refresh) {
      return (refresh > 0) &&
      (n == 0 || ((n + 1) % refresh == 0) );
    }

    void print_progress(int m, int start, int finish, int refresh, bool warmup) {
      
      int it_print_width = std::ceil(std::log10(finish));
      
      if (do_print(m, refresh)) {
        
        std::cout << "Iteration: ";
        std::cout << std::setw(it_print_width) << m + 1 + start
                  << " / " << finish;
          
        std::cout << " [" << std::setw(3) 
                  << static_cast<int>( (100.0 * (m + 1 + start)) / finish )
                  << "%] ";
        std::cout << (warmup ? " (Warmup)" : " (Sampling)");
        std::cout << std::endl;
          
      }
    
    }
    
    template <class Model, class RNG>
    void run_markov_chain(stan::mcmc::base_mcmc* sampler,
                          int num_iterations,
                          int start,
                          int finish,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          stan::io::mcmc_writer<Model>& writer,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng) {
      
      for (int m = 0; m < num_iterations; ++m) {
      
        print_progress(m, start, finish, refresh, warmup);
      
        init_s = sampler->transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          writer.print_sample_params(init_s, sampler, model);
          writer.print_diagnostic_params(init_s, sampler);
        }

      }
      
    }

    template <class Model, class RNG>
    void warmup(stan::mcmc::base_mcmc* sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                stan::io::mcmc_writer<Model>& writer,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Model, RNG>(sampler, num_warmup, 0, num_warmup + num_samples, num_thin,
                                   refresh, save, true,
                                   writer,
                                   init_s, model, base_rng);
      
    }

    template <class Model, class RNG>
    void sample(stan::mcmc::base_mcmc* sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                stan::io::mcmc_writer<Model>& writer,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Model, RNG>(sampler, num_samples, num_warmup, num_warmup + num_samples, num_thin,
                                   refresh, save, false,
                                   writer,
                                   init_s, model, base_rng);
      
    }
      
    
    template<class Sampler>
    bool init_static_hmc(stan::mcmc::base_mcmc* sampler, argument* algorithm) {

      categorical_argument* hmc = dynamic_cast<categorical_argument*>(
                                  algorithm->arg("hmc"));
      
      categorical_argument* base = dynamic_cast<categorical_argument*>(
                                   algorithm->arg("hmc")->arg("engine")->arg("static"));
      
      double epsilon = dynamic_cast<real_argument*>(hmc->arg("stepsize"))->value();
      double epsilon_jitter = dynamic_cast<real_argument*>(hmc->arg("stepsize_jitter"))->value();
      double int_time = dynamic_cast<real_argument*>(base->arg("int_time"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize_and_T(epsilon, int_time);
      dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);
      
      try {
        dynamic_cast<Sampler*>(sampler)->init_stepsize();
      } catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
        return false;
      }
      
      return true;
      
    }
  
    template<class Sampler>
    bool init_nuts(stan::mcmc::base_mcmc* sampler, argument* algorithm) {
      
      categorical_argument* hmc = dynamic_cast<categorical_argument*>(
                                   algorithm->arg("hmc"));
      
      categorical_argument* base = dynamic_cast<categorical_argument*>(
                                   algorithm->arg("hmc")->arg("engine")->arg("nuts"));

      double epsilon = dynamic_cast<real_argument*>(hmc->arg("stepsize"))->value();
      double epsilon_jitter = dynamic_cast<real_argument*>(hmc->arg("stepsize_jitter"))->value();
      int max_depth = dynamic_cast<int_argument*>(base->arg("max_depth"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize(epsilon);
      dynamic_cast<Sampler*>(sampler)->set_stepsize_jitter(epsilon_jitter);
      dynamic_cast<Sampler*>(sampler)->set_max_depth(max_depth);
      
      try {
        dynamic_cast<Sampler*>(sampler)->init_stepsize();
      } catch (std::runtime_error e) {
        std::cout << e.what() << std::endl;
        return false;
      }
      
      return true;
      
    }
    
    template<class Sampler>
    bool init_adapt(stan::mcmc::base_mcmc* sampler, categorical_argument* adapt) {

      double delta = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
      double gamma = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
      double kappa = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
      double t0    = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();
      
      double epsilon = dynamic_cast<Sampler*>(sampler)->get_nominal_stepsize();
      
      dynamic_cast<Sampler*>(sampler)->get_stepsize_adaptation().set_mu(log(10 * epsilon));
      dynamic_cast<Sampler*>(sampler)->get_stepsize_adaptation().set_delta(delta);
      dynamic_cast<Sampler*>(sampler)->get_stepsize_adaptation().set_gamma(gamma);
      dynamic_cast<Sampler*>(sampler)->get_stepsize_adaptation().set_kappa(kappa);
      dynamic_cast<Sampler*>(sampler)->get_stepsize_adaptation().set_t0(t0);
      
      dynamic_cast<Sampler*>(sampler)->engage_adaptation();
      
      return true;
      
    }
    
    template <class Model>
    int command(int argc, const char* argv[]) {

      std::vector<argument*> valid_arguments;
      valid_arguments.push_back(new arg_id());
      valid_arguments.push_back(new arg_data());
      valid_arguments.push_back(new arg_init());
      valid_arguments.push_back(new arg_test_gradient());
      valid_arguments.push_back(new arg_random());
      valid_arguments.push_back(new arg_output());
      
      argument_parser parser(valid_arguments, Model::model_name());
      
      if (!parser.parse_args(argc, argv, &std::cout, &std::cout)) {
        std::cout << "Failed to parse arguments, terminating Stan" << std::endl;
        return 0;
      }
      
      if (parser.help_printed())
        return 0;
      
      // Identification
      unsigned int id = dynamic_cast<int_argument*>(parser.arg("id"))->value();
      
      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////
      
      unsigned int random_seed = 0;
      u_int_argument* random_arg = dynamic_cast<u_int_argument*>(parser.arg("random")->arg("seed"));
      
      if (random_arg->is_default()) {
        random_seed = (boost::posix_time::microsec_clock::universal_time() -
                       boost::posix_time::ptime(boost::posix_time::min_date_time))
                      .total_milliseconds();
        
        random_arg->set_value(random_seed);
        
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
      std::string data_file = dynamic_cast<string_argument*>(parser.arg("data"))->value();
      
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();
      
      // Sample output
      std::string sample_file = dynamic_cast<string_argument*>(
                                parser.arg("output")->arg("sample"))->value();

      bool append_sample = dynamic_cast<bool_argument*>(
                           parser.arg("output")->arg("append_sample"))->value();
      
      std::ios_base::openmode samples_append_mode
        = append_sample
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;
      
      std::fstream* sample_stream = 0;
      if(sample_file != "") {
        sample_stream = new std::fstream(sample_file.c_str(),
                                         samples_append_mode);
      }
      
      // Diagnostic output
      std::string diagnostic_file = dynamic_cast<string_argument*>(
                                    parser.arg("output")->arg("diagnostic"))->value();
      
      bool append_diagnostic = dynamic_cast<bool_argument*>(
                               parser.arg("output")->arg("append_diagnostic"))->value();
      
      std::ios_base::openmode diagnostic_append_mode
        = append_diagnostic
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;
      
      std::fstream* diagnostic_stream = 0;
      if(diagnostic_file != "") {
        diagnostic_stream = new std::fstream(diagnostic_file.c_str(),
                                             diagnostic_append_mode);
      }
      
      // Refresh rate
      int refresh = dynamic_cast<int_argument*>(
                    parser.arg("output")->arg("refresh"))->value();
      
      //////////////////////////////////////////////////
      //                Initialize Model              //
      //////////////////////////////////////////////////
      
      Model model(data_var_context, &std::cout);
      
      std::vector<double> cont_params(model.num_params_r());
      std::vector<int> disc_params(model.num_params_i());
      
      int num_init_tries = -1;
      
      std::string init = dynamic_cast<string_argument*>(
                         parser.arg("init"))->value();
      
      try {
        
        double R = boost::lexical_cast<double>(init);
        
        if (R == 0) {
        
          cont_params = std::vector<double>(model.num_params_r(), 0.0);
          disc_params = std::vector<int>(model.num_params_i(), 0);
          
          double init_log_prob;
          std::vector<double> init_grad;
          
          try {
            init_log_prob = model.grad_log_prob(cont_params, disc_params, init_grad, &std::cout);
          } catch (std::domain_error e) {
            std::cout << "Rejecting inititialization at zero because of grad_log_prob failure." << std::endl;
            return 0;
          }
          
          if (!boost::math::isfinite(init_log_prob)) {
            std::cout << "Rejecting inititialization at zero because of vanishing density." << std::endl;
            return 0;
          }
          
          for (size_t i = 0; i < init_grad.size(); ++i) {
            if (!boost::math::isfinite(init_grad[i])) {
              std::cout << "Rejecting inititialization at zero because of divergent gradient." << std::endl;
              return 0;
            }
          }

        } else {
          
          boost::random::uniform_real_distribution<double>
          init_range_distribution(-R, R);
          
          boost::variate_generator<rng_t&,
          boost::random::uniform_real_distribution<double> >
          init_rng(base_rng, init_range_distribution);
          
          cont_params = std::vector<double>(model.num_params_r());
          disc_params = std::vector<int>(model.num_params_i(), 0);
          
          // Random initializations until log_prob is finite
          std::vector<double> init_grad;
          static int MAX_INIT_TRIES = 100;
          
          for (num_init_tries = 1; num_init_tries <= MAX_INIT_TRIES; ++num_init_tries) {
            
            for (size_t i = 0; i < cont_params.size(); ++i)
              cont_params[i] = init_rng();
            
            // FIXME: allow config vs. std::cout
            double init_log_prob;
            try {
              init_log_prob = model.grad_log_prob(cont_params, disc_params, init_grad, &std::cout);
            } catch (std::domain_error e) {
              write_error_msg(&std::cout, e);
              std::cout << "Rejecting proposed initial value with zero density." << std::endl;
              init_log_prob = -std::numeric_limits<double>::infinity();
            }
            
            if (!boost::math::isfinite(init_log_prob))
              continue;
            for (size_t i = 0; i < init_grad.size(); ++i)
              if (!boost::math::isfinite(init_grad[i]))
                continue;
            break;
            
          }
          
          if (num_init_tries > MAX_INIT_TRIES) {
            std::cout << std::endl << std::endl
                      << "Initialization between (" << -R << ", " << R << ") failed after "
                      << MAX_INIT_TRIES << " attempts. " << std::endl;
            std::cout << " Try specifying initial values,"
                      << " reducing ranges of constrained values,"
                      << " or reparameterizing the model."
                      << std::endl;
            return -1;
          }
          
        }
        
      } catch(...) {
      
        try {
        
          std::fstream init_stream(init.c_str(), std::fstream::in);
          if (init_stream.fail()) {
            std::string msg("ERROR: specified initialization file does not exist: ");
            msg += init;
            throw std::invalid_argument(msg);
          }
          
          stan::io::dump init_var_context(init_stream);
          init_stream.close();
          model.transform_inits(init_var_context, disc_params, cont_params);
        
        } catch (const std::exception& e) {
          std::cerr << "Error during user-specified initialization:"
                    << std::endl
                    << e.what()
                    << std::endl;
          return -5;
        }
        
        double init_log_prob;
        std::vector<double> init_grad;
        
        try {
          init_log_prob = model.grad_log_prob(cont_params, disc_params, init_grad, &std::cout);
        } catch (std::domain_error e) {
          std::cout << "Rejecting user-specified inititialization because of grad_log_prob failure." << std::endl;
          return 0;
        }
        
        if (!boost::math::isfinite(init_log_prob)) {
          std::cout << "Rejecting user-specified inititialization because of vanishing density." << std::endl;
          return 0;
        }
        
        for (size_t i = 0; i < init_grad.size(); ++i) {
          if (!boost::math::isfinite(init_grad[i])) {
            std::cout << "Rejecting user-specified inititialization because of divergent gradient." << std::endl;
            return 0;
          }
        }
        
      }
      
      // Test gradient
      if (dynamic_cast<unvalued_argument*>(parser.arg("test_gradient"))->is_present()) {
        std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;
        return model.test_gradients(cont_params, disc_params);
        return 0;
      }
      
      // Initial output
      parser.print(&std::cout);
      std::cout << std::endl;
      
      if (!append_sample && sample_stream) {
        write_stan(sample_stream, '#');
        parser.print(sample_stream, '#');
      }
      
      if (!append_diagnostic && diagnostic_stream) {
        write_stan(diagnostic_stream, '#');
        parser.print(diagnostic_stream, '#');
      }
      
      //////////////////////////////////////////////////
      //           Optimization Algorithms            //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("optimize")) {
        
        list_argument* algo = dynamic_cast<list_argument*>
                              (parser.arg("method")->arg("optimize")->arg("algorithm"));

        int num_iterations = dynamic_cast<int_argument*>(
                             parser.arg("method")->arg("optimize")->arg("iter"))->value();

        bool save_iterations = dynamic_cast<bool_argument*>(
                               parser.arg("method")->arg("optimize")->arg("save_iterations"))->value();

        if (algo->value() == "nesterov") {

          bool epsilon = dynamic_cast<real_argument*>(
                         algo->arg("nesterov")->arg("stepsize"))->value();
          
          if (sample_stream) {
            *sample_stream << "lp__,";
            model.write_csv_header(*sample_stream);
          }
          
          stan::optimization::NesterovGradient ng(model, cont_params, disc_params,
                                                  epsilon, &std::cout);
          
          double lp = ng.logp();
          
          double lastlp = lp - 1;
          std::cout << "initial log joint probability = " << lp << std::endl;
          int m = 0;
          for (size_t i = 0; i < num_iterations; i++) {
            lastlp = lp;
            lp = ng.step();
            ng.params_r(cont_params);
            if (do_print(i, refresh)) {
              std::cout << "Iteration ";
              std::cout << std::setw(2) << (m + 1) << ". ";
              std::cout << "Log joint probability = " << std::setw(10) << lp;
              std::cout << ". Improved by " << (lp - lastlp) << ".";
              std::cout << std::endl;
              std::cout.flush();
            }
            m++;
            if (sample_stream && save_iterations) {
              *sample_stream << lp << ',';
              model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
              sample_stream->flush();
            }

          }
        
          if (sample_stream) {
            *sample_stream << lp << ',';
            model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
          }
          
          return 0;
        
        } else if (algo->value() == "newton") {
          
          if (sample_stream) {
            *sample_stream << "lp__,";
            model.write_csv_header(*sample_stream);
          }
          
          std::vector<double> gradient;
          double lp;
          try {
            lp = model.grad_log_prob(cont_params, disc_params, gradient);
          } catch (std::domain_error e) {
            write_error_msg(&std::cout, e);
            lp = -std::numeric_limits<double>::infinity();
          }
          
          double lastlp = lp - 1;
          std::cout << "initial log joint probability = " << lp << std::endl;
          int m = 0;
          while ((lp - lastlp) / fabs(lp) > 1e-8) {
            
            lastlp = lp;
            lp = stan::optimization::newton_step(model, cont_params, disc_params);
            std::cout << "Iteration ";
            std::cout << std::setw(2) << (m + 1) << ". ";
            std::cout << "Log joint probability = " << std::setw(10) << lp;
            std::cout << ". Improved by " << (lp - lastlp) << ".";
            std::cout << std::endl;
            std::cout.flush();
            m++;

            if (sample_stream && save_iterations) {
              *sample_stream << lp << ',';
              model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
            }
            
          }
          
          if (sample_stream) {
            *sample_stream << lp << ',';
            model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
          }
          
          return 0;
          
        } else if (algo->value() == "bfgs") {
          
          bool epsilon = dynamic_cast<real_argument*>(
                         algo->arg("bfgs")->arg("stepsize"))->value();
          
          if (sample_stream) {
            *sample_stream << "lp__,";
            model.write_csv_header(*sample_stream);
          }
          
          stan::optimization::BFGSLineSearch ng(model, cont_params, disc_params,
                                                &std::cout);
          if (epsilon > 0)
            ng._opts.alpha0 = epsilon;
          
          double lp = ng.logp();
          
          std::cout << "initial log joint probability = " << lp << std::endl;
          int m = 0;
          int ret = 0;
          
          for (size_t i = 0; i < num_iterations && ret == 0; i++) {
            
            ret = ng.step();
            lp = ng.logp();
            ng.params_r(cont_params);
            
            if (do_print(i, 50*refresh)) {
              std::cout << "    Iter ";
              std::cout << "     log prob ";
              std::cout << "       ||dx|| ";
              std::cout << "     ||grad|| ";
              std::cout << "      alpha ";
              std::cout << "     alpha0 ";
              std::cout << " # evals ";
              std::cout << " Notes " << std::endl;
            }
            
            if (do_print(i, refresh) || ret != 0 || !ng.note().empty()) {
              std::cout << " " << std::setw(7) << (m + 1) << " ";
              std::cout << " " << std::setw(12) << std::setprecision(6) << lp << " ";
              std::cout << " " << std::setw(12) << std::setprecision(6) << ng.prev_step_size() << " ";
              std::cout << " " << std::setw(12) << std::setprecision(6) << ng.curr_g().norm() << " ";
              std::cout << " " << std::setw(10) << std::setprecision(4) << ng.alpha() << " ";
              std::cout << " " << std::setw(10) << std::setprecision(4) << ng.alpha0() << " ";
              std::cout << " " << std::setw(7) << ng.grad_evals() << " ";
              std::cout << " " << ng.note() << " ";
              std::cout << std::endl;
            }
            
            m++;
            
            if (sample_stream && save_iterations) {
              *sample_stream << lp << ',';
              model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
              sample_stream->flush();
            }
          
          }
          
          if (ret != 0)
            std::cout << "Optimization terminated with code " << ret << std::endl;
          else
            std::cout << "Maximum number of iterations hit, optimization terminated." << std::endl;
          
          if (sample_stream) {
            *sample_stream << lp << ',';
            model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
          }
          
          return 0;
        
        }

      }
        
      //////////////////////////////////////////////////
      //              Sampling Algorithms             //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("sample")) {
        
        stan::io::mcmc_writer<Model> writer(sample_stream, diagnostic_stream);
        
        // Sampling parameters
        int num_warmup = dynamic_cast<int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_warmup"))->value();
        
        int num_samples = dynamic_cast<int_argument*>(
                          parser.arg("method")->arg("sample")->arg("num_samples"))->value();
        
        int num_thin = dynamic_cast<int_argument*>(
                       parser.arg("method")->arg("sample")->arg("thin"))->value();
        
        bool save_warmup = dynamic_cast<bool_argument*>(
                           parser.arg("method")->arg("sample")->arg("save_warmup"))->value();
        
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        
        double warmDeltaT;
        double sampleDeltaT;
        
        // Sampler
        stan::mcmc::base_mcmc* sampler_ptr = 0;
        
        list_argument* algo = dynamic_cast<list_argument*>
                              (parser.arg("method")->arg("sample")->arg("algorithm"));
        
        categorical_argument* adapt = dynamic_cast<categorical_argument*>(
                                      parser.arg("method")->arg("sample")->arg("adapt"));
        bool adapt_engaged = dynamic_cast<bool_argument*>(adapt->arg("engaged"))->value();
        
        if (algo->value() == "metro") {
          
          std::cout << algo->arg("metro")->description() << std::endl;
          return 0;
        
        } else if (algo->value() == "hmc") {
          
          int engine_index = 0;
          list_argument* engine = dynamic_cast<list_argument*>(algo->arg("hmc")->arg("engine"));
          
          if (engine->value() == "static") {
            engine_index = 0;
          
          } else if (engine->value() == "nuts") {
            engine_index = 1;
          }
          
          int metric_index = 0;
          list_argument* metric = dynamic_cast<list_argument*>(algo->arg("hmc")->arg("metric"));
          
          if (metric->value() == "unit_e") {
            metric_index = 0;
            
          } else if (metric->value() == "diag_e") {
            metric_index = 1;
            
          } else if (metric->value() == "dense_e") {
            metric_index = 2;
          }
          
          int sampler_select = engine_index + 10 * metric_index + 100 * static_cast<int>(adapt_engaged);
          
          switch (sampler_select) {
              
            case 0: {
              typedef stan::mcmc::unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
              
            case 1: {        
              typedef stan::mcmc::unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
              
            case 10: {
              typedef stan::mcmc::diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 11: {
              typedef stan::mcmc::diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 20: {
              typedef stan::mcmc::dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 21: {
              typedef stan::mcmc::dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              break;
            }
            
            case 100: {
              typedef stan::mcmc::adapt_unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt)) return 0;
              break;
            }
            
            case 101: {
              typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt)) return 0;
              break;
            }
            
            case 110: {
              typedef stan::mcmc::adapt_diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, num_warmup);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt)) return 0;
              break;
            }
            
            case 111: {
              typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, num_warmup);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt)) return 0;
              break;
            }
            
            case 120: {
              typedef stan::mcmc::adapt_dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, num_warmup);
              if (!init_static_hmc<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt)) return 0;
              break;
            }
            
            case 121: {
              typedef stan::mcmc::adapt_dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng, num_warmup);
              if (!init_nuts<sampler>(sampler_ptr, algo)) return 0;
              if (!init_adapt<sampler>(sampler_ptr, adapt)) return 0;
              break;
            }
            
            default:
              std::cout << "No sampler matching HMC specification!" << std::endl;
              return 0;
          }
          
        }
        
        // Headers
        if (!append_sample) writer.print_sample_names(s, sampler_ptr, model);
        if (!append_diagnostic) writer.print_diagnostic_names(s, sampler_ptr, model);
        
        // Warm-Up
        clock_t start = clock();
        
        warmup<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                             refresh, save_warmup,
                             writer,
                             s, model, base_rng);
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        if (adapt_engaged) {
          dynamic_cast<mcmc::base_adapter*>(sampler_ptr)->disengage_adaptation();
          writer.print_adapt_finish(sampler_ptr);
        }
        
        // Sampling
        start = clock();
        
        sample<Model, rng_t>(sampler_ptr, num_warmup, num_samples, num_thin,
                             refresh, true,
                             writer,
                             s, model, base_rng);
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        writer.print_timing(warmDeltaT, sampleDeltaT);
        
      }
      
      if(sample_stream) {
        sample_stream->close();
        delete sample_stream;
      }
        
      if(diagnostic_stream) {
        diagnostic_stream->close();
        delete diagnostic_stream;
      }
      
      return 0;
 
    }

  } // namespace gm

} // namespace stan

#endif
