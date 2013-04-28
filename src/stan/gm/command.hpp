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

#include <stan/gm/arguments/argument_parser.hpp>
#include <stan/gm/arguments/arg_help.hpp>
#include <stan/gm/arguments/arg_id.hpp>
#include <stan/gm/arguments/arg_data.hpp>
#include <stan/gm/arguments/arg_init.hpp>
#include <stan/gm/arguments/arg_test_gradient.hpp>
#include <stan/gm/arguments/arg_random.hpp>
#include <stan/gm/arguments/arg_method.hpp>
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
    
    void write_stan(std::ostream* s) {
      if(!s) return;
      
      *s << "stan_version_major = " << stan::MAJOR_VERSION << std::endl;
      *s << "stan_version_minor = " << stan::MINOR_VERSION << std::endl;
      *s << "stan_version_patch = " << stan::PATCH_VERSION << std::endl;
      
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

    void print_progress(int m, int num_iterations, int refresh, bool warmup) {
      
      int it_print_width = std::ceil(std::log10(num_iterations));
      
      if (do_print(m, refresh)) {
        
        std::cout << "Iteration: ";
        std::cout << std::setw(it_print_width) << (m + 1)
                  << " / " << num_iterations;
          
        std::cout << " [" << std::setw(3) 
                  << static_cast<int>( (100.0 * (m + 1)) / num_iterations )
                  << "%] ";
        std::cout << (warmup ? " (Warmup)" : " (Sampling)");
        std::cout << std::endl;
          
      }
    
    }

    template <class Model, class RNG>
    void print_sample(std::ostream* sample_file_stream,
                      std::ostream* debug_file_stream,
                      stan::mcmc::sample& s,
                      stan::mcmc::base_mcmc* sampler,
                      Model& model,
                      RNG& base_rng) {
      
      // Temporary as model::write_csv isn't a const method
      std::vector<double> cont(s.cont_params());
      std::vector<int> disc(s.disc_params());
      
      *sample_file_stream << s.log_prob() << ",";
      sampler->write_sampler_params(*sample_file_stream);
      model.write_csv(base_rng, cont, disc, 
                      *sample_file_stream, &std::cout);
      
      //sampler.z().write(debug_file_stream);
      //debug_file_stream << std::endl;
      
      
    }
    
    template <class Model, class RNG>
    void run_markov_chain(stan::mcmc::base_mcmc* sampler,
                          int num_iterations,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          std::ostream* sample_file_stream,
                          std::ostream* debug_file_stream,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng) {
      
      for (size_t m = 0; m < num_iterations; ++m) {
      
        print_progress(m, num_iterations, refresh, warmup);
      
        init_s = sampler->transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          print_sample<Model, RNG>(sample_file_stream, debug_file_stream,
                                   init_s, sampler, model, base_rng);
        }

      }
      
    }

    template <class Model, class RNG>
    void warmup(stan::mcmc::base_mcmc* sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream* sample_file_stream,
                std::ostream* debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Model, RNG>(sampler, num_iterations, num_thin, 
                                   refresh, save, true,
                                   sample_file_stream,
                                   debug_file_stream,
                                   init_s, model, base_rng);
      
    }

    template <class Model, class RNG>
    void sample(stan::mcmc::base_mcmc* sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream* sample_file_stream,
                std::ostream* debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Model, RNG>(sampler, num_iterations, num_thin, 
                                   refresh, save, false,
                                   sample_file_stream,
                                   debug_file_stream,
                                   init_s, model, base_rng);
      
    }
    
    template<class Sampler>
    void init_static_hmc(stan::mcmc::base_mcmc* sampler, argument* algorithm) {
      
      categorical_argument* base = dynamic_cast<categorical_argument*>(
                                                algorithm->arg("hmc")->arg("engine")->arg("static"));
      
      double epsilon = dynamic_cast<real_argument*>(base->arg("stepsize"))->value();
      double int_time = dynamic_cast<real_argument*>(base->arg("int_time"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize_and_T(epsilon, int_time);
      
      dynamic_cast<Sampler*>(sampler)->init_stepsize();
      
    }
  
    template<class Sampler>
    void init_nuts(stan::mcmc::base_mcmc* sampler, argument* algorithm) {
      
      categorical_argument* base = dynamic_cast<categorical_argument*>(
                                                algorithm->arg("hmc")->arg("engine")->arg("nuts"));
      
      double epsilon = dynamic_cast<real_argument*>(base->arg("stepsize"))->value();
      int max_depth = dynamic_cast<int_argument*>(base->arg("max_depth"))->value();
      
      dynamic_cast<Sampler*>(sampler)->set_nominal_stepsize(epsilon);
      dynamic_cast<Sampler*>(sampler)->set_max_depth(max_depth);
      
      dynamic_cast<Sampler*>(sampler)->init_stepsize();
      
    }
    
    template<class Sampler>
    void init_adapt(stan::mcmc::base_mcmc* sampler, categorical_argument* adapt) {
      
      double delta = dynamic_cast<real_argument*>(adapt->arg("delta"))->value();
      double gamma = dynamic_cast<real_argument*>(adapt->arg("gamma"))->value();
      double kappa = dynamic_cast<real_argument*>(adapt->arg("kappa"))->value();
      double t0    = dynamic_cast<real_argument*>(adapt->arg("t0"))->value();
      
      double epsilon = dynamic_cast<Sampler*>(sampler)->get_nominal_stepsize();
      
      dynamic_cast<Sampler*>(sampler)->set_adapt_mu(log(10 * epsilon));
      dynamic_cast<Sampler*>(sampler)->set_adapt_delta(delta);
      dynamic_cast<Sampler*>(sampler)->set_adapt_gamma(gamma);
      dynamic_cast<Sampler*>(sampler)->set_adapt_kappa(kappa);
      dynamic_cast<Sampler*>(sampler)->set_adapt_t0(t0);
      
      dynamic_cast<mcmc::stepsize_adapter*>(sampler)->engage_adaptation();
      
    }
    
    template <class Model>
    int nuts_command(int argc, const char* argv[]) {

      std::vector<argument*> valid_arguments;
      valid_arguments.push_back(new arg_help());
      valid_arguments.push_back(new arg_id());
      valid_arguments.push_back(new arg_data());
      valid_arguments.push_back(new arg_init());
      valid_arguments.push_back(new arg_test_gradient());
      valid_arguments.push_back(new arg_random());
      valid_arguments.push_back(new arg_output());
      valid_arguments.push_back(new arg_method());
      
      argument_parser parser(valid_arguments);
      
      parser.parse_args(argc, argv, &std::cout);
      
      // Help
      if (dynamic_cast<unvalued_argument*>(parser.arg("help"))->is_present()) {
        parser.print_help(&std::cout);
        return 0;
      }

      // Identification
      unsigned int id = dynamic_cast<int_argument*>(parser.arg("id"))->value();
      
      //////////////////////////////////////////////////
      //            Random number generator           //
      //////////////////////////////////////////////////
      
      unsigned int random_seed = dynamic_cast<int_argument*>(
                                 parser.arg("random")->arg("seed"))->value();
      if (random_seed < 0) {
        random_seed = (boost::posix_time::microsec_clock::universal_time() -
                       boost::posix_time::ptime(boost::posix_time::min_date_time))
                      .total_milliseconds();
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
      
      if (init == "0") {
        cont_params = std::vector<double>(model.num_params_r(), 0.0);
        disc_params = std::vector<int>(model.num_params_i(), 0);
      } else if (init == "random") {

        boost::random::uniform_real_distribution<double>
        init_range_distribution(-2.0, 2.0);
        
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
                    << "Initialization failed after " << MAX_INIT_TRIES
                    << " attempts. " << std::endl;
          std::cout << " Try specifying initial values,"
                    << " reducing ranges of constrained values,"
                    << " or reparameterizing the model."
                    << std::endl;
          return -1;
        }
        
      } else if (init != "") {
        
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
        
      }
      
      // Test gradient
      if (dynamic_cast<unvalued_argument*>(parser.arg("test_gradient"))->is_present()) {
        std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;
        return model.test_gradients(cont_params, disc_params);
        return 0;
      }
      
      // Initial output
      parser.print(&std::cout);
      
      if (!append_sample && sample_stream) {
        write_stan(sample_stream);
        parser.print(sample_stream);
      }
      
      if (!append_diagnostic && diagnostic_stream) {
        write_stan(diagnostic_stream);
        parser.print(diagnostic_stream);
      }
      
      //////////////////////////////////////////////////
      //           Optimization Algorithms            //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("optimize")) {
        
        list_argument* algo = dynamic_cast<list_argument*>
                              (parser.arg("method")->arg("optimize")->arg("algorithm"));
        
        int num_iterations = dynamic_cast<int_argument*>(
                             parser.arg("method")->arg("sample")->arg("iter"))->value();
        
        bool save_iterations = dynamic_cast<bool_argument*>(
                               parser.arg("method")->arg("sample")->arg("save_iterations"))->value();
        
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
          
          if (sample_stream && !save_iterations) {
            *sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,*sample_stream);
            sample_stream->flush();
          }
          
        } else if (algo->value() == "bfgs") {
          
          if(sample_stream) {
            *sample_stream << "lp__,"; // log probability first
            model.write_csv_header(*sample_stream);
          }
          
          stan::optimization::BFGSLineSearch ng(model, cont_params, disc_params,
                                                &std::cout);
          double lp = ng.logp();
          
          double lastlp = lp - 1;
          std::cout << "initial log joint probability = " << lp << std::endl;
          int m = 0;
          int ret = 0;
          for (size_t i = 0; i < num_iterations && ret == 0; i++) {
            ret = ng.step();
            lastlp = lp;
            lp = ng.logp();
            ng.params_r(cont_params);
            if (do_print(i, refresh)) {
              std::cout << "Iteration ";
              std::cout << std::setw(3) << (m + 1) << ". ";
              std::cout << "Log joint probability = " << std::setw(10) << lp;
              std::cout << ". Improved by " << (lp - lastlp) << ". ";
              std::cout << "Step size " << ng.step_size() << " (initial " << ng.init_step_size() << ").";
              std::cout << " # grad evals = " << ng.grad_evals();
              std::cout << std::endl;
              std::cout.flush();
            }
            m++;
            if (sample_stream && save_iterations) {
              *sample_stream << lp << ',';
              model.write_csv(base_rng,cont_params,disc_params,*sample_stream);
              sample_stream->flush();
            }
          }
          
          if (sample_stream) {
            *sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,*sample_stream);
            sample_stream->flush();
          }
          
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
            //           for (size_t i = 0; i < params_r.size(); i++)
            //             fprintf(stderr, "%f ", params_r[i]);
            //           fprintf(stderr, "   %f  (last = %f)\n", lp, lastlp);
            if (sample_stream && save_iterations) {
              *sample_stream << lp << ',';
              model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
            }
            
          }
          
          if (sample_stream) {
            *sample_stream << lp << ',';
            model.write_csv(base_rng, cont_params, disc_params, *sample_stream);
          }
          
        }
        
      }
      
      //////////////////////////////////////////////////
      //              Sampling Algorithms             //
      //////////////////////////////////////////////////
      
      if (parser.arg("method")->arg("sample")) {
        
        // Sampling parameters
        int num_warmup = dynamic_cast<int_argument*>(
                          parser.arg("method")->arg("sample")->arg("warmup"))->value();
        
        int num_samples = dynamic_cast<int_argument*>(
                          parser.arg("method")->arg("sample")->arg("iter"))->value();
        
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
              
            case 000: {
              typedef stan::mcmc::unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_static_hmc<sampler>(sampler_ptr, algo);
              break;
            }
              
            case 001: {        
              typedef stan::mcmc::unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_nuts<sampler>(sampler_ptr, algo);
              break;
            }
              
            case 010: {
              typedef stan::mcmc::diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_static_hmc<sampler>(sampler_ptr, algo);
              break;
            }
            
            case 011: {
              typedef stan::mcmc::diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_nuts<sampler>(sampler_ptr, algo);
              break;
            }
            
            case 020: {
              typedef stan::mcmc::dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_static_hmc<sampler>(sampler_ptr, algo);
              break;
            }
            
            case 021: {
              typedef stan::mcmc::dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_nuts<sampler>(sampler_ptr, algo);
              break;
            }
            
            case 100: {
              typedef stan::mcmc::adapt_unit_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_static_hmc<sampler>(sampler_ptr, algo);
              init_adapt<sampler>(sampler_ptr, adapt);
              break;
            }
            
            case 101: {
              typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_nuts<sampler>(sampler_ptr, algo);
              init_adapt<sampler>(sampler_ptr, adapt);
              break;
            }
            
            case 110: {
              typedef stan::mcmc::adapt_diag_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_static_hmc<sampler>(sampler_ptr, algo);
              init_adapt<sampler>(sampler_ptr, adapt);
              break;
            }
            
            case 111: {
              typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_nuts<sampler>(sampler_ptr, algo);
              init_adapt<sampler>(sampler_ptr, adapt);
              break;
            }
            
            case 120: {
              typedef stan::mcmc::adapt_dense_e_static_hmc<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_static_hmc<sampler>(sampler_ptr, algo);
              init_adapt<sampler>(sampler_ptr, adapt);
              break;
            }
            
            case 121: {
              typedef stan::mcmc::adapt_dense_e_nuts<Model, rng_t> sampler;
              sampler_ptr = new sampler(model, base_rng);
              init_nuts<sampler>(sampler_ptr, algo);
              init_adapt<sampler>(sampler_ptr, adapt);
              break;
            }
            
            default:
              std::cout << "No sampler matching HMC specification!" << std::endl;
              return 0;
          }
          
        } //hmc
        
        // Headers
        if (!append_sample) {
          *sample_stream << "lp__,";
          sampler_ptr->write_sampler_param_names(*sample_stream);
          model.write_csv_header(*sample_stream);
        }
        
        if(!append_diagnostic) {
          //sampler.z().write_header(diagnostic_stream);
          //sampler.z().write_names(diagnostic_stream);
          //diagnostic_stream << std::endl;
        }
        
        // Warm-Up
        clock_t start = clock();
        
        warmup<Model, rng_t>(sampler_ptr, num_warmup, num_thin,
                             refresh, save_warmup,
                             sample_stream, diagnostic_stream,
                             s, model, base_rng);
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        /*
         if(adapt) {
         dynamic_cast<mcmc::stepsize_adapter*>(sampler_ptr)->disengage_adaptation();
         
         *sample_stream << "# (" << sampler_ptr->name() << ")" << std::endl;
         *sample_stream << "# Adaptation terminated" << std::endl;
         *sample_stream << "# Step size = "
         << dynamic_cast<mcmc::base_hmc*>(sampler_ptr)->get_stepsize() << std::endl;
         (dynamic_cast<mcmc::base_hmc*>(sampler_ptr))->z().write_metric(*sample_stream);
         }
         */
        
        // Sampling
        start = clock();
        
        sample<Model, rng_t>(sampler_ptr, num_samples, num_thin,
                             refresh, true,
                             sample_stream, diagnostic_stream,
                             s, model, base_rng);
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        std::cout << std::endl
                  << "Elapsed Time: " << warmDeltaT
                  << " seconds (Warm Up)"  << std::endl
                  << "              " << sampleDeltaT
                  << " seconds (Sampling)"  << std::endl
                  << "              " << warmDeltaT + sampleDeltaT
                  << " seconds (Total)"  << std::endl
                  << std::endl << std::endl;
        
        //delete sampler_ptr;
        
      } // sample
      
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
