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

    void print_help(std::string cmd) {
      
      using stan::io::print_help_option;
      
      std::cout << std::endl;
      std::cout << "Compiled Stan Graphical Model Command" << std::endl;
      std::cout << std::endl;
      
      std::cout << "USAGE:  " << cmd << " [options]" << std::endl;
      std::cout << std::endl;
      
      std::cout << "OPTIONS:" << std::endl;
      std::cout << std::endl;
      
      print_help_option(&std::cout,
                        "help", "",
                        "Display this information");
      
      print_help_option(&std::cout,
                        "data", "file",
                        "Read data from specified dump-format file",
                        "required if model declares data");
      
      print_help_option(&std::cout,
                        "init", "file",
                        "Use initial values from specified file or zero values if <file>=0",
                        "default is random initialization");
      
      print_help_option(&std::cout,
                        "samples", "file",
                        "File into which samples are written",
                        "default = samples.csv");
      
      print_help_option(&std::cout,
                        "append_samples", "",
                        "Append samples to existing file if it exists",
                        "does not write header in append mode");
      
      print_help_option(&std::cout,
                        "seed", "int",
                        "Random number generation seed",
                        "default = randomly generated from time");
      
      print_help_option(&std::cout,
                        "chain_id", "int",
                        "Markov chain identifier",
                        "default = 1");
      
      print_help_option(&std::cout,
                        "iter", "+int",
                        "Total number of iterations, including warmup",
                        "default = 2000");
      
      print_help_option(&std::cout,
                        "warmup", "+int",
                        "Discard the specified number of initial samples",
                        "default = iter / 2");
      
      print_help_option(&std::cout,
                        "thin", "+int",
                        "Period between saved samples after warm up",
                        "default = max(1, floor(iter - warmup) / 1000)");
      
      print_help_option(&std::cout,
                        "refresh", "int",
                        "Period between samples updating progress report print (0 for no printing)",
                        "default = max(1,iter/200))");
      
      print_help_option(&std::cout,
                        "leapfrog_steps", "int",
                        "Number of leapfrog steps; -1 for no-U-turn adaptation",
                        "default = -1");
      
      print_help_option(&std::cout,
                        "max_treedepth", "int",
                        "Limit NUTS leapfrog steps to 2^max_tree_depth; -1 for no limit",
                        "default = 10");
      
      print_help_option(&std::cout,
                        "epsilon", "float",
                        "Initial value for step size, or -1 to set automatically",
                        "default = -1");
      
      print_help_option(&std::cout,
                        "epsilon_pm", "[0,1]",
                        "Sample epsilon +/- epsilon * epsilon_pm",
                        "default = 0.0");
      
      print_help_option(&std::cout,
                        "equal_step_sizes", "",
                        "Use same step size for every parameter with NUTS",
                        "default is to estimate varying step sizes during warmup");
      
      print_help_option(&std::cout,
                        "delta", "[0,1]",
                        "Accuracy target for step-size adaptation (higher means smaller step sizes)",
                        "default = 0.5");
      
      print_help_option(&std::cout,
                        "gamma", "+float",
                        "Gamma parameter for dual averaging step-size adaptation",
                        "default = 0.05");
      
      print_help_option(&std::cout,
                        "save_warmup", "",
                        "Save the warmup samples");
      
      print_help_option(&std::cout,
                        "test_grad", "",
                        "Test gradient calculations using finite differences");
      
      print_help_option(&std::cout,
                        "point_estimate","",
                        "Fit point estimate of hidden parameters by maximizing log joint probability using Nesterov's accelerated gradient method");
      
      print_help_option(&std::cout,
                        "point_estimate_newton","",
                        "Fit point estimate of hidden parameters by maximizing log joint probability using Newton's method");

      print_help_option(&std::cout,
                        "point_estimate_bfgs","",
                        "Fit point estimate of hidden parameters by maximizing log joint probability using the BFGS method with line search");
      
      print_help_option(&std::cout,
                        "nondiag_mass", "",
                        "Use a nondiagonal matrix to do the sampling");
      
      print_help_option(&std::cout,
                        "cov_matrix", "file",
                        "Preset an estimated covariance matrix");
      
      std::cout << std::endl;
    }
    
    void write_comment(std::ostream& o) {
      o << "#" << std::endl;
    }
    
    template <typename M>
    void write_comment(std::ostream& o,
                       const M& msg) {
      o << "# " << msg << std::endl;
    }
    
    template <typename K, typename V>
    void write_comment_property(std::ostream& o,
                                const K& key,
                                const V& val) {
      o << "# " << key << "=" << val << std::endl;
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

    template <class Sampler, class Model, class RNG>
    void print_sample(std::ostream& sample_file_stream,
                      std::ostream& debug_file_stream,
                      stan::mcmc::sample& s, 
                      Sampler& sampler,
                      Model& model,
                      RNG& base_rng) {
      
      // Temporary as model::write_csv isn't a const method
      std::vector<double> cont(s.cont_params());
      std::vector<int> disc(s.disc_params());
      
      sample_file_stream << s.log_prob() << ",";
      sampler.write_sampler_params(sample_file_stream);
      model.write_csv(base_rng, cont, disc, 
                      sample_file_stream, &std::cout);
      
      //sampler.z().write(debug_file_stream);
      //debug_file_stream << std::endl;
      
      
    }
    
    template <class Sampler, class Model, class RNG>
    void run_markov_chain(Sampler& sampler,
                          int num_iterations,
                          int start,
                          int finish,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          std::ostream& sample_file_stream,
                          std::ostream& debug_file_stream,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng) {
      
      for (size_t m = 0; m < num_iterations; ++m) {
      
        print_progress(m, start, finish, refresh, warmup);
      
        init_s = sampler.transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          print_sample<Sampler, Model, RNG>(sample_file_stream, debug_file_stream,
                                            init_s, sampler, model, base_rng);
        }

      }
      
    }

    template <class Sampler, class Model, class RNG>
    void warmup(Sampler& sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                std::ostream& debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_warmup, 0, num_warmup + num_samples, num_thin,
                                            refresh, save, true,
                                            sample_file_stream,
                                            debug_file_stream,
                                            init_s, model, base_rng);
      
    }

    template <class Sampler, class Model, class RNG>
    void sample(Sampler& sampler,
                int num_warmup,
                int num_samples,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                std::ostream& debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_samples, num_warmup, num_warmup + num_samples, num_thin,
                                            refresh, save, false,
                                            sample_file_stream,
                                            debug_file_stream,
                                            init_s, model, base_rng);
      
    }
    
    template <class Model>
    int nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);
      
      // Call help
      if (command.has_flag("help")) {
        print_help(argv[0]);
        return 0;
      }
      
      // Format data file
      std::string data_file;
      command.val("data",data_file);
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();
      
      // Input arguments
      bool point_estimate = command.has_flag("point_estimate");
      bool point_estimate_newton = command.has_flag("point_estimate_newton");
      bool point_estimate_bfgs = command.has_flag("point_estimate_bfgs");

      std::string sample_file = "samples.csv";
      command.val("samples", sample_file);
      
      // Fix so that default somehow does not produce output
      std::string diagnostic_file = "diagnostic.csv";
      command.val("diagnostic", diagnostic_file);
      
      unsigned int num_iterations = 2000U;
      command.val("iter", num_iterations);
      
      unsigned int num_warmup = num_iterations / 2;
      command.val("warmup", num_warmup);
      
      unsigned int calculated_thin = (num_iterations - num_warmup) / 1000U;
      unsigned int num_thin = (calculated_thin > 1) ? calculated_thin : 1U;
      command.val("thin", num_thin);

      int leapfrog_steps = -1;
      command.val("leapfrog_steps", leapfrog_steps);
      
      double epsilon = -1.0;
      command.val("epsilon", epsilon);
      
      int max_treedepth = 10;
      command.val("max_treedepth", max_treedepth);
      
      double epsilon_pm = 0.0;
      command.val("epsilon_pm",epsilon_pm);
      if (epsilon_pm < 0.0 || epsilon_pm > 1.0) {
        std::stringstream ss;
        ss << "epsilon_pm must be between 0 and 1"
           << "; found epsilon_pm = " << epsilon_pm;
        throw std::invalid_argument(ss.str());
      }
      
      bool equal_step_sizes = command.has_flag("equal_step_sizes");
      
      double delta = 0.5;
      command.val("delta", delta);
      
      double gamma = 0.05;
      command.val("gamma", gamma);
      
      int refresh = num_iterations / 200;
      refresh = refresh <= 0 ? 1 : refresh;
      command.val("refresh", refresh);
      
      bool nondiag_mass = command.has_flag("nondiag_mass");
      
      std::string cov_file = "";
      command.val("cov_matrix", cov_file);
      
      unsigned int random_seed = 0;
      if (command.has_key("seed")) {
        bool well_formed = command.val("seed",random_seed);
        if (!well_formed) {
          std::string seed_val;
          command.val("seed", seed_val);
          std::cerr << "value for seed must be integer"
                    << "; found value = " << seed_val << std::endl;
          return -1;
        }
      } else {
        random_seed = (boost::posix_time::microsec_clock::universal_time() -
                       boost::posix_time::ptime(boost::posix_time::min_date_time))
                      .total_milliseconds();
      }
      
      int chain_id = 1;
      if (command.has_key("chain_id")) {
        bool well_formed = command.val("chain_id", chain_id);
        if (!well_formed || chain_id <= 0) {
          std::string chain_id_val;
          command.val("chain_id", chain_id_val);
          std::cerr << "value for chain_id must be positive integer"
                    << "; found chain_id = " << chain_id_val
                    << std::endl;
          return -1;
        }
      }
      
      bool save_warmup = command.has_flag("save_warmup");
      
      bool append_samples = command.has_flag("append_samples");
      std::ios_base::openmode samples_append_mode 
        = append_samples
          ? (std::fstream::out | std::fstream::app)
          : std::fstream::out;

      // Instatitate random number generator and model
      
      // (2**50 = 1T samples, 1000 chains)
      typedef boost::ecuyer1988 rng_t;
      rng_t base_rng(random_seed);
      
      // DISCARD_STRIDE <<= 50;
      static boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50;
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      
      Model model(data_var_context, &std::cout);
      
      std::vector<double> cont_params(model.num_params_r());
      std::vector<int> disc_params(model.num_params_i());
      
      int num_init_tries = -1;
      
      std::string init_val;
      
      if (command.has_key("init")) {
        command.val("init", init_val);
        if (init_val == "0") {
          cont_params = std::vector<double>(model.num_params_r(), 0.0);
          disc_params = std::vector<int>(model.num_params_i(), 0);
        } else {
          
          try {
            std::fstream init_stream(init_val.c_str(), std::fstream::in);
            if (init_stream.fail()) {
              std::string msg("ERROR: specified init file does not exist: ");
              msg += init_val;
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
      } else {
        init_val = "random initialization";
        
        boost::random::uniform_real_distribution<double> 
          init_range_distribution(-2.0, 2.0);
        
        boost::variate_generator<rng_t&, 
                                boost::random::uniform_real_distribution<double> >
          init_rng(base_rng, init_range_distribution);

        cont_params = std::vector<double>(model.num_params_r());
        disc_params = std::vector<int>(model.num_params_i(), 0);
        
        // Try random initializations until log_prob is finite
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
                    << "Initialization failed after " << MAX_INIT_TRIES 
                    << " attempts. " << std::endl;
          std::cout << " Try specifying initial values,"
                    << " reducing ranges of constrained values,"
                    << " or reparameterizing the model."
                    << std::endl;
          return -1;
        }
        
      }
      
      if (command.has_flag("test_grad")) {
        std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;
        return model.test_gradients(cont_params, disc_params);
      }
      
      //////////////////////////////////////////////////
      //           Optimization Algorithms            //
      //////////////////////////////////////////////////
      
      if (point_estimate_newton) {
        
        std::cout << "STAN OPTIMIZATION COMMAND" << std::endl;
        if (data_file == "")
          std::cout << "data = (specified model requires no data)" << std::endl;
        else 
          std::cout << "data = " << data_file << std::endl;
        
        std::cout << "init = " << init_val << std::endl;
        if (num_init_tries > 0)
          std::cout << "init tries = " << num_init_tries << std::endl;
        
        std::cout << "output = " << sample_file << std::endl;
        std::cout << "save_warmup = " << save_warmup<< std::endl;
        
        std::cout << "seed = " << random_seed 
                  << " (" << (command.has_key("seed") 
                    ? "user specified"
                    : "randomly generated") << ")"
                  << std::endl;
        
        std::fstream sample_stream(sample_file.c_str(), 
                                   samples_append_mode);
        
        write_comment(sample_stream,"Point Estimate Generated by Stan");
        write_comment(sample_stream);
        write_comment_property(sample_stream, "stan_version_major", stan::MAJOR_VERSION);
        write_comment_property(sample_stream, "stan_version_minor", stan::MINOR_VERSION);
        write_comment_property(sample_stream, "stan_version_patch", stan::PATCH_VERSION);
        write_comment_property(sample_stream, "data", data_file);
        write_comment_property(sample_stream, "init", init_val);
        write_comment_property(sample_stream, "save_warmup", save_warmup);
        write_comment_property(sample_stream, "seed", random_seed);
        write_comment(sample_stream);
        
        sample_stream << "lp__,";
        model.write_csv_header(sample_stream);
        
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
          if (save_warmup) {
            sample_stream << lp << ',';
            model.write_csv(base_rng, cont_params, disc_params, sample_stream);
          }
        }
        
        sample_stream << lp << ',';
        model.write_csv(base_rng, cont_params, disc_params, sample_stream);
        
        return 0;
        
      }
      
      if (point_estimate) {
        
        std::cout << "STAN OPTIMIZATION COMMAND" << std::endl;
        if (data_file == "")
          std::cout << "data = (specified model requires no data)" << std::endl;
        else 
          std::cout << "data = " << data_file << std::endl;
        
        std::cout << "init = " << init_val << std::endl;
        if (num_init_tries > 0)
          std::cout << "init tries = " << num_init_tries << std::endl;
        
        std::cout << "output = " << sample_file << std::endl;
        std::cout << "save_warmup = " << save_warmup<< std::endl;
        
        std::cout << "seed = " << random_seed 
                  << " (" << (command.has_key("seed") 
                    ? "user specified"
                    : "randomly generated") << ")"
                  << std::endl;
        
        std::fstream sample_stream(sample_file.c_str(), 
                                   samples_append_mode);
        
        write_comment(sample_stream,"Point Estimate Generated by Stan");
        write_comment(sample_stream);
        write_comment_property(sample_stream, "stan_version_major", stan::MAJOR_VERSION);
        write_comment_property(sample_stream, "stan_version_minor", stan::MINOR_VERSION);
        write_comment_property(sample_stream, "stan_version_patch", stan::PATCH_VERSION);
        write_comment_property(sample_stream, "data", data_file);
        write_comment_property(sample_stream, "init", init_val);
        write_comment_property(sample_stream, "save_warmup", save_warmup);
        write_comment_property(sample_stream, "seed", random_seed);
        write_comment(sample_stream);
        
        sample_stream << "lp__,";
        model.write_csv_header(sample_stream);

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
          if (save_warmup) {
            sample_stream << lp << ',';
            model.write_csv(base_rng, cont_params, disc_params, sample_stream);
            sample_stream.flush();
          }
        }
        
        if (!save_warmup) {
          sample_stream << lp << ',';
          model.write_csv(base_rng,cont_params,disc_params,sample_stream);
          sample_stream.flush();
        }

        return 0;
      }
      
      if (point_estimate_bfgs) {
        std::cout << "STAN OPTIMIZATION COMMAND" << std::endl;
        if (data_file == "")
          std::cout << "data = (specified model requires no data)" << std::endl;
        else 
          std::cout << "data = " << data_file << std::endl;
        
        std::cout << "init = " << init_val << std::endl;
        if (num_init_tries > 0)
          std::cout << "init tries = " << num_init_tries << std::endl;
        
        std::cout << "output = " << sample_file << std::endl;
        std::cout << "save_warmup = " << save_warmup << std::endl;
        std::cout << "epsilon = " << epsilon << std::endl;
        
        std::cout << "seed = " << random_seed 
        << " (" << (command.has_key("seed") 
                    ? "user specified"
                    : "randomly generated") << ")"
        << std::endl;
        
        std::fstream sample_stream(sample_file.c_str(), 
                                   samples_append_mode);
        
        write_comment(sample_stream,"Point Estimate Generated by Stan");
        write_comment(sample_stream);
        write_comment_property(sample_stream,"stan_version_major",stan::MAJOR_VERSION);
        write_comment_property(sample_stream,"stan_version_minor",stan::MINOR_VERSION);
        write_comment_property(sample_stream,"stan_version_patch",stan::PATCH_VERSION);
        write_comment_property(sample_stream,"data",data_file);
        write_comment_property(sample_stream,"init",init_val);
        write_comment_property(sample_stream,"save_warmup",save_warmup);
        write_comment_property(sample_stream,"seed",random_seed);
        write_comment_property(sample_stream,"epsilon",epsilon);
        write_comment(sample_stream);
        
        sample_stream << "lp__,"; // log probability first
        model.write_csv_header(sample_stream);
        
        stan::optimization::BFGSLineSearch ng(model, cont_params, disc_params,
                                              &std::cout);
        if (epsilon > 0)
          ng._opts.alpha0 = epsilon;
        
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
          if (do_print(i, refresh)) {
            std::cout << " " << std::setw(7) << (m + 1) << " ";
            std::cout << " " << std::setw(12) << std::setprecision(6) << lp << " ";
            std::cout << " " << std::setw(12) << std::setprecision(6) << ng.prev_step_size() << " ";
            std::cout << " " << std::setw(12) << std::setprecision(6) << ng.curr_g().norm() << " ";
            std::cout << " " << std::setw(10) << std::setprecision(4) << ng.alpha() << " ";
            std::cout << " " << std::setw(10) << std::setprecision(4) << ng.alpha0() << " ";
            std::cout << " " << std::setw(7) << ng.grad_evals() << " ";
            std::cout << " " << ng.note() << " ";
            std::cout << std::endl;
            std::cout.flush();
          }
          m++;
          if (save_warmup) {
            sample_stream << lp << ',';
            model.write_csv(base_rng,cont_params,disc_params,sample_stream);
            sample_stream.flush();
          }
        }
        if (ret != 0)
          std::cout << "Optimization terminated with code " << ret << std::endl;
        
        sample_stream << lp << ',';
        model.write_csv(base_rng,cont_params,disc_params,sample_stream);
        sample_stream.flush();
        
        return 0;
      }
      
      //////////////////////////////////////////////////
      //             Sampling Algorithms              // 
      //////////////////////////////////////////////////
      
      std::cout << "STAN SAMPLING COMMAND" << std::endl;
      if (data_file == "")
        std::cout << "data = (specified model requires no data)" << std::endl;
      else 
        std::cout << "data = " << data_file << std::endl;
      
      std::cout << "init = " << init_val << std::endl;
      if (num_init_tries > 0)
        std::cout << "init tries = " << num_init_tries << std::endl;
      
      std::cout << "samples = " << sample_file << std::endl;
      std::cout << "append_samples = " << append_samples << std::endl;
      std::cout << "save_warmup = " << save_warmup<< std::endl;
      
      std::cout << "seed = " << random_seed 
                << " (" << (command.has_key("seed") 
                  ? "user specified"
                  : "randomly generated") << ")"
                << std::endl;
      std::cout << "chain_id = " << chain_id
                << " (" << (command.has_key("chain_id")
                  ? "user specified"
                  : "default") << ")"
                << std::endl;
      
      std::cout << "iter = " << num_iterations << std::endl;
      std::cout << "warmup = " << num_warmup << std::endl;
      std::cout << "thin = " << num_thin << std::endl;
      
      std::cout << "equal_step_sizes = " << equal_step_sizes << std::endl;
      std::cout << "nondiag_mass = " << nondiag_mass << std::endl;
      std::cout << "leapfrog_steps = " << leapfrog_steps << std::endl;
      std::cout << "max_treedepth = " << max_treedepth << std::endl;;
      std::cout << "epsilon = " << epsilon << std::endl;;
      std::cout << "epsilon_pm = " << epsilon_pm << std::endl;;
      std::cout << "delta = " << delta << std::endl;
      std::cout << "gamma = " << gamma << std::endl;
      
      std::fstream sample_stream(sample_file.c_str(), 
                                 samples_append_mode);
      
      std::fstream diagnostic_stream(diagnostic_file.c_str(), 
                                 std::fstream::out);
      
      write_comment(sample_stream,"Samples Generated by Stan");
      write_comment(sample_stream);
      write_comment_property(sample_stream, "stan_version_major", stan::MAJOR_VERSION);
      write_comment_property(sample_stream, "stan_version_minor", stan::MINOR_VERSION);
      write_comment_property(sample_stream, "stan_version_patch", stan::PATCH_VERSION);
      write_comment_property(sample_stream, "data", data_file);
      write_comment_property(sample_stream, "init", init_val);
      write_comment_property(sample_stream, "append_samples", append_samples);
      write_comment_property(sample_stream, "save_warmup", save_warmup);
      write_comment_property(sample_stream, "seed", random_seed);
      write_comment_property(sample_stream, "chain_id", chain_id);
      write_comment_property(sample_stream, "iter", num_iterations);
      write_comment_property(sample_stream, "warmup", num_warmup);
      write_comment_property(sample_stream, "thin", num_thin);
      write_comment_property(sample_stream, "nondiag_mass", nondiag_mass);
      write_comment_property(sample_stream, "equal_step_sizes", equal_step_sizes);
      write_comment_property(sample_stream, "leapfrog_steps", leapfrog_steps);
      write_comment_property(sample_stream, "max_treedepth", max_treedepth);
      write_comment_property(sample_stream, "epsilon", epsilon);
      write_comment_property(sample_stream, "epsilon_pm", epsilon_pm);
      write_comment_property(sample_stream, "delta", delta);
      write_comment_property(sample_stream, "gamma", gamma);
      write_comment(sample_stream);
      
      double warmDeltaT;
      double sampleDeltaT;
      
      if (nondiag_mass) {

        // Euclidean NUTS with Dense Metric
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        
        typedef stan::mcmc::adapt_dense_e_nuts<Model, rng_t> a_Dm_nuts;
        a_Dm_nuts sampler(model, base_rng, num_warmup);
        sampler.seed(cont_params, disc_params);
        
        if (!append_samples) {
          sample_stream << "lp__,";
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }
        
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else             sampler.set_nominal_stepsize(epsilon);
        
        sampler.set_stepsize_jitter(epsilon_pm);
        
        sampler.set_max_depth(max_treedepth);
        
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        
        clock_t start = clock();
        
        warmup<a_Dm_nuts, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin,
                                        refresh, save_warmup, 
                                        sample_stream, diagnostic_stream,
                                        s, model, base_rng); 
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        sampler.disengage_adaptation();

        sample_stream << "# (" << sampler.name() << ")" << std::endl;
        sample_stream << "# Adaptation terminated" << std::endl;
        sample_stream << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(sample_stream);
        
        // Sampling
        start = clock();
        
        sample<a_Dm_nuts, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin,
                                        refresh, true, 
                                        sample_stream, diagnostic_stream, 
                                        s, model, base_rng); 
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
      }
      else if (leapfrog_steps < 0 && !equal_step_sizes) {
        
        // Euclidean NUTS with Diagonal Metric
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        
        typedef stan::mcmc::adapt_diag_e_nuts<Model, rng_t> a_dm_nuts;
        a_dm_nuts sampler(model, base_rng, num_warmup);
        sampler.seed(cont_params, disc_params);
        
        if (!append_samples) {
          sample_stream << "lp__,";
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);          
        }
        
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else             sampler.set_nominal_stepsize(epsilon);
        
        sampler.set_stepsize_jitter(epsilon_pm);
        
        sampler.set_max_depth(max_treedepth);
        
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        
        clock_t start = clock();
        
        warmup<a_dm_nuts, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin, 
                                        refresh, save_warmup, 
                                        sample_stream, diagnostic_stream,
                                        s, model, base_rng); 
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        sampler.disengage_adaptation();
        
        sample_stream << "# (" << sampler.name() << ")" << std::endl;
        sample_stream << "# Adaptation terminated" << std::endl;
        sample_stream << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(sample_stream);
        
        // Sampling
        start = clock();
        
        sample<a_dm_nuts, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin,
                                        refresh, true, 
                                        sample_stream, diagnostic_stream, 
                                        s, model, base_rng);
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;

        
      } else if (leapfrog_steps < 0 && equal_step_sizes) {
        
        // Euclidean NUTS with Unit Metric
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        
        typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> a_um_nuts;
        a_um_nuts sampler(model, base_rng);
        sampler.seed(cont_params, disc_params);
        
        if (!append_samples) {
          sample_stream << "lp__,";
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }
        
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else             sampler.set_nominal_stepsize(epsilon);
        
        sampler.set_stepsize_jitter(epsilon_pm);
        
        sampler.set_max_depth(max_treedepth);
        
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        
        clock_t start = clock();
        
        warmup<a_um_nuts, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin, 
                                        refresh, save_warmup, 
                                        sample_stream, diagnostic_stream,
                                        s, model, base_rng); 
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        sampler.disengage_adaptation();

        sample_stream << "# (" << sampler.name() << ")" << std::endl;
        sample_stream << "# Adaptation terminated" << std::endl;
        sample_stream << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(sample_stream);
        
        // Sampling
        start = clock();
        
        sample<a_um_nuts, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin,
                                        refresh, true, 
                                        sample_stream, diagnostic_stream, 
                                        s, model, base_rng); 
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
      } else {
        
        // Unit Metric HMC with Static Integration Time
        stan::mcmc::sample s(cont_params, disc_params, 0, 0);
        
        typedef stan::mcmc::adapt_unit_e_static_hmc<Model, rng_t> a_um_hmc;
        a_um_hmc sampler(model, base_rng);
        sampler.seed(cont_params, disc_params);
        
        if (!append_samples) {
          sample_stream << "lp__,";
          sampler.write_sampler_param_names(sample_stream);
          model.write_csv_header(sample_stream);
        }
        
        // Warm-Up
        if (epsilon <= 0) sampler.init_stepsize();
        else              sampler.set_nominal_stepsize(epsilon);
        
        sampler.set_stepsize_jitter(epsilon_pm);
        
        sampler.set_nominal_stepsize_and_L(epsilon, leapfrog_steps);
        
        sampler.get_stepsize_adaptation().set_delta(delta);
        sampler.get_stepsize_adaptation().set_gamma(gamma);
        sampler.get_stepsize_adaptation().set_mu(log(10 * sampler.get_nominal_stepsize()));
        sampler.engage_adaptation();
        
        clock_t start = clock();
        
        warmup<a_um_hmc, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin, 
                                       refresh, save_warmup, 
                                       sample_stream, diagnostic_stream,
                                       s, model, base_rng); 
        
        clock_t end = clock();
        warmDeltaT = (double)(end - start) / CLOCKS_PER_SEC;
        
        sampler.disengage_adaptation();

        sample_stream << "# (" << sampler.name() << ")" << std::endl;
        sample_stream << "# Adaptation terminated" << std::endl;
        sample_stream << "# Step size = " << sampler.get_nominal_stepsize() << std::endl;
        sampler.z().write_metric(sample_stream);
        
        // Sampling
        start = clock();
        
        sample<a_um_hmc, Model, rng_t>(sampler, num_warmup, num_iterations - num_warmup, num_thin,
                                       refresh, true, 
                                       sample_stream, diagnostic_stream, 
                                       s, model, base_rng); 
        
        end = clock();
        sampleDeltaT = (double)(end - start) / CLOCKS_PER_SEC;

      }
      
      std::cout << std::endl
                << "Elapsed Time: " << warmDeltaT 
                << " seconds (Warm Up)"  << std::endl
                << "              " << sampleDeltaT 
                << " seconds (Sampling)"  << std::endl
                << "              " << warmDeltaT + sampleDeltaT 
                << " seconds (Total)"  << std::endl
                << std::endl << std::endl;
      
      sample_stream.close();
      
      
      return 0;
      
    }

  } // namespace prob


} // namespace stan

#endif
