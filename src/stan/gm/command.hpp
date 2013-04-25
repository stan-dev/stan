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
      
        print_progress(m, num_iterations, refresh, warmup);
      
        init_s = sampler.transition(init_s);
          
        if ( save && ( (m % num_thin) == 0) ) {
          print_sample<Sampler, Model, RNG>(sample_file_stream, debug_file_stream,
                                            init_s, sampler, model, base_rng);
        }

      }
      
    }

    template <class Sampler, class Model, class RNG>
    void warmup(Sampler& sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                std::ostream& debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, true,
                                            sample_file_stream,
                                            debug_file_stream,
                                            init_s, model, base_rng);
      
    }

    template <class Sampler, class Model, class RNG>
    void sample(Sampler& sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                std::ostream& debug_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, false,
                                            sample_file_stream,
                                            debug_file_stream,
                                            init_s, model, base_rng);
      
    }
    
    template <class Model>
    int nuts_command(int argc, const char* argv[]) {

      std::vector<argument*> valid_arguments;
      valid_arguments.push_back(new arg_output());
      valid_arguments.push_back(new arg_method());
      
      argument_parser parser(valid_arguments);
      
      parser.print(&std::cout);
      std::cout << std::endl;
      
      parser.print_help(&std::cout);
      std::cout << std::endl;
      
      parser.parse_args(argc, argv, &std::cout);
      std::cout << std::endl;
      
      //real_argument* arg = dynamic_cast<real_argument*>(
      //                     parser.arg("hmc")->arg("engine")->arg("nuts")->arg("stepsize"));
      //std::cout << arg->print_value() << std::endl;
      //std::cout << std::endl;
      
      parser.print(&std::cout);
      
      return 0;
 
    }

  } // namespace prob


} // namespace stan

#endif
