#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <fstream>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/uniform_real_distribution.hpp>

#include <stan/version.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>

#include <stan/mcmc/static_hmc.hpp>

#include <stan/optimization/newton.hpp>
#include <stan/optimization/nesterov_gradient.hpp>

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
                        "point_estimate", "",
                        "Fit point estimate of hidden parameters by maximizing log joint probability using Nesterov's accelerated gradient method");
      
      print_help_option(&std::cout,
                        "point_estimate_newton", "",
                        "Fit point estimate of hidden parameters by maximizing log joint probability using Newton's method");
      
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
    
    bool do_print(int n, int refresh) {
      return (refresh > 0) &&
      (n == 0 || ((n + 1) % refresh == 0) );
    }

    void print_progress(int m, int num_iterations, int refresh, bool warmup) {
      
      int it_print_width = std::ceil(std::log10(num_iterations));
      
      if (do_print(m, refresh)) {
        
        std::cout << "Printing! m = " << m << ", refresh = " << refresh << std::endl;
          
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
      
    }
    
    template <class Sampler, class Model, class RNG>
    void run_markov_chain(Sampler& sampler,
                          int num_iterations,
                          int num_thin,
                          int refresh,
                          bool save,
                          bool warmup,
                          std::ostream& sample_file_stream,
                          stan::mcmc::sample& init_s,
                          Model& model,
                          RNG& base_rng) {
      
      for (size_t m = 0; m < num_iterations; ++m) {
      
        print_progress(m, num_iterations, refresh, warmup);
      
        std::cout << "Testing transition in run_markov_chains function" << std::endl;
        std::cout << "Before calling transition, address of sampler = " << &sampler << std::endl;
        sampler.transition(init_s);
        std::cout << "After calling transition address of sampler = " << &sampler << std::endl << std::endl;
        //init_s = s;
          
        if ( save && ( (m % num_thin) == 0) ) {
          print_sample<Sampler, Model, RNG>(sample_file_stream, init_s, 
                                            sampler, model, base_rng);
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
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, true,
                                            sample_file_stream,
                                            init_s, model, base_rng);
      
    }

    template <class Sampler, class Model, class RNG>
    void sample(Sampler& sampler,
                int num_iterations,
                int num_thin,
                int refresh,
                bool save,
                std::ostream& sample_file_stream,
                stan::mcmc::sample& init_s,
                Model& model,
                RNG& base_rng) {
      
      run_markov_chain<Sampler, Model, RNG>(sampler, num_iterations, num_thin, 
                                            refresh, save, false,
                                            sample_file_stream,
                                            init_s, model, base_rng);
      
    }
    
    template <class Model>
    int nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);

      if (command.has_flag("help")) {
        print_help(argv[0]);
        return 0;
      }

      std::string data_file;
      command.val("data",data_file);
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      Model model(data_var_context, &std::cout);

      //stan::model::prob_grad model(data_var_context, &std::cout);
      
      typedef boost::ecuyer1988 rng_t;
      
      unsigned int random_seed = 0;
      rng_t base_rng(random_seed);
      
      /*
      // (2**50 = 1T samples, 1000 chains)
      //static boost::uintmax_t DISCARD_STRIDE = static_cast<boost::uintmax_t>(1) << 50;
      
      // DISCARD_STRIDE <<= 50;
      //base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      */
      
      std::vector<double> q(model.num_params_r(), 1.0);
      std::vector<int> r;
      
      stan::mcmc::sample s(q, r, 0, 0);
      //std::cout << "Before: " << s.cont_params(0) << std::endl;
      
      std::fstream sample_stream("out.txt", std::fstream::out);
      
      //typedef stan::mcmc::adapt_unit_metric_hmc<Model, rng_t> a_um_hmc;
      typedef stan::mcmc::unit_metric_hmc<Model, rng_t> a_um_hmc;
      a_um_hmc sampler(model, base_rng);
      
      std::cout << "Testing transition in main function (1)" << std::endl;
      std::cout << "Before calling transition, address of sampler = " << &sampler << std::endl;
      sampler.transition(s);
      std::cout << "After calling transition address of sampler = " << &sampler << std::endl << std::endl;
      
      std::cout << "Testing transition in main function (2)" << std::endl;
      std::cout << "Before calling transition, address of sampler = " << &sampler << std::endl;
      sampler.transition(s);
      std::cout << "After calling transition address of sampler = " << &sampler << std::endl << std::endl;
      
      std::cout << "Testing transition in main function (3)" << std::endl;
      std::cout << "Before calling transition, address of sampler = " << &sampler << std::endl;
      sampler.transition(s);
      std::cout << "After calling transition address of sampler = " << &sampler << std::endl << std::endl;
      
      std::cout << "Testing transition in main function (4)" << std::endl;
      std::cout << "Before calling transition, address of sampler = " << &sampler << std::endl;
      sampler.transition(s);
      std::cout << "After calling transition address of sampler = " << &sampler << std::endl << std::endl;
      
      //sampler.engage_adaptation();
      
      warmup<a_um_hmc, Model, rng_t>(sampler, 10, 1, 2, true, sample_stream, s, model, base_rng); 
      
      //sampler.disengage_adaptation();
      sampler.write_sampler_params(sample_stream);
      
      //sample<a_um_hmc, Model, rng_t>(sampler, 10, 1, 2, true, sample_stream, s, model, base_rng); 
      
      //std::cout << "After: " << s.cont_params(0) << std::endl;
      
      sample_stream.close();
      
      return 0;
      
    }

  } // namespace prob


} // namespace stan

#endif
