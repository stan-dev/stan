#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <cmath>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/mcmc/adaptive_sampler.hpp>
#include <stan/mcmc/adaptive_hmc.hpp>
#include <stan/mcmc/hmc.hpp>
#include <stan/mcmc/nuts.hpp>
#include <stan/model/prob_grad_ad.hpp>
#include <stan/model/prob_grad.hpp>
#include <stan/mcmc/sampler.hpp>

namespace stan {

  namespace gm {

    void print_help_2(const std::string& key_val,
                      const std::string& msg,
                      const std::string& note = "") {
      stan::io::pad_help_option(key_val);
      std::cout << msg 
                << std::endl;
      if (note.size() > 0) {
        stan::io::pad_help_option("");
        std::cout << "    (" << note << ")" 
                  << std::endl;
      }
      std::cout << std::endl;
    }

    void print_help_option(const std::string& key,
                           const std::string& value_type,
                           const std::string& msg,
                           const std::string& note = "") {
      std::stringstream ss;
      ss << "--" << key;
      if (value_type.size() > 0)
        ss << "=<" << value_type << ">";
      print_help_2(ss.str(),msg,note);
    }
        
    void print_nuts_help(std::string cmd) {
      using stan::io::pad_help_option;

      std::cout << std::endl;
      std::cout << "Compiled Stan Graphical Model Command" << std::endl;
      std::cout << std::endl;

      std::cout << "USAGE:  " << cmd << " [options]" << std::endl;
      std::cout << std::endl;

      std::cout << "OPTIONS:" << std::endl;
      std::cout << std::endl;

      print_help_option("help","",
                        "Display this information");

      print_help_option("data","file",
                        "Read data from specified dump-format file",
                        "required if model declares data");
      
      print_help_option("init","file",
                        "Use initial values from specified file or zero values if <file>=0",
                        "default is random initialization");

      print_help_option("samples","file",
                        "File into which samples are written",
                        "default = samples.csv");

      print_help_option("append_samples","",
                        "Append samples to existing file if it exists",
                        "does not write header in append mode");

      print_help_option("seed","int",
                        "Random number generation seed",
                        "default = randomly generated from time");

      print_help_option("chain_id","int",
                        "Markov chain identifier",
                        "default = 1");

      print_help_option("iter","+int",
                        "Total number of iterations, including warmup",
                        "default = 2000");

      print_help_option("warmup","+int",
                        "Discard the specified number of initial samples",
                        "default = iter / 2");

      print_help_option("thin","+int",
                        "Period between saved samples after warm up",
                        "default = max(1, floor(iter - warmup) / 1000)");

      print_help_option("refresh","+int"
                        "Period between samples updating progress report print",
                        "default = max(1,iter/200))");

      print_help_option("leapfrog_steps","int",
                        "Number of leapfrog steps; -1 for No-U-Turn adaptation",
                        "default = -1");

      print_help_option("epsilon","float",
                        "Initial value for step size, or -1 to set automatically",
                        "default = -1");
      
      print_help_option("epsilon_pm","[0,1]",
                        "Sample epsilon +/- epsilon * epsilon_pm",
                        "default = 0.0");

      print_help_option("epsilon_adapt_off","",
                        "Turn off step size adaptation (default is on)");

      print_help_option("delta","+float",
                        "Initial parameter for NUTS step-size tuning.",
                        "default = 0.5");

      print_help_option("gamma","+float",
                        "Gamma parameter for dual averaging step-size adaptation.",
                        "default = 0.05");

      print_help_option("test_grad","",
                        "Test gradient calculations using finite differences");
      
      std::cout << std::endl;
    }

    bool do_print(int refresh) {
      return refresh > 0;
    }

    bool do_print(int n, int refresh) {
      return do_print(refresh)
        && ((n + 1) % refresh == 0);
    }

    template <typename T_model>
    void sample_from(stan::mcmc::adaptive_sampler& sampler,
                     bool epsilon_adapt,
                     int refresh,
                     int num_iterations,
                     int num_warmup,
                     int num_thin,
                     std::ostream& sample_file_stream,
                     std::vector<double>& params_r,
                     std::vector<int>& params_i,
                     T_model& model) {
     
      int it_print_width = std::ceil(std::log10(num_iterations));
      std::cout << std::endl;

      if (epsilon_adapt)
        sampler.adapt_on(); 
      for (int m = 0; m < num_iterations; ++m) {
        if (do_print(m,refresh)) {
          std::cout << "\rIteration: ";
          std::cout << std::setw(it_print_width) << (m + 1)
                    << " / " << num_iterations;
          std::cout << " [" << std::setw(3) 
                    << static_cast<int>((100.0 * (m + 1))/num_iterations)
                    << "%] ";
          std::cout << ((m < num_warmup) ? " (Adapting)" : " (Sampling)");
          std::cout.flush();
        }
        if (m < num_warmup) {
          sampler.next(); // discard
        } else {
          if (epsilon_adapt)
            sampler.adapt_off();
          if (((m - num_warmup) % num_thin) != 0) {
            sampler.next();
            continue;
          } else {
            stan::mcmc::sample sample = sampler.next();

            // FIXME: use csv_writer arg to make comma optional?
            sample_file_stream << sample.log_prob() << ',';
            sample.params_r(params_r);
            sample.params_i(params_i);
            model.write_csv(params_r,params_i,sample_file_stream);
          }
        }
      }
    }

    void write_comment(std::ostream& o,
                       const std::string& msg) {
      o << "# " << msg << std::endl;
    }

    template <typename T_model>
    int nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);

      if (command.has_flag("help")) {
        print_nuts_help(argv[0]);
        return 0;
      }

      std::string data_file;
      command.val("data",data_file);
      std::fstream data_stream(data_file.c_str(),
                               std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      T_model model(data_var_context);

      std::string sample_file = "samples.csv";
      command.val("samples",sample_file);
      
      unsigned int num_iterations = 2000U;
      command.val("iter",num_iterations);
      
      unsigned int num_warmup = num_iterations / 2;
      command.val("warmup",num_warmup);
      
      unsigned int calculated_thin = (num_iterations - num_warmup) / 1000U;
      unsigned int num_thin = (calculated_thin > 1) ? calculated_thin : 1U;
      command.val("thin",num_thin);

      int leapfrog_steps = -1;
      command.val("leapfrog_steps",leapfrog_steps);

      double epsilon = -1.0;
      command.val("epsilon",epsilon);

      double epsilon_pm = 0.0;
      command.val("epsilon_pm",epsilon_pm);

      bool epsilon_adapt_off = command.has_flag("epsilon_adapt_off");
      bool epsilon_adapt = !epsilon_adapt_off;

      double delta = 0.5;
      command.val("delta", delta);

      double gamma = 0.05;
      command.val("gamma", gamma);

      int refresh = 1;
      command.val("refresh",refresh);

      int random_seed = 0;
      if (command.has_key("seed")) {
        bool well_formed = command.val("seed",random_seed);
        if (!well_formed) {
          std::string seed_val;
          command.val("seed",seed_val);
          std::cerr << "value for seed must be integer"
                    << "; found value=" << seed_val << std::endl;
          return -1;
        }
      } else {
        random_seed = std::time(0);
      }

      int chain_id = 1;
      if (command.has_key("chain_id")) {
        bool well_formed = command.val("chain_id",chain_id);
        if (!well_formed || chain_id < 0) {
          std::string chain_id_val;
          command.val("chain_id",chain_id_val);
          std::cerr << "value for chain_id must be positive integer"
                    << "; found chain_id=" << chain_id_val
                    << std::endl;
          return -1;
        }
      }
      
      // FASTER, but no parallel guarantees:
      // typedef boost::mt19937 rng_t;
      // rng_t base_rng(static_cast<unsigned int>(random_seed + chain_id - 1);

      typedef boost::ecuyer1988 rng_t;
      rng_t base_rng(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static long unsigned int DISCARD_STRIDE = (1UL << 50);
      base_rng.discard(DISCARD_STRIDE * (chain_id - 1));
      
      std::vector<int> params_i;
      std::vector<double> params_r;

      std::string init_val;
      // parameter initialization
      if (command.has_key("init")) {
        command.val("init",init_val);
        if (init_val == "0") {
          params_i = std::vector<int>(model.num_params_i(),0);
          params_r = std::vector<double>(model.num_params_r(),0.0);
        } else {
          std::cout << "init file=" << init_val << std::endl;
        
          std::fstream init_stream(init_val.c_str(),std::fstream::in);
          stan::io::dump init_var_context(init_stream);
          init_stream.close();
          model.transform_inits(init_var_context,params_i,params_r);
        }
      } else {
        init_val = "random initialization";
        // init_rng generates uniformly from -2 to 2
        boost::random::uniform_real_distribution<double> 
          init_range_distribution(-2.0,2.0);
        boost::variate_generator<rng_t&, 
                       boost::random::uniform_real_distribution<double> >
          init_rng(base_rng,init_range_distribution);

        params_i = std::vector<int>(model.num_params_i(),0);
        params_r = std::vector<double>(model.num_params_r());
        for (size_t i = 0; i < params_r.size(); ++i)
          params_r[i] = init_rng();
      }

      bool append_samples = command.has_flag("append_samples");
      std::ios_base::openmode samples_append_mode
        = append_samples
        ? (std::fstream::out | std::fstream::app)
        : std::fstream::out;
      

      std::cout << "STAN SAMPLING COMMAND" << std::endl;
      std::cout << "data = " << data_file << std::endl;
      std::cout << "init = " << init_val << std::endl;
      std::cout << "samples = " << sample_file << std::endl;
      std::cout << "append_samples = " << append_samples << std::endl;

      std::cout << "seed = " << random_seed 
                << " (" << (command.has_key("seed") 
                            ? "user specified"
                            : "randomly generated") << ")"
                << std::endl;
      std::cout << "chain_id=" << chain_id
                << " (" << (command.has_key("seed")
                            ? "user specified"
                            : "default") << ")"
                << std::endl;

      std::cout << "iter = " << num_iterations << std::endl;
      std::cout << "warmup = " << num_warmup << std::endl;
      std::cout << "thin = " << num_thin << std::endl;

      std::cout << "leapfrog steps = " << leapfrog_steps << std::endl;
      std::cout << "epsilon = " << epsilon << std::endl;;
      std::cout << "epsilon_pm = " << epsilon_pm << std::endl;;
      std::cout << "epsilon_adapt_off = " << epsilon_adapt_off << std::endl;;
      std::cout << "delta = " << delta << std::endl;
      std::cout << "gamma = " << gamma << std::endl;


      if (command.has_flag("test_grad")) {
        std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;
        model.test_gradients(params_r,params_i);
        return 0;
      }

      std::fstream sample_stream(sample_file.c_str(), 
                                 samples_append_mode);
      
      write_comment(sample_stream,
                    "STAN output");


      write_comment(sample_stream,
                    
      
      if (!append_samples) {
        sample_stream << "lp__,"; // log probability first
        model.write_csv_header(sample_stream);
      }

      if (leapfrog_steps < 0) {
        stan::mcmc::nuts<rng_t> nuts_sampler(model, 
                                             epsilon, epsilon_pm, epsilon_adapt,
                                             delta, gamma, 
                                             base_rng);
        sample_from(nuts_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,params_r,params_i,
                    model);
      } else {
        stan::mcmc::adaptive_hmc<rng_t> hmc_sampler(model,
                                                    leapfrog_steps,
                                                    epsilon, epsilon_pm, epsilon_adapt,
                                                    delta, gamma,
                                                    base_rng);
        sample_from(hmc_sampler,epsilon_adapt,refresh,
                    num_iterations,num_warmup,num_thin,
                    sample_stream,params_r,params_i,
                    model);
      }
      
      sample_stream.close();
      std::cout << std::endl << std::endl;
      return 0;
    }

  } // namespace prob


} // namespace stan

#endif
