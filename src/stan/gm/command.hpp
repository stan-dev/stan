#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <cmath>
#include <cstddef>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <boost/random/additive_combine.hpp> // L'Ecuyer RNG
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <stan/io/cmd_line.hpp>
#include <stan/io/dump.hpp>
#include <stan/mcmc/hmc.hpp>
#include <stan/mcmc/nuts.hpp>
#include <stan/mcmc/prob_grad_ad.hpp>
#include <stan/mcmc/prob_grad.hpp>
#include <stan/mcmc/sampler.hpp>

namespace stan {

  namespace gm {

    void hmc_command(const stan::io::cmd_line& command,
                     stan::mcmc::prob_grad& model) {

      std::string sample_file = "samples.csv";
      command.val("sample_file",sample_file);
      std::fstream sample_file_stream(sample_file.c_str(), std::fstream::out);
      
      unsigned int num_iterations = 2000U;
      command.val("num_iterations",num_iterations);
      
      unsigned int num_burnin = num_iterations / 2;
      command.val("num_burnin",num_burnin);
      
      unsigned int calculated_thin = (num_iterations - num_burnin) / 1000U;
      unsigned int num_thin = (calculated_thin > 1) ? calculated_thin : 1U;
      command.val("num_thin",num_thin);
      
      double step_size = 0.01;
      command.val("step_size",step_size);
      
      unsigned int num_steps = 50;
      command.val("num_steps",num_steps);
      
      std::cout << "HMC" << std::endl;
      std::cout << "sample_file=" << sample_file << std::endl;
      std::cout << "num_iterations=" << num_iterations << std::endl;
      std::cout << "num_burnin=" << num_burnin << std::endl;
      std::cout << "num_thin=" << num_thin << std::endl;
      std::cout << "step_size=" << step_size << std::endl;
      std::cout << "num_steps=" << num_steps << std::endl;

      stan::mcmc::hmc sampler(model,step_size,num_steps);

      std::vector<double> params_r;
      std::vector<int> params_i;
      for (unsigned int m = 0; m < num_iterations; ++m) {
        std::cout << "iteration=" << (m + 1);
        if (m < num_burnin) {
          std::cout << " burning in" << std::endl;
          sampler.next();
          continue;
        }
        if (((m - num_burnin) % num_thin) != 0) {
          std::cout << " thinning" << std::endl;
          sampler.next();
          continue;
        } 
        std::cout << " saving" << std::endl;
        stan::mcmc::sample sample = sampler.next();
        sample.params_r(params_r);
        sample.params_i(params_i);
        model.write_csv(params_r,params_i,sample_file_stream);
      }
      sample_file_stream.close();
    }

    // void pad_help_option(std::string option) {
    //   std::cout << "  " << option;
    //   for (unsigned int i = option.size(); i < 20; ++i)
    //     std::cout << ' ';
    // }

    void print_nuts_help(std::string cmd) {
      using stan::io::pad_help_option;

      std::cout << std::endl;
      std::cout << "Compiled Stan Graphical Model Command" << std::endl;
      std::cout << std::endl;

      std::cout << "USAGE:  " << cmd << " [options]" << std::endl;
      std::cout << std::endl;

      std::cout << "OPTIONS:" << std::endl;
      std::cout << std::endl;

      pad_help_option("--help");
      std::cout << "Display this information" << std::endl;
      std::cout << std::endl;

      pad_help_option("--data=<file>");
      std::cout << "Read data from specified dump-format file" << std::endl;
      pad_help_option();
      std::cout << "  (required if model declares data)" << std::endl;
      std::cout << std::endl;

      pad_help_option("--seed=<int>");
      std::cout << "Set random number generation seed" << std::endl;
      std::cout << std::endl;

      pad_help_option("--chain_id=<int>");
      std::cout << "Set chain identifier" << std::endl;
      pad_help_option();
      std::cout << "  (default = 1)" << std::endl;

      pad_help_option("--init=<file>");
      std::cout << "Use initial values from specified file or zero values if <file>=0"
                << std::endl;
      pad_help_option();
      std::cout << "    (default is random initialization)" << std::endl;
      std::cout << std::endl;

      pad_help_option("--iter=<+int>");
      std::cout << "Total number of iterations, including burn in"
                << std::endl;
      pad_help_option();
      std::cout << "    (default = 2000)" << std::endl;
      std::cout << std::endl;
      
      pad_help_option("--burnin=<+int>");
      std::cout << "Discard the specified number of initial samples"
                << std::endl;
      pad_help_option();
      std::cout << "    (default = iter / 2)" << std::endl;
      std::cout << std::endl;

      pad_help_option("--thin=<+int>");
      std::cout << "Period between saved samples after burn in" << std::endl;
      pad_help_option();
      std::cout << "    (default = max(1, floor(iter - burnin) / 1000))"
                << std::endl;
      std::cout << std::endl;
      
      pad_help_option("--delta=<+float>");
      std::cout << "Initial parameter for NUTS step-size tuning." << std::endl;
      pad_help_option();
      std::cout << "    (default = 0.5)" << std::endl;
      std::cout << std::endl;

      pad_help_option("--samples=<file>");
      std::cout << "File into which samples are written." << std::endl;
      pad_help_option();
      std::cout << "    (default = samples.csv)" << std::endl;
      std::cout << std::endl;

      pad_help_option("--append_samples");
      std::cout << "Append samples to existing samples file if it exists"
                << std::endl;
      pad_help_option();
      std::cout << "    (default erases existing samples file before writing)"
                << std::endl;
      std::cout << std::endl;

      pad_help_option("--refresh=<+int>");
      std::cout << "Period between samples producing progress output" 
                << std::endl;
      pad_help_option();
      std::cout << "    (default = max(1,iter/200))" << std::endl;
      std::cout << std::endl;

      pad_help_option("--test_grad");
      std::cout << "Test gradient calculations using finite differences"
                << std::endl;
      
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
    int nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);

      if (command.has_flag("help")) {
        print_nuts_help(argv[0]);
        return 0;
      }

      std::string data_path;
      command.val("data",data_path);
      std::fstream data_stream(data_path.c_str(),std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      T_model model(data_var_context);

      std::string sample_file = "samples.csv";
      command.val("samples",sample_file);
      
      unsigned int num_iterations = 2000U;
      command.val("iter",num_iterations);
      
      unsigned int num_burnin = num_iterations / 2;
      command.val("burnin",num_burnin);
      
      unsigned int calculated_thin = (num_iterations - num_burnin) / 1000U;
      unsigned int num_thin = (calculated_thin > 1) ? calculated_thin : 1U;
      command.val("thin",num_thin);

      double delta = 0.5;
      command.val("delta", delta);

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
      
      // FASTER, but no parallel guarantees
      // typedef boost::mt19937 rng_t;
      // rng_t base_rng(static_cast<unsigned int>(random_seed + chain_id - 1);

      typedef boost::ecuyer1988 rng_t;
      rng_t base_rng(random_seed);
      // (2**50 = 1T samples, 1000 chains)
      static long unsigned int DISCARD_STRIDE = (1 << 50);
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

      std::cout << "NUTS" << std::endl;
      std::cout << "sample_file=" << sample_file << std::endl;
      std::cout << "init=" << init_val << std::endl;
      std::cout << "num_iterations=" << num_iterations << std::endl;
      std::cout << "num_burnin=" << num_burnin << std::endl;
      std::cout << "num_thin=" << num_thin << std::endl;
      std::cout << "delta=" << delta << std::endl;
      std::cout << "random seed=" << random_seed 
                << " (" << (command.has_key("seed") 
                            ? "user specified"
                            : "randomly generated") << ")"
                << std::endl;
      std::cout << "chain id=" << chain_id
                << " (" << (command.has_key("seed")
                            ? "user specified"
                            : "randomly generated") << ")"
                << std::endl;

      if (command.has_flag("test_grad")) {
        std::cout << std::endl << "TEST GRADIENT MODE" << std::endl;
        model.test_gradients(params_r,params_i);
        return 0;
      }

      std::fstream sample_file_stream(sample_file.c_str(), std::fstream::out);
      model.write_csv_header(sample_file_stream);
      int it_print_width = std::ceil(std::log10(num_iterations));
      std::cout << std::endl;

      stan::mcmc::nuts<rng_t> sampler(model, delta, -1, base_rng);
      sampler.adapt_on();
      for (unsigned int m = 0; m < num_iterations; ++m) {
        if (do_print(m,refresh)) {
          std::cout << "\rIteration: ";
          std::cout << std::setw(it_print_width) << (m + 1)
                    << " / " << num_iterations;
          std::cout << " [" << std::setw(3) 
                    << static_cast<int>((100.0 * (m + 1))/num_iterations)
                    << "%] ";
          std::cout << ((m < num_burnin) ? " (Adapting)" : " (Sampling)");
          std::cout.flush();
        }
        if (m < num_burnin) {
          sampler.next(); // discard
        } else {
          sampler.adapt_off();
          if (((m - num_burnin) % num_thin) != 0) {
            sampler.next();
            continue;
          } else {
            stan::mcmc::sample sample = sampler.next();
            sample.params_r(params_r);
            sample.params_i(params_i);
            model.write_csv(params_r,params_i,sample_file_stream);
          }
        }
      }
      sample_file_stream.close();
      std::cout << std::endl << std::endl;
      return 0;
    }

  } // namespace prob


} // namespace stan

#endif
