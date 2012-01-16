#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
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
      
      unsigned int num_iterations = 2000;
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

    void pad_help_option(std::string option) {
      std::cout << "  " << option;
      for (unsigned int i = option.size(); i < 25; ++i)
        std::cout << ' ';
    }

    void print_nuts_help(std::string cmd) {
      std::cout << "USAGE:  " << cmd << " [options]" << std::endl;
      std::cout << "OPTIONS:" << std::endl;

      pad_help_option("--help");
      std::cout << "Display this information" << std::endl;

      pad_help_option("--random_seed=<int>");
      std::cout << "Set random number generation seed" << std::endl;

      pad_help_option("--inits=<path to file>");
      std::cout << "Use initial values from specialized file"
                << std::endl;
      pad_help_option("");
      std::cout << "    (default is random initialization)" << std::endl;

      pad_help_option("--iter=<positive int>");
      std::cout << "Total number of iterations, including burn in"
                << std::endl;
      pad_help_option("");
      std::cout << "    (default = 2000)" << std::endl;
      
      pad_help_option("--burn_in=<positive int>");
      std::cout << "Discard the specified number of initial samples"
                << std::endl;
      pad_help_option("");
      std::cout << "    (default = iter / 2)" << std::endl;

      pad_help_option("--thin=<positive int>");
      std::cout << "Period between saved samples after burn in" << std::endl;
      pad_help_option("");
      std::cout << "    (default = max(1, floor(iter - burn_in) / 1000))"
                << std::endl;
      
      pad_help_option("--samples=<path to file>");
      std::cout << "File into which samples are written." << std::endl;
      pad_help_option("");
      std::cout << "    (default = samples.csv)" << std::endl;

      pad_help_option("--append_samples");
      std::cout << "Append samples to existing samples file if it exists"
                << std::endl;
      pad_help_option("");
      std::cout << "    (default erases existing samples file before writing)"
                << std::endl;

      pad_help_option("--progress_refresh");
      std::cout << "Period between samples producing progress output" 
                << std::endl;
      pad_help_option("");
      std::cout << "    (default = max(1,iter/200))" << std::endl;
    }

    template <typename T_model>
    void nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);

      if (command.has_flag("help")) {
        print_nuts_help(argv[0]);
        return;
      }

      std::string data_path;
      command.val("data_file",data_path);
      std::fstream data_stream(data_path.c_str(),std::fstream::in);
      stan::io::dump data_var_context(data_stream);
      data_stream.close();

      T_model model(data_var_context);

      std::string sample_file = "samples.csv";
      command.val("sample_file",sample_file);
      std::fstream sample_file_stream(sample_file.c_str(), std::fstream::out);
      
      unsigned int num_iterations = 2000;
      command.val("num_iterations",num_iterations);
      
      unsigned int num_burnin = num_iterations / 2;
      command.val("num_burnin",num_burnin);
      
      unsigned int calculated_thin = (num_iterations - num_burnin) / 1000U;
      unsigned int num_thin = (calculated_thin > 1) ? calculated_thin : 1U;
      command.val("num_thin",num_thin);

      double delta = 0.5;
      command.val("delta", delta);

      int random_seed(0);
      if (command.has_key("random_seed"))
        command.val("random_seed",random_seed);
      else
        random_seed = std::time(0);

      boost::mt19937 base_rng(random_seed);
      boost::random::uniform_real_distribution<double> init_range_distribution(-2.0,2.0);
      boost::variate_generator<boost::mt19937&, 
                               boost::random::uniform_real_distribution<double> >
        init_rng(base_rng,init_range_distribution);


      
      std::cout << "NUTS" << std::endl;
      std::cout << "sample_file=" << sample_file << std::endl;
      std::cout << "num_iterations=" << num_iterations << std::endl;
      std::cout << "num_burnin=" << num_burnin << std::endl;
      std::cout << "num_thin=" << num_thin << std::endl;
      std::cout << "delta=" << delta << std::endl;
      std::cout << "random seed=" << random_seed 
                << " (" << (command.has_key("random_seed") ? "user specified" : "randomly generated") << ")"
                << std::endl;

      // Choose default value for delta (0.5)
      stan::mcmc::nuts<> sampler(model, 0.5, -1, base_rng);
      sampler.adapt_on();

      std::vector<int> params_i;
      std::vector<double> params_r;
      if (command.has_key("init")) {
        std::string init_path;
        command.val("init",init_path);
        std::cout << "init file=" << init_path << std::endl;
        
        std::fstream init_stream(init_path.c_str(),std::fstream::in);
        stan::io::dump init_var_context(init_stream);
        init_stream.close();
        model.transform_inits(init_var_context,params_i,params_r);

      } else if (command.has_key("zero_init")) {
        // FIXME:  default to random inits rather than 0s
        params_i = std::vector<int>(model.num_params_i(),0);
        params_r = std::vector<double>(model.num_params_r(),0.0);
      } else {
        params_i = std::vector<int>(model.num_params_i(),0);
        params_r = std::vector<double>(model.num_params_r());
        for (unsigned int i = 0; i < params_r.size(); ++i)
          params_r[i] = init_rng();
      }
      // FIXME: put back
      model.write_csv_header(sample_file_stream);
      for (unsigned int m = 0; m < num_iterations; ++m) {
        std::cout << "iteration=" << (m + 1);
        if (m < num_burnin) {
          std::cout << " burning in" << std::endl;
          sampler.next();
          continue;
        } else {
          sampler.adapt_off();
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

  }


}

#endif
