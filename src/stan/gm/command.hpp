#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <cmath>
#include <iostream>
#include <fstream>
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

    template <typename T_model>
    void nuts_command(int argc, const char* argv[]) {

      stan::io::cmd_line command(argc,argv);

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
      
      std::cout << "NUTS" << std::endl;
      std::cout << "sample_file=" << sample_file << std::endl;
      std::cout << "num_iterations=" << num_iterations << std::endl;
      std::cout << "num_burnin=" << num_burnin << std::endl;
      std::cout << "num_thin=" << num_thin << std::endl;
      std::cout << "delta=" << delta << std::endl;

      // Choose default value for delta (0.5)
      stan::mcmc::nuts sampler(model, 0.5, -1);
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

      } else {
	// FIXME:  default to random inits rather than 0s
	params_i = std::vector<int>(model.num_params_i(),0.0);
	params_r = std::vector<double>(model.num_params_r(),0.0);
      }
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
