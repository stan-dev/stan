#ifndef __STAN__GM__COMMAND_HPP__
#define __STAN__GM__COMMAND_HPP__

#include <cmath>

namespace stan {

  namespace gm {

    class prob_grad_ad_cmd : public prob_grad_ad {
    public:
      prob_grad_ad_cmd() : prob_grad_ad(0U) { }

      void run(int argc,
	       const char* argv[],
	       prob_grad_ad_cmd* (*factory)(const stan::io::dump&)) {

	std::string data_file = "data.dump";
	cmd_line.val("data_file",data_file);

	std::fstream data_file_stream(data_file);
	stan::io::dump data(data_file);
	prob_grad_ad_cmd* model = factory(dump);
	data_file_stream.close();

	std::string init_file = "init.dump";
	cmd_line.val("init_file",init_file);

	std::string sample_file = "samples.csv";
	sample_file.val("sample_file",sample_file);
      
	unsigned int num_iterations = 2000;
	cmd_line.val("num_iterations",num_iterations);
      
	unsigned int num_burnin = num_iterations / 2;
	cmd_line.val("num_burnin",num_burnin);
      
	unsigned int num_thin = std::max(1, (num_iter - num_burnin) / 1000);
	cmd_line.val("num_thin",num_thin);
      
	double step_size = 0.01;
	cmd_line.val("step_size",step_size);
      
	unsigned int num_steps = 50;
	cmd_line.val("num_steps",num_steps);
      
	stan::mcmc::hmc sampler(model,step_size,num_steps);

	std::vector<double> params_r;
	std::vector<int> params_i;
	for (unsigned int m = 0; m < num_samples; ++m) {
	stan::mcmc::sample sample = sampler.next();
	  if (m < num_burnin)
	    return; // still burning in
	  if (m || (m - burn_in) % thin != 0)
	    return;

	  sample = sampler.next(sample);
	  sample.params_r(params_r);
	  sample.params_i(params_i);
	  model->write_csv(sample_file,params_r,params_i);
	}
      }

      virtual void write_csv(const stan::mcmc::sample& sample
			     std::ostream& out) = 0;

      // in generated code
      // int main(int argc, const char* argv[]) {
      //     run(argc,argv,&factory__);
      // }

      // or simplify run to remove function pointer and instead expand
      // int main(int argc__, const char* argv__[]) {
      //    stan::io::cmd_line cmd(argc__,argv__);
      //    std::string data_file_path__ = cmd.val("data_file");
      //    std::fstream data_file__(data_file_path__.c_str(),std::fstream::in);
      //    stan::io::dump dump__(data_file__);
      //    test_model_namespace::test_model model__(dump__);
      //    data_file__.close();
      // }
}


  }

#endif
