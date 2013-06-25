#include <test/mcmc/mock_hmc.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>

#include <boost/random/additive_combine.hpp>

#include <gtest/gtest.h>
#include <test/models/utility.hpp>

typedef boost::ecuyer1988 rng_t;


namespace stan {
  
  namespace mcmc {
    
    class mock_hmc: public base_hmc<mock_model,
                                    ps_point,
                                    mock_hamiltonian,
                                    mock_integrator,
                                    rng_t> {
      
    public:
      
      mock_hmc(mock_model& m, rng_t& rng, std::ostream* o, std::ostream* e) : 
        base_hmc<mock_model,ps_point,mock_hamiltonian,mock_integrator,rng_t>
        (m, rng, o, e)
      { this->_name = "Mock HMC"; }
      
      
      sample transition(sample& init_sample) {
        this->seed(init_sample.cont_params(), init_sample.disc_params());
        return sample(this->_z.q, this->_z.r, - this->_hamiltonian.V(this->_z), 0);
      }
      
      void write_sampler_param_names(std::ostream& o) {};
      
      void write_sampler_params(std::ostream& o) {};
      
      void get_sampler_param_names(std::vector<std::string>& names) {};
      
      void get_sampler_params(std::vector<double>& values) {};
      
    };
    
  }
  
}

TEST(McmcBaseHMC, point_construction) {
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_hmc sampler(model, base_rng, &std::cout, &std::cerr);

  EXPECT_EQ(static_cast<size_t>(q.size()), sampler.z().q.size());
  EXPECT_EQ(static_cast<int>(q.size()), sampler.z().g.size());
}

TEST(McmcBaseHMC, seed) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r;
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_hmc sampler(model, base_rng, &std::cout, &std::cerr);
  
  std::vector<double> q_seed(q.size(), -1.0);
  
  sampler.seed(q_seed, r);
  
  for (std::vector<double>::size_type i = 0; i < q.size(); ++i)
    EXPECT_EQ(q_seed.at(i), sampler.z().q.at(i));
  
}

TEST(McmcBaseHMC, set_nominal_stepsize) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_hmc sampler(model, base_rng, &std::cout, &std::cerr);
  
  double old_epsilon = 1.0;
  sampler.set_nominal_stepsize(old_epsilon);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  
  sampler.set_nominal_stepsize(-0.1);
  EXPECT_EQ(old_epsilon, sampler.get_nominal_stepsize());
  
}

TEST(McmcBaseHMC, set_stepsize_jitter) {
  
  rng_t base_rng(0);
  
  std::vector<double> q(5, 1.0);
  std::vector<int> r(2, 2);
  
  stan::mcmc::mock_model model(q.size());
  
  stan::mcmc::mock_hmc sampler(model, base_rng, &std::cout, &std::cerr);
  
  double old_jitter = 0.1;
  sampler.set_stepsize_jitter(old_jitter);
  EXPECT_EQ(old_jitter, sampler.get_stepsize_jitter());
  
  sampler.set_nominal_stepsize(-0.1);
  EXPECT_EQ(old_jitter, sampler.get_stepsize_jitter());
  
}

TEST(McmcBaseHMC, init_stepsize) {
  
  std::vector<std::string> model_path;
  model_path.push_back("src");
  model_path.push_back("test");
  model_path.push_back("mcmc");
  model_path.push_back("models");
  model_path.push_back("improper");
  
  std::string command = convert_model_path(model_path);
  std::string command_output;
  long time;
  
  try {
    command_output = run_command(command, time);
  } catch(...) {
    ADD_FAILURE() << "Failed running command: " << command;
  }
  
  command_output.erase(command_output.length() - 1);
  
  EXPECT_EQ("Posterior is improper. Please check your model.",
            command_output.substr(command_output.rfind('\n')).erase(0, 1) );
  
}
