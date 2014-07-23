#include <gtest/gtest.h>
#include <stan/common/command.hpp>
#include <stdexcept>
#include <sstream>
#include <test/test-models/no-main/gm/raise_ex_func_call_model.cpp>

/* to test that stan program throws exception in model block:
   setup: instantiate model
   test:  call model's log_prob function
   (teardown: delete model)
*/

typedef raise_ex_func_call_model_model_namespace::raise_ex_func_call_model_model Model;
typedef boost::ecuyer1988 rng_t;
typedef stan::mcmc::adapt_unit_e_nuts<Model, rng_t> sampler;

class StanCommon : public testing::Test {
public:
  void SetUp() {
    std::fstream empty_data_stream(std::string("").c_str());
    stan::io::dump empty_data_context(empty_data_stream);
    empty_data_stream.close();
    
    model_output.str("");
    model = new Model(empty_data_context, &model_output);
    base_rng.seed(123456);

    output.str("");
    error.str("");
    sampler_ptr = new sampler((*model), base_rng, &output, &error);
    sampler_ptr->set_nominal_stepsize(1);
    sampler_ptr->set_stepsize_jitter(0);
    sampler_ptr->set_max_depth(10);
  }

  void TearDown() {
    delete sampler_ptr;
    delete model;
  }

  rng_t base_rng;
  Model* model;
  sampler* sampler_ptr;

  std::stringstream model_output, output, error;
};


TEST_F(StanCommon, raise_ex_func_call_model) {
  std::string error_msg = "user-specified exception";
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model->num_params_r());
  std::vector<double> cont_vector(cont_params.size());
  for (int i = 0; i < cont_params.size(); ++i)
    cont_vector.at(i) = cont_params(i);
  std::vector<int> disc_vector;
  double lp(0);
  try {
    lp = model->log_prob<false, false>(cont_vector, disc_vector, &std::cout);
  } catch (const std::domain_error& e) {
    if (std::string(e.what()).find(error_msg) == std::string::npos) {
      FAIL() << std::endl << "*********************************" << std::endl
             << "*** EXPECTED: error_msg=" << error_msg << std::endl
             << "*** FOUND: e.what()=" << e.what() << std::endl
             << "*********************************" << std::endl
             << std::endl;
    }
    return;
  }
  FAIL() << "model failed to raise exception" << std::endl;

}

