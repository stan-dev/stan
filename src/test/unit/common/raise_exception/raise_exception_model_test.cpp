#include <gtest/gtest.h>
#include <stan/model/util.hpp>
#include <stdexcept>
#include <sstream>
#include <test/test-models/no-main/gm/raise_exception_model.cpp>

/* to test that stan program throws exception in model block:
   setup: instantiate model, sampler, 
   test:  calculate gradient
   (teardown: delete model, sampler)
*/

typedef raise_exception_model_model_namespace::raise_exception_model_model Model;
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


TEST_F(StanCommon, raise_exception_model) {
  std::string error_msg = "user-specified exception";
  double init_log_prob;
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model->num_params_r());
  Eigen::VectorXd init_grad = Eigen::VectorXd::Zero(model->num_params_r());
  try {
    stan::model::gradient((*model), cont_params, init_log_prob, init_grad, &std::cout);
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

