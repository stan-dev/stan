#include <gtest/gtest.h>
#include <stan/common/write_iteration.hpp>

#include <stdexcept>
#include <sstream>
#include <test/test-models/good/gm/reject_generated_quantities.cpp>

/* tests that stan program throws exception in generated quantities block
   this is compiled into cpp model object's method write_array
*/

TEST(StanCommon, reject_generated_quantities) {
  std::string error_msg = "user-specified rejection";

  std::fstream empty_data_stream(std::string("").c_str());
  stan::io::dump empty_data_context(empty_data_stream);
  empty_data_stream.close();
  std::stringstream model_output;
  model_output.str("");

  // instantiate model
  reject_generated_quantities_model_namespace::reject_generated_quantities_model* model 
       = new reject_generated_quantities_model_namespace::reject_generated_quantities_model(empty_data_context, &model_output);

  // instantiate args to log_prob function
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model->num_params_r());
  std::vector<double> cont_vector(cont_params.size());
  for (int i = 0; i < cont_params.size(); ++i)
    cont_vector.at(i) = cont_params(i);
  std::vector<int> disc_vector;
  double lp(0);

  boost::ecuyer1988 base_rng;
  base_rng.seed(123456);
  
  lp = model->log_prob<false, false>(cont_vector, disc_vector, &std::cout);
  try {
    stan::common::write_iteration(model_output, *model, base_rng,
                    lp, cont_vector, disc_vector);
  } catch (const std::domain_error& e) {
    if (std::string(e.what()).find(error_msg) == std::string::npos) {
      FAIL() << std::endl << "---------------------------------" << std::endl
             << "--- EXPECTED: error_msg=" << error_msg << std::endl
             << "--- FOUND: e.what()=" << e.what() << std::endl
             << "---------------------------------" << std::endl
             << std::endl;
    }
    return;
  }
  FAIL() << "model failed to do reject" << std::endl;
}

