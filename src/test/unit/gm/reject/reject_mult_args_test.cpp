#include <gtest/gtest.h>
#include <stdexcept>
#include <sstream>
#include <test/test-models/good/gm/reject_mult_args.cpp>

/* tests that stan program throws exception in model block
   this block gets compiled into .cpp model object's log_prob method
*/

TEST(StanCommon, reject_mult_args) {
  std::string error_msg = "user-specified rejection";

  std::fstream empty_data_stream(std::string("").c_str());
  stan::io::dump empty_data_context(empty_data_stream);
  empty_data_stream.close();
  std::stringstream model_output;
  model_output.str("");

  // instantiate model
  reject_mult_args_model_namespace::reject_mult_args_model* model 
       = new reject_mult_args_model_namespace::reject_mult_args_model(empty_data_context, &model_output);

  // instantiate args to log_prob function
  Eigen::VectorXd cont_params = Eigen::VectorXd::Zero(model->num_params_r());
  std::vector<double> cont_vector(cont_params.size());
  for (int i = 0; i < cont_params.size(); ++i)
    cont_vector.at(i) = cont_params(i);
  std::vector<int> disc_vector;
  double lp(0);

  // call model's log_prob function, check that exception is thrown
  try {
    lp = model->log_prob<false, false>(cont_vector, disc_vector, &std::cout);
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

