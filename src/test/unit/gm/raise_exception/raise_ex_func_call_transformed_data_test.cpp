#include <gtest/gtest.h>
#include <stdexcept>
#include <sstream>
#include <test/test-models/no-main/gm/raise_ex_func_call_transformed_data.cpp>

/* tests that stan program throws exception in transformed data block
   this block is part of the generated cpp object's constructor
*/


TEST(StanCommon, raise_ex_func_call_transformed_data) {
  std::string error_msg = "user-specified exception";

  std::fstream empty_data_stream(std::string("").c_str());
  stan::io::dump empty_data_context(empty_data_stream);
  empty_data_stream.close();
  std::stringstream model_output;
  model_output.str("");

  // instantiate model
  try {
     raise_ex_func_call_transformed_data_model_namespace::raise_ex_func_call_transformed_data_model* model 
       = new raise_ex_func_call_transformed_data_model_namespace::raise_ex_func_call_transformed_data_model(empty_data_context, &model_output);
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

