#ifndef TEST_UNIT_LANG_REJECT_REJECT_HELPER_HPP
#define TEST_UNIT_LANG_REJECT_REJECT_HELPER_HPP

#include <gtest/gtest.h>
#include <stdexcept>
#include <sstream>
#include <stan/services/io/write_iteration.hpp>

void expect_substring(const std::string& msg,
                      const std::string& expected_substring) {
  if (msg.find(expected_substring) == std::string::npos)
    FAIL() << "expected to find substring=" << expected_substring
           << " in string=" << msg 
           << std::endl;
}

template <class M>
void reject_write_iteration_test(const std::string& expected_msg1 = "",
                                 const std::string& expected_msg2 = "",
                                 const std::string& expected_msg3 = "") {

  std::fstream empty_data_stream("");
  stan::io::dump empty_data_context(empty_data_stream);
  empty_data_stream.close();

  std::stringstream model_output;

  M model(empty_data_context, &model_output);

  std::vector<double> cont_vector(model.num_params_r(), 0.0);
  std::vector<int> disc_vector;

  boost::ecuyer1988 base_rng;
  base_rng.seed(123456);

  double lp;
  lp = model.template log_prob<false,false>(cont_vector, disc_vector, &std::cout);
  try {
    using stan::services::io::write_iteration;
    write_iteration(model_output, model, base_rng, lp, cont_vector, disc_vector);
  } catch (const std::domain_error& e) {
    expect_substring(e.what(), expected_msg1);
    expect_substring(e.what(), expected_msg2);
    expect_substring(e.what(), expected_msg3);
    return;
  }
  FAIL() << "model failed to reject" << std::endl;
}

#endif 
