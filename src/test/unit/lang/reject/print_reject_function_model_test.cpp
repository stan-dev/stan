#include <stdexcept>
#include <test/test-models/good/lang/print_reject_function_model.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>


// test print statement output captured before reject
// in model block through a function call
TEST(StanCommon, printRejectFuncModel) {
  typedef 
    print_reject_function_model_model_namespace
    ::print_reject_function_model_model
    model_t;
  print_reject_test<model_t,std::domain_error>("quitting time");
}
