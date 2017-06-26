#include <stdexcept>
#include <test/test-models/good/lang/print_reject_function_tparams.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>

// test print statement captured before reject
// in function called from transformed data block
TEST(StanCommon, printRejectFunctionTparams) { 
  typedef 
    print_reject_function_tparams_model_namespace
    ::print_reject_function_tparams_model
    model_t;
  print_reject_test<model_t,std::domain_error>("quitting time");
}
