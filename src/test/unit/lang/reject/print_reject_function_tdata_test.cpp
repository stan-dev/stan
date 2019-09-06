#include <stdexcept>
#include <test/test-models/good/lang/print_reject_function_tdata.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>

// test print statement captured before reject
// in function called from transformed data block
TEST(StanCommon, printRejectFunctionTData) { 
  typedef 
    print_reject_function_tdata_model_namespace
    ::print_reject_function_tdata_model
    model_t;
  print_reject_test<model_t,std::domain_error>("quitting time");
}
