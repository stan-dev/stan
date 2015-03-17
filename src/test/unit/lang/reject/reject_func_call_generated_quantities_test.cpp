#include <gtest/gtest.h>
#include <test/test-models/good/lang/reject_func_call_generated_quantities.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>

// test exception w. source locations in generated quantities block via write_array
TEST(StanCommon, rejectFuncCallGeneratedQuantities) {
  typedef 
    reject_func_call_generated_quantities_model_namespace
    ::reject_func_call_generated_quantities_model 
    model_t;
  reject_write_iteration_test<model_t>("user-specified rejection","3","13");
}
