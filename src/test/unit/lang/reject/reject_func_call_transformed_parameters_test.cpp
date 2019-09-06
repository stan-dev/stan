#include <stdexcept>
#include <test/test-models/good/lang/reject_func_call_transformed_parameters.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>

// test reject() statement throws exception from transformed
// parameters through a function call
TEST(StanCommon, rejectFuncCallTransformedParameters) {
  typedef 
    reject_func_call_transformed_parameters_model_namespace
    ::reject_func_call_transformed_parameters_model
    model_t;
  reject_test<model_t,std::domain_error>("user-specified rejection", "11", "3");
}
