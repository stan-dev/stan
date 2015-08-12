#include <stdexcept>
#include <test/test-models/good/lang/reject_func_call_transformed_data.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>

// test reject() statement throws exception from transformed data
// through functions
TEST(StanCommon, rejectTransformedParameters) { 
  typedef 
    reject_func_call_transformed_data_model_namespace
    ::reject_func_call_transformed_data_model
    model_t;
  reject_test<model_t,std::domain_error>("user-specified rejection", "3", "8");
}
