#include <stdexcept>
#include <test/test-models/good/lang/reject_func_call_model.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>


// tests reject() statement throwing from model block through
// a function call
TEST(StanCommon, reject_func_call_transformed_parameters) {
  typedef 
    reject_func_call_model_model_namespace
    ::reject_func_call_model_model
    model_t;
  reject_test<model_t,std::domain_error>("user-specified rejection", "10", "3");
}
