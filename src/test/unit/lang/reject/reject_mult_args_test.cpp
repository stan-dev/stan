#include <stdexcept>
#include <test/test-models/good/lang/reject_mult_args.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>

// test throw in model block through function call
TEST(StanCommon, rejectMultArgs) {
  typedef 
    reject_mult_args_model_namespace::reject_mult_args_model
    model_t;
  reject_test<model_t,std::domain_error>("user-specified rejection", "29", "8");
}
