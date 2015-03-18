#include <stdexcept>
#include <test/test-models/good/lang/reject_generated_quantities.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>

// test reject() statement throws from generated quantities
TEST(StanCommon, rejectGeneratedQuantities) {
  typedef 
    reject_generated_quantities_model_namespace
    ::reject_generated_quantities_model
    model_t;
  reject_test<model_t,std::domain_error>("user-specified rejection", "9");
}
