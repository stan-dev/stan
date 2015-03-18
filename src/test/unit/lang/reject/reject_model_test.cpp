#include <stdexcept>
#include <test/test-models/good/lang/reject_model.hpp>
#include <test/unit/lang/reject/reject_helper.hpp>
#include <gtest/gtest.h>


// test reject() statement throws from model block
TEST(StanCommon, rejectModel) {
  typedef 
    reject_model_model_namespace::reject_model_model
    model_t;
  reject_test<model_t,std::domain_error>("user-specified rejection", "13");
}
