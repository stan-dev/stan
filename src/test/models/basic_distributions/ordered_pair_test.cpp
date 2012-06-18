#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicDistributions_OrderedPair : 
  public Model_Test_Fixture<Models_BasicDistributions_OrderedPair> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("ordered_pair");
    return model_path;
  }
  static bool has_data() {
    return false;
  }
};

TEST_F(Models_BasicDistributions_OrderedPair,RunModel) {
  run_model();
}
