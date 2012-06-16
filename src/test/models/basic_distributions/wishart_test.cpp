#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicDistributions_Wishart :
  public Model_Test_Fixture<Models_BasicDistributions_Wishart,
                                       false> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("wishart");
    return model_path;
  }
};

TEST_F(Models_BasicDistributions_Wishart,RunModel) {
  run_model();
}
