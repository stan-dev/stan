#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicDistributions_Wishart2x2 :
  public Model_Test_Fixture<Models_BasicDistributions_Wishart2x2,
                                       false> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("wishart2x2");
    return model_path;
  }
};

TEST_F(Models_BasicDistributions_Wishart2x2,RunModel) {
  run_model();
}
