#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_Transforms_Simplex : 
  public Model_Test_Fixture<Models_Transforms_Simplex,
                                       false> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("transforms");
    model_path.push_back("simplex");
    return model_path;
  }
};

TEST_F(Models_Transforms_Simplex,RunModel) {
  run_model();
}
