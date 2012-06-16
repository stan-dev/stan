#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol1_Equiv : 
  public Model_Test_Fixture<Models_BugsExamples_Vol1_Equiv,
                                       true> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol1");
    model_path.push_back("equiv");
    model_path.push_back("equiv");
    return model_path;
  }

};

TEST_F(Models_BugsExamples_Vol1_Equiv,RunModel) {
  run_model();
}
