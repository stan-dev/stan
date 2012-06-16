#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BugsExamples_Vol2_Endo3 : 
  public Model_Test_Fixture<Models_BugsExamples_Vol2_Endo3,
                                       true> {
protected:
  virtual void SetUp() {}
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("bugs_examples");
    model_path.push_back("vol2");
    model_path.push_back("endo");
    model_path.push_back("endo3");
    return model_path;
  }

};

TEST_F(Models_BugsExamples_Vol2_Endo3,RunModel) {
  run_model();
}
