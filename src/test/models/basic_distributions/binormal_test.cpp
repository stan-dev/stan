#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>
#include <boost/math/distributions/students_t.hpp>

class Models_BasicDistributions_Binormal : 
  public Model_Test_Fixture<Models_BasicDistributions_Binormal> {
protected:
  double expected_y1;
  double expected_y2;
  
  virtual void SetUp() {
    expected_y1 = 0.0;
    expected_y2 = 0.0;
  }
  
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("binormal");
    return model_path;
  }
    
  static bool has_data() {
    return false;
  }
};

TEST_F(Models_BasicDistributions_Binormal,RunModel) {
  run_model();
}
TEST_F(Models_BasicDistributions_Binormal, y1) {
  using boost::math::students_t;
  using boost::math::quantile;
  
  size_t index;
  std::vector<size_t> idxs;
  idxs.push_back(0);
  index = chains->get_total_param_index
    (chains->param_name_to_index("y"),idxs);

  double neff = chains->effective_sample_size(index);

  double y1_mean = chains->mean(index);
  double se = std::sqrt(chains->variance(index)/neff);
  double T = quantile(students_t(neff-1.0), 0.975);
  EXPECT_NEAR(expected_y1, y1_mean, T*se)
    << "T is: " << T << " and se is: " << se << std::endl;
}
TEST_F(Models_BasicDistributions_Binormal, y2) {
  using boost::math::students_t;
  using boost::math::quantile;
  
  size_t index;
  std::vector<size_t> idxs;
  idxs.push_back(1);
  index = chains->get_total_param_index
    (chains->param_name_to_index("y"),idxs);

  double neff = chains->effective_sample_size(index);

  double y2_mean = chains->mean(index);
  double se = std::sqrt(chains->variance(index)/neff);
  double T = quantile(students_t(neff-1.0), 0.975);
  EXPECT_NEAR(expected_y2, y2_mean, T*se)
    << "T is: " << T << " and se is: " << se << std::endl;
}
