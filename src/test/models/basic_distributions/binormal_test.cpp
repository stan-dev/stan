#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

#include <boost/math/distributions/students_t.hpp>

class Models_BasicDistributions_Binormal : 
  public ::testing::Model_Test_Fixture<Models_BasicDistributions_Binormal,
                                       false> {
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
    
};

TEST_F(Models_BasicDistributions_Binormal,RunModel) {
  run_models();
}

/*TEST_F(Models_BasicDistributions_Binormal, y1) {
  std::vector<std::string> names;
  std::vector<std::vector<size_t> > dimss;
  stan::mcmc::read_variables(output1, 2,
                             names, dimss);

  stan::mcmc::chains<> c(2, names, dimss);
  stan::mcmc::add_chain(c, 0U, output1, 2U);
  stan::mcmc::add_chain(c, 1U, output2, 2U);

  size_t index;
  std::vector<size_t> idxs;
  idxs.push_back(0);
  index = c.get_total_param_index(c.param_name_to_index("y"), 
                                  idxs);

  double neff = c.effective_sample_size(index);

  boost::math::students_t t_dist(neff-1.0);  
  double T = boost::math::quantile(t_dist, 0.975);
  
  EXPECT_NEAR(expected_y1, c.mean(index), T*sqrt(c.variance(index)/neff));
}
TEST_F(Models_BasicDistributions_Binormal, y2) {
  std::vector<std::string> names;
  std::vector<std::vector<size_t> > dimss;
  stan::mcmc::read_variables(output1, 2,
                             names, dimss);

  stan::mcmc::chains<> c(2, names, dimss);
  stan::mcmc::add_chain(c, 0U, output1, 2U);
  stan::mcmc::add_chain(c, 1U, output2, 2U);

  size_t index;
  std::vector<size_t> idxs;
  idxs.push_back(1);
  index = c.get_total_param_index(c.param_name_to_index("y"), 
                                  idxs);

  double neff = c.effective_sample_size(index);

  boost::math::students_t t_dist(neff-1.0);  
  double T = boost::math::quantile(t_dist, 0.975);
  
  EXPECT_NEAR(expected_y1, c.mean(index), T*sqrt(c.variance(index)/neff));
}
*/
