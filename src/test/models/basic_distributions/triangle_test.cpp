#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>
#include <algorithm>

class Models_BasicDistributions_Triangle : 
  public Model_Test_Fixture<Models_BasicDistributions_Triangle> {
protected:
  virtual void SetUp() {
  }
public:
  static std::vector<std::string> get_model_path() {
    std::vector<std::string> model_path;
    model_path.push_back("models");
    model_path.push_back("basic_distributions");
    model_path.push_back("triangle");
    return model_path;
  }
  static bool has_data() {
    return false;
  }

  static bool has_init() {
    return false;
  }

  static size_t num_iterations() {
    return iterations;
  }

  static std::vector<size_t> skip_chains_test() {
    std::vector<size_t> params_to_skip;
    return params_to_skip;
  }

  static void populate_chains() {
    default_populate_chains();
  }

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    using std::make_pair;
    std::vector<std::pair<size_t, double> > expected_values;
    
    expected_values.push_back(make_pair(0U, 0));
    
    return expected_values;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_Triangle,
			      Model_Test_Fixture,
			      Models_BasicDistributions_Triangle);

TEST_F(Models_BasicDistributions_Triangle,
	Test_Triangle) {
  populate_chains();
  
  std::vector<double> y;
  
  chains->get_samples(0U, y);
  
  double min, max;
  min = *std::min_element(y.begin(), y.end());
  max = *std::max_element(y.begin(), y.end());

  EXPECT_LE(min, -0.9)
    << "expecting to get close to the corner";
  EXPECT_GE(max, 0.9)
    << "expecting to get close to the corner";
}

