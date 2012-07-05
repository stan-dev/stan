#include <gtest/gtest.h>
#include <test/models/model_test_fixture.hpp>

class Models_BasicDistributions_OrderedPair : 
  public Model_Test_Fixture<Models_BasicDistributions_OrderedPair> {
protected:
  virtual void SetUp() {
    populate_chains();
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

  static std::vector<std::pair<size_t, double> >
  get_expected_values() {
    std::vector<std::pair<size_t, double> > expected_values;
    return expected_values;
  }

};


INSTANTIATE_TYPED_TEST_CASE_P(Models_BasicDistributions_OrderedPair,
			      Model_Test_Fixture,
			      Models_BasicDistributions_OrderedPair);


TEST_F(Models_BasicDistributions_OrderedPair,
       Test_Ordered_Pair) {
  using std::vector;
  vector<double> a, b;
  chains->get_samples(0U, a);
  chains->get_samples(1U, b);

  for (size_t n = 0; n < chains->num_samples(); n++) {
    EXPECT_GE(a[n], -5);
    EXPECT_LE(a[n], 5);
    EXPECT_GE(b[n], -5);
    EXPECT_LE(b[n], 5);
    EXPECT_LT(a[n], b[n])
      << n << ": expecting " << a[n] << " to be less than " << b[n];
  }
}
