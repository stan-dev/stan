#define _LOG_PROB_ lognormal_log
#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsLognormal : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 1.5;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.509802579); // expected log_prob

    param[0] = 12.0;          // y
    param[1] = 3.0;           // mu
    param[2] = 0.9;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.462263161); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsLognormal,
			      DistributionTestFixture,
			      ProbDistributionsLognormal);


TEST(ProbDistributionsLognormal,Cumulative) {
  using stan::prob::lognormal_cdf;
  EXPECT_FLOAT_EQ(0.4687341, lognormal_cdf(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(0.2835506, lognormal_cdf(12.0,3.0,0.9));

  double pos_inf = std::numeric_limits<double>::infinity();
  
  // ?? double neg_inf = -pos_inf;
  // ?? EXPECT_FLOAT_EQ(0.0,lognormal_cdf(neg_inf,0.0,1.0));

  EXPECT_FLOAT_EQ(0.0,lognormal_cdf(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(1.0,lognormal_cdf(pos_inf,0.0,1.0));
}
