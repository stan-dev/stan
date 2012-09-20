#define _LOG_PROB_ normal_log
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsNormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    parameters.push_back(param);
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
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <class T_y, class T_loc, class T_scale>
  var log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      using stan::prob::include_summand;
      using stan::math::pi;
      using stan::math::square;
      var lp(0.0);
      if (include_summand<true,T_y,T_loc,T_scale>::value)
	lp -= 0.5 * (y - mu) * (y - mu) / (sigma * sigma);
      if (include_summand<true,T_scale>::value)
	lp -= log(sigma);
      if (include_summand<true>::value)
	lp -= log(sqrt(2.0 * pi()));
      return lp;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsNormal,
			      AgradDistributionTestFixture,
			      AgradDistributionsNormal);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsNormal,
			      AgradDistributionTestFixture2,
			      AgradDistributionsNormal);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsNormal,
			      AgradDistributionTestFixture3,
			      AgradDistributionsNormal);

TEST(AgradDistributions,NormalCdfGrad) {
  using stan::agrad::var;
  using std::vector;
  var y = 1.0;
  var mu = 5.0;
  var sigma = 12.0;
  std::vector<var> x(3);
  x[0] = y;
  x[1] = mu;
  x[2] = sigma;

  var p = stan::prob::normal_cdf(y,mu,sigma);

  std::vector<double> g;
  p.grad(x,g);
}
