#define _LOG_PROB_ student_t_log
#include <stan/prob/distributions/univariate/continuous/student_t.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_4_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsStudentT : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(4);

    param[0] = 1.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.0;           // mu
    param[3] = 1.0;           // sigma
    parameters.push_back(param);

    param[0] = -3.0;          // y
    param[1] = 2.0;           // nu
    param[2] = 0.0;           // mu
    param[3] = 1.0;           // sigma
    parameters.push_back(param);

    param[0] = 2.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.0;           // mu
    param[3] = 2.0;           // sigma
    parameters.push_back(param);
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // nu
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // mu
    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(3U);
    value.push_back(0.0);

    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <class T_y, class T_dof, class T_loc, class T_scale,
	    typename T4, typename T5, typename T6, 
	    typename T7, typename T8, typename T9>
  var log_prob(const T_y& y, const T_dof& nu, const T_loc& mu, const T_scale& sigma,
	       const T4&, const T5&, const T6&, 
	       const T7&, const T8&, const T9&) {
    using std::log;
    using stan::math::square;
    using stan::math::log1p;
    using boost::math::lgamma;
    using stan::prob::include_summand;
    using stan::prob::NEG_LOG_SQRT_PI;
    
    var logp(0);
    if (include_summand<true,T_dof>::value)
      logp += lgamma( (nu + 1.0) / 2.0) - lgamma(nu / 2.0);
    if (include_summand<true>::value)
      logp += NEG_LOG_SQRT_PI;
    if (include_summand<true,T_dof>::value)
      logp -= 0.5 * log(nu);
    if (include_summand<true,T_scale>::value)
      logp -= log(sigma);
    if (include_summand<true,T_y,T_dof,T_loc,T_scale>::value)
      logp -= ((nu + 1.0) / 2.0) 
	* log1p( square(((y - mu) / sigma)) / nu);
    return logp;
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsStudentT,
			      AgradDistributionTestFixture,
			      AgradDistributionsStudentT);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsStudentT,
			      AgradDistributionTestFixture2,
			      AgradDistributionsStudentT);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsStudentT,
			      AgradDistributionTestFixture3,
			      AgradDistributionsStudentT);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsStudentT,
			      AgradDistributionTestFixture4,
			      AgradDistributionsStudentT);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsStudentT,
			      AgradDistributionTestFixture5,
			      AgradDistributionsStudentT);
