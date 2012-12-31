#define _LOG_PROB_ uniform_log
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsUniform : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 0.1;                // y
    param[1] = -0.1;               // alpha
    param[2] = 0.8;                // beta
    parameters.push_back(param);

    param[0] = 0.2;                // y
    param[1] = -0.25;              // alpha
    param[2] = 0.25;               // beta
    parameters.push_back(param);

    param[0] = 0.05;               // y
    param[1] = -5;                 // alpha
    param[2] = 5;                  // beta
    parameters.push_back(param);
  }
 
  void invalid_values(vector<size_t>& /*index*/, 
                      vector<double>& /*value*/) {
    // y
    
    // alpha

    // beta
  }
                             
  template <class T_y, class T_low, class T_high,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, typename T9>
  var log_prob(const T_y& y, const T_low& alpha, const T_high& beta,
               const T3&, const T4&, const T5&, 
               const T6&, const T7&, const T8&, const T9&) {
      using stan::prob::include_summand;
      using stan::prob::LOG_ZERO;

      if (y < alpha || y > beta)
        return LOG_ZERO;

      var lp(0.0);
      if (include_summand<true,T_low,T_high>::value)
          lp -= log(beta - alpha);
      return lp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsUniform,
                              AgradDistributionTestFixture,
                              AgradDistributionsUniform);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsUniform,
                              AgradDistributionTestFixture2,
                              AgradDistributionsUniform);
