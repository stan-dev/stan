// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/lomax.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsLomax : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.5;           // y
    param[1] = 0.5;           // lambda
    param[2] = 3.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-std::log(128/3)); // expected log_prob

    param[0] = 19.5;          // y
    param[1] = 0.15;          // lambda
    param[2] = 5.0;           // alpha
    parameters.push_back(param);
    log_prob.push_back(-std::log(15161739432843/100)); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // lambda
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // alpha
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }


  template <class T_y, class T_scale, class T_shape,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_scale& lambda, const T_shape& alpha,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::lomax_log(y, lambda, alpha);
  }

  template <bool propto, 
            class T_y, class T_scale, class T_shape,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  log_prob(const T_y& y, const T_scale& lambda, const T_shape& alpha,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::lomax_log<propto>(y, lambda, alpha);
  }
  

  template <class T_y, class T_scale, class T_shape,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, typename T9>
  var log_prob_function(const T_y& y, const T_scale& lambda, const T_shape& alpha,
                        const T3&, const T4&, const T5&, 
                        const T6&, const T7&, const T8&, const T9&) {

      return log(alpha) - log(lambda) - (alpha + 1) * log(1 + y / lambda);
  }
};

// TEST(ProbDistributionsLomaxCDF, Values) {
//     EXPECT_FLOAT_EQ(0.60434447, stan::prob::lomax_cdf(3.45, 2.89, 5.235));
// }
