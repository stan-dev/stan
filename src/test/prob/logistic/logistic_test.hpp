// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/logistic_log.hpp>
#include <stan/math/prim/scal/prob/logistic_cdf.hpp>

#include <stan/math/prim/scal/fun/log1p.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsLogistic : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 2.0;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.129645072554855023128); // expected log_prob

    param[0] = -1.0;          // y
    param[1] = 0.2;           // mu
    param[2] = 0.25;          // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.430097773556644469295); // expected log_prob
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

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::logistic_log(y, mu, sigma);
  }

  template <bool propto, 
            typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::logistic_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T3&, const T4&, const T5&) {
      using stan::math::log1p;
      return -(y - mu) / sigma - log(sigma) - 2.0 * log1p(exp(-(y - mu)/sigma));
  }
};


TEST(ProbDistributionsLogisticCDF, Values) {
    EXPECT_FLOAT_EQ(0.047191944, stan::math::logistic_cdf(-3.45, 5.235, 2.89));
}
