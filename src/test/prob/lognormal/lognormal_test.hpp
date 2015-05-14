// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/lognormal_log.hpp>
#include <stan/math/prim/scal/prob/lognormal_cdf.hpp>

#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsLognormal : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 1.5;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.509802579439102343528); // expected log_prob

    param[0] = 12.0;          // y
    param[1] = 3.0;           // mu
    param[2] = 0.9;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.462263160811773765602); // expected log_prob
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
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::lognormal_log(y, mu, sigma);
  }

  template <bool propto, 
            typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::lognormal_log<propto>(y, mu, sigma);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T3&, const T4&, const T5&) {
    using stan::math::pi;
    using stan::math::square;
    using stan::math::NEG_LOG_SQRT_TWO_PI;
      
    return NEG_LOG_SQRT_TWO_PI - log(sigma) - log(y) 
      - square(log(y) - mu) / (2.0 * sigma * sigma);
  }
};



TEST(ProbDistributionsLognormal,Cumulative) {
  using stan::math::lognormal_cdf;
  EXPECT_FLOAT_EQ(0.4687341, lognormal_cdf(1.2,0.3,1.5));
  EXPECT_FLOAT_EQ(0.2835506, lognormal_cdf(12.0,3.0,0.9));

  double pos_inf = std::numeric_limits<double>::infinity();
 
  // ?? double neg_inf = -pos_inf;
  // ?? EXPECT_FLOAT_EQ(0.0,lognormal_cdf(neg_inf,0.0,1.0));

  EXPECT_FLOAT_EQ(0.0,lognormal_cdf(0.0,0.0,1.0));
  EXPECT_FLOAT_EQ(1.0,lognormal_cdf(pos_inf,0.0,1.0));
}

