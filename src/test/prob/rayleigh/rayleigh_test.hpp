// Arguments: Doubles, Doubles
#include <stan/math/prim/scal/prob/rayleigh_log.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionRayleigh : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 4;           // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-6.613705638880109381165535757083646863848999731279489491758639981013212756060610568788273346007162625); // expected log_prob

    param[0] = 1;           // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.5); // expected log_prob

    param[0] = 2;          // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.306852819440054690582767878541823431924499865639744745879319990506606378030305284394136673003581312); // expected log_prob

    param[0] = 3.5;          // y
    param[1] = 7.2;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.8135510897214907645); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {    
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // sigma
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  template <typename T_y, typename T_scale, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale>::type 
  log_prob(const T_y& y, const T_scale& sigma, const T2&,
           const T3&, const T4&, const T5&) {
    return stan::math::rayleigh_log(y, sigma);
  }

  template <bool propto, 
            typename T_y, typename T_scale, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale>::type 
  log_prob(const T_y& y, const T_scale& sigma, const T2&,
           const T3&, const T4&, const T5&) {
    return stan::math::rayleigh_log<propto>(y, sigma);
  }
  
  
  template <typename T_y, typename T_scale, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale>::type 
  log_prob_function(const T_y& y, const T_scale& sigma, const T2&,
                    const T3&, const T4&, const T5&) {
    using stan::math::pi;
    using stan::math::square;
    return -0.5 * y * y / (sigma * sigma) - 2.0 * log(sigma) + log(y);
  }
};

