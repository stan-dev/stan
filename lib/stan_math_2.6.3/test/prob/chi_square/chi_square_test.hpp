// Arguments: Doubles, Doubles
#include <stan/math/prim/scal/prob/chi_square_log.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsChiSquare : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 7.9;                 // y
    param[1] = 3.0;                 // nu
    parameters.push_back(param);
    log_prob.push_back(-3.835507153468185048695);  // expected log_prob

    param[0] = 1.9;                 // y
    param[1] = 0.5;                 // nu
    parameters.push_back(param);
    log_prob.push_back(-2.892699734467359284906);    // expected log_prob
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
  }


  template <typename T_y, typename T_dof, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof, T2>::type 
  log_prob(const T_y& y, const T_dof& nu, 
           const T2&, const T3&, const T4&, const T5&) {
    return stan::math::chi_square_log(y, nu);
  }

  template <bool propto, 
      typename T_y, typename T_dof, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof>::type 
  log_prob(const T_y& y, const T_dof& nu, 
           const T2&, const T3&, const T4&, const T5&) {
    return stan::math::chi_square_log<propto>(y, nu);
  }
  
  
  template <typename T_y, typename T_dof, typename T2,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof>::type 
  log_prob_function(const T_y& y, const T_dof& nu, 
                    const T2&, const T3&, const T4&, const T5&) {
    using stan::math::multiply_log;
    using boost::math::lgamma;
    using stan::math::NEG_LOG_TWO_OVER_TWO;
    
    return nu * NEG_LOG_TWO_OVER_TWO - lgamma(0.5 * nu) 
      + multiply_log(0.5*nu-1.0, y) - 0.5 * y;
  }
};
