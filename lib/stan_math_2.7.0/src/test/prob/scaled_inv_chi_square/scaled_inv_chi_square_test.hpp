// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/scaled_inv_chi_square_log.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsScaledInvChiSquare : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 12.7;          // y
    param[1] = 6.1;           // nu
    param[2] = 3.0;           // s
    parameters.push_back(param);
    log_prob.push_back(-3.091965468919148054052190729092); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.5;           // s
    parameters.push_back(param);
    log_prob.push_back(-1.737085713764618051197561857864); // expected log_prob
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

    // s
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }


  template <class T_y, class T_dof, class T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob(const T_y& y, const T_dof& nu, const T_scale& s,
           const T3&, const T4&, const T5&) {
    return stan::math::scaled_inv_chi_square_log(y, nu, s);
  }

  template <bool propto, 
            class T_y, class T_dof, class T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob(const T_y& y, const T_dof& nu, const T_scale& s,
           const T3&, const T4&, const T5&) {
    return stan::math::scaled_inv_chi_square_log<propto>(y, nu, s);
  }
  
  
  template <class T_y, class T_dof, class T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob_function(const T_y& y, const T_dof& nu, const T_scale& s,
                    const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::multiply_log;
    using stan::math::square;

    if (y <= 0)
      return stan::math::LOG_ZERO;
    
    return multiply_log(0.5*nu,0.5*nu) - lgamma(0.5*nu) + nu * log(s) 
      - multiply_log(nu*0.5+1.0, y) - nu * 0.5 * square(s) / y;
  }
};
