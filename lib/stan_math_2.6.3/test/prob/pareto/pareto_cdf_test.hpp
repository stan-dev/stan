// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/pareto_cdf.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfPareto : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 0.75;          // y
    param[1] = 0.5;           // y_min (Scale)
    param[2] = 3.3;           // alpha (Shape)
    parameters.push_back(param);
    cdf.push_back(0.7376392612457611797083);  // expected CDF

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
    // y_min
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());
      
    // alpha
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

  }
  
  bool has_lower_bound() {
    return true;
  }
  
  double lower_bound() {
    return 0.0;
  }
    
  bool has_upper_bound() {
    return false;
  }
    
  template <typename T_y, typename T_scale, typename T_shape,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  cdf(const T_y& y, const T_scale& y_min, const T_shape& alpha,
      const T3&, const T4&, const T5&) {
    return stan::math::pareto_cdf(y, y_min, alpha);
  }


  
  template <typename T_y, typename T_scale, typename T_shape,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale, T_shape>::type 
  cdf_function(const T_y& y, const T_scale& y_min, const T_shape& alpha,
               const T3&, const T4&, const T5&) {
      using std::exp;
      using std::log;
      return 1.0 - exp( alpha * log( y_min / y ) );
  }
    
};
