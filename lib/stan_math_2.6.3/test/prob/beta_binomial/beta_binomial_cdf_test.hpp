// Arguments: Ints, Ints, Doubles, Doubles
#include <stan/math/prim/scal/prob/beta_binomial_cdf.hpp>
#include <boost/math/special_functions/binomial.hpp>

#include <stan/math/prim/scal/fun/lbeta.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfBetaBinomial : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 17;         // n
    param[1] = 45;         // N
    param[2] = 13;         // alpha
    param[3] = 15;         // beta
    parameters.push_back(param);
    cdf.push_back(0.2680523296111902594541368047465042591108987357277389728827508022844102288442195709664943608618480998); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // N
    index.push_back(1U);
    value.push_back(-1);
      
    // alpha
    index.push_back(2U);
    value.push_back(-1);
      
    // beta
    index.push_back(3U);
    value.push_back(-1);

  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }

  template <typename T_n, typename T_N, typename T_size1, typename T_size2, 
            typename T4, typename T5>
  typename stan::return_type<T_size1,T_size2>::type
  cdf(const T_n& n, const T_N& N, const T_size1& alpha, const T_size2& beta, 
      const T4&, const T5&) {
    return stan::math::beta_binomial_cdf(n, N, alpha, beta);
  }


  template <typename T_n, typename T_N, typename T_size1, typename T_size2, 
            typename T4, typename T5>
  typename stan::return_type<T_size1,T_size2>::type
  cdf_function(const T_n& n, const T_N& N, const T_size1& alpha, 
               const T_size2& beta, const T4&, const T5&) {

    using std::exp;
    using stan::math::lbeta;
    using boost::math::binomial_coefficient;
      
    typename stan::return_type<T_size1,T_size2>::type cdf(0);
 
    for (int i = 0; i <= n; i++) {
      cdf += binomial_coefficient<double>(N, i) 
             * exp( lbeta(alpha + i, N - i + beta) - lbeta(alpha, beta) );
    }
      
    return cdf;
      
  }
};
