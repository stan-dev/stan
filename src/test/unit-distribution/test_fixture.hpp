#ifndef TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_HPP_
#define TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_HPP_

#include <stdexcept>
#include <stan/error_handling.hpp>
#include <stan/math/matrix.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit-distribution/utility.hpp>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::agrad::var;
using stan::scalar_type;
using stan::is_vector;
using stan::is_constant;
using stan::is_constant_struct;
using stan::math::value_of;


/** 
 * To test a distribution, define a subclass of AgradDistributionTest.
 * Implement each of the functions.
 * 
 */
class AgradDistributionTest {
public:
  virtual void valid_values(vector<vector<double> >& /*parameters*/,
                            vector<double>& /* log_prob */) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& /*index*/, 
                              vector<double>& /*value*/) {
    throw std::runtime_error("invalid_values() not implemented");
  }

  // also include 3 templated functions:
  /*
    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_log(y, mu, sigma);
    }

    template <bool propto, 
    typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    log_prob(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_log<propto>(y, mu, sigma);
    }
    
    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::pi;
    using stan::math::square;
    var lp(0.0);
    if (include_summand<true,T_y,T_loc,T_scale>::value)
    lp -= 0.5 * (y - mu) * (y - mu) / (sigma * sigma);
    if (include_summand<true,T_scale>::value)
    lp -= log(sigma);
    if (include_summand<true>::value)
    lp -= log(sqrt(2.0 * pi()));
    return lp;
    }
  */
};

class AgradCdfLogTest {
public:
  virtual void valid_values(vector<vector<double> >& /*parameters*/,
                            vector<double>& /* cdf_log */) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& /*index*/, 
                              vector<double>& /*value*/) {
    throw std::runtime_error("invalid_values() not implemented");
  }
  
  virtual bool has_lower_bound() {
    return false;
  }
  
  virtual double lower_bound() {
    return -std::numeric_limits<double>::infinity();
  }

  virtual bool has_upper_bound() {
    return false;
  }
  
  virtual double upper_bound() {
    return std::numeric_limits<double>::infinity();
  }

  // also include 2 templated functions:
  /*
    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_cdf_log(y, mu, sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    cdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::math::erf;
    return (0.5 + 0.5 * erf((y - mu) / (sigma * SQRT_2)));
    }
  */
};

class AgradCdfTest {
public:
  virtual void valid_values(vector<vector<double> >& /*parameters*/,
                            vector<double>& /* cdf */) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& /*index*/, 
                              vector<double>& /*value*/) {
    throw std::runtime_error("invalid_values() not implemented");
  }
  
  virtual bool has_lower_bound() {
    return false;
  }
  
  virtual double lower_bound() {
    return -std::numeric_limits<double>::infinity();
  }

  virtual bool has_upper_bound() {
    return false;
  }
  
  virtual double upper_bound() {
    return std::numeric_limits<double>::infinity();
  }

  // also include 2 templated functions:
  /*
    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_cdf(y, mu, sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    cdf_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::math::erf;
    return (0.5 + 0.5 * erf((y - mu) / (sigma * SQRT_2)));
    }
  */
};

class AgradCcdfLogTest {
public:
  virtual void valid_values(vector<vector<double> >& /*parameters*/,
                            vector<double>& /* cdf_log */) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& /*index*/, 
                              vector<double>& /*value*/) {
    throw std::runtime_error("invalid_values() not implemented");
  }
  
  virtual bool has_lower_bound() {
    return false;
  }
  
  virtual double lower_bound() {
    return -std::numeric_limits<double>::infinity();
  }

  virtual bool has_upper_bound() {
    return false;
  }
  
  virtual double upper_bound() {
    return std::numeric_limits<double>::infinity();
  }

  // also include 2 templated functions:
  /*
    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::normal_cdf_log(y, mu, sigma);
    }

    template <typename T_y, typename T_loc, typename T_scale,
    typename T3, typename T4, typename T5, 
    typename T6, typename T7, typename T8, 
    typename T9>
    typename stan::return_type<T_y, T_loc, T_scale>::type 
    cdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
    const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::math::erf;
    return (0.5 + 0.5 * erf((y - mu) / (sigma * SQRT_2)));
    }
  */
};

using boost::mpl::at_c;
template<class T>
class AgradDistributionTestFixture : public ::testing::Test {
public:
  typename at_c<T,0>::type TestClass;
  typedef typename at_c<typename at_c<T,1>::type, 0>::type T0;
  typedef typename at_c<typename at_c<T,1>::type, 1>::type T1;
  typedef typename at_c<typename at_c<T,1>::type, 2>::type T2;
  typedef typename at_c<typename at_c<T,1>::type, 3>::type T3;
  typedef typename at_c<typename at_c<T,1>::type, 4>::type T4;
  typedef typename at_c<typename at_c<T,1>::type, 5>::type T5;
  typedef typename at_c<typename at_c<T,1>::type, 6>::type T6;
  typedef typename at_c<typename at_c<T,1>::type, 7>::type T7;
  typedef typename at_c<typename at_c<T,1>::type, 8>::type T8;
  typedef typename at_c<typename at_c<T,1>::type, 9>::type T9;

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;
  typedef typename scalar_type<T6>::type Scalar6;
  typedef typename scalar_type<T7>::type Scalar7;
  typedef typename scalar_type<T8>::type Scalar8;
  typedef typename scalar_type<T9>::type Scalar9;
  
  void call_all_versions() {
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    T0 p0 = get_params<T0>(parameters, 0);
    T1 p1 = get_params<T1>(parameters, 1);
    T2 p2 = get_params<T2>(parameters, 2);
    T3 p3 = get_params<T3>(parameters, 3);
    T4 p4 = get_params<T4>(parameters, 4);
    T5 p5 = get_params<T5>(parameters, 5);
    T6 p6 = get_params<T6>(parameters, 6);
    T7 p7 = get_params<T7>(parameters, 7);
    T8 p8 = get_params<T8>(parameters, 8);
    T9 p9 = get_params<T9>(parameters, 9);
    
    EXPECT_NO_THROW(({ TestClass.template log_prob
            <T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with default parameters";

    EXPECT_NO_THROW(({ TestClass.template log_prob
            <true, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with propto=true";

    EXPECT_NO_THROW(({ TestClass.template log_prob
            <false, T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling log_prob throws exception with propto=false";
  }

  void test_valid_values() {
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_params<T0>(parameters, n, 0);
      T1 p1 = get_params<T1>(parameters, n, 1);
      T2 p2 = get_params<T2>(parameters, n, 2);
      T3 p3 = get_params<T3>(parameters, n, 3);
      T4 p4 = get_params<T4>(parameters, n, 4);
      T5 p5 = get_params<T5>(parameters, n, 5);
      T6 p6 = get_params<T6>(parameters, n, 6);
      T7 p7 = get_params<T7>(parameters, n, 7);
      T8 p8 = get_params<T8>(parameters, n, 8);
      T9 p9 = get_params<T9>(parameters, n, 9);

      var lp(0);
      EXPECT_NO_THROW(({ lp = TestClass.template log_prob
              <true,T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];

      if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
        // all double inputs should result in a log probability of 0
        EXPECT_FLOAT_EQ(0.0, lp.val())
          << "All constant inputs should result in 0 log probability. Failed at index: " << n;
      }
      if (all_scalar<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
        lp = TestClass.template log_prob
          <false,T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
          (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
        EXPECT_FLOAT_EQ(log_prob[n], lp.val())
          << "For all scalar inputs, when propto is false, log_prob should match the provided value. Failed at index: " << n;
      }
      if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value 
          && all_scalar<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
        lp = TestClass.template log_prob
          <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
          (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
        EXPECT_FLOAT_EQ(log_prob[n], lp.val())
          << "For all scalar and all constant inputs log_prob should match the provided value. Failed at index: " << n;
      }
    }
  }

  void test_nan_value(const vector<double>& parameters, const size_t n) {
    var lp(0);
    vector<double> invalid_params(parameters);
    invalid_params[n] = std::numeric_limits<double>::quiet_NaN();
    
    Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
    Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
    Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
    Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
    Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
    Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
    Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
    Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
    Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
    Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);
      
    EXPECT_THROW(({ TestClass.template log_prob
            <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
            (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
      std::domain_error) 
      << "NaN value at index " << n << " should have failed" << std::endl
      << invalid_params;
  }
  
  void test_invalid_values() {
    if (!all_scalar<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value)
      return;

    vector<double> parameters = this->first_valid_params();
    
    vector<size_t> index;
    vector<double> invalid_values;
    TestClass.invalid_values(index, invalid_values);

    for (size_t n = 0; n < index.size(); n++) {
      var lp(0);
      vector<double> invalid_params(parameters);
      invalid_params[index[n]] = invalid_values[n];
      
      Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
      Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
      Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
      Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
      Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
      Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
      Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
      Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
      Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
      Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);

      EXPECT_THROW(({ TestClass.template log_prob
              <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
        std::domain_error) 
        << "Invalid value " << n << " should have failed" << std::endl
        << invalid_params;
    }
    if (std::numeric_limits<Scalar0>::has_quiet_NaN && parameters.size() > 0) 
      test_nan_value(parameters, 0);
    if (std::numeric_limits<Scalar1>::has_quiet_NaN && parameters.size() > 1) 
      test_nan_value(parameters, 1);
    if (std::numeric_limits<Scalar2>::has_quiet_NaN && parameters.size() > 2) 
      test_nan_value(parameters, 2);
    if (std::numeric_limits<Scalar3>::has_quiet_NaN && parameters.size() > 3) 
      test_nan_value(parameters, 3);
    if (std::numeric_limits<Scalar4>::has_quiet_NaN && parameters.size() > 4) 
      test_nan_value(parameters, 4);
    if (std::numeric_limits<Scalar5>::has_quiet_NaN && parameters.size() > 5) 
      test_nan_value(parameters, 5);
    if (std::numeric_limits<Scalar6>::has_quiet_NaN && parameters.size() > 6) 
      test_nan_value(parameters, 6);
    if (std::numeric_limits<Scalar7>::has_quiet_NaN && parameters.size() > 7) 
      test_nan_value(parameters, 7);
    if (std::numeric_limits<Scalar8>::has_quiet_NaN && parameters.size() > 8) 
      test_nan_value(parameters, 8);
    if (std::numeric_limits<Scalar9>::has_quiet_NaN && parameters.size() > 9) 
      test_nan_value(parameters, 9);
  }

  void test_propto() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << " No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    var reference_logprob_true;
    var reference_logprob_false;
    {
      Scalar0 p0 = get_param<Scalar0>(parameters[0], 0);
      Scalar1 p1 = get_param<Scalar1>(parameters[0], 1);
      Scalar2 p2 = get_param<Scalar2>(parameters[0], 2);
      Scalar3 p3 = get_param<Scalar3>(parameters[0], 3);
      Scalar4 p4 = get_param<Scalar4>(parameters[0], 4);
      Scalar5 p5 = get_param<Scalar5>(parameters[0], 5);
      Scalar6 p6 = get_param<Scalar6>(parameters[0], 6);
      Scalar7 p7 = get_param<Scalar7>(parameters[0], 7);
      Scalar8 p8 = get_param<Scalar8>(parameters[0], 8);
      Scalar9 p9 = get_param<Scalar9>(parameters[0], 9);

      reference_logprob_true 
        = TestClass.template log_prob
        <true,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      reference_logprob_false 
        = TestClass.template log_prob
        <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    }
    
    for (size_t n = 0; n < parameters.size(); n++) {
      Scalar0 p0 = select_var_param<T0>(parameters, n, 0);
      Scalar1 p1 = select_var_param<T1>(parameters, n, 1);
      Scalar2 p2 = select_var_param<T2>(parameters, n, 2);
      Scalar3 p3 = select_var_param<T3>(parameters, n, 3);
      Scalar4 p4 = select_var_param<T4>(parameters, n, 4);
      Scalar5 p5 = select_var_param<T5>(parameters, n, 5);
      Scalar6 p6 = select_var_param<T6>(parameters, n, 6);
      Scalar7 p7 = select_var_param<T7>(parameters, n, 7);
      Scalar8 p8 = select_var_param<T8>(parameters, n, 8);
      Scalar9 p9 = select_var_param<T9>(parameters, n, 9);

      var logprob_true
        = TestClass.template log_prob
        <true,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      var logprob_false
        = TestClass.template log_prob
        <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);

      EXPECT_FLOAT_EQ(reference_logprob_false.val() - logprob_false.val(),
                      reference_logprob_true.val() - logprob_true.val())
        << "Proportional test failed at index: " << n << std::endl
        << "  reference params: " << parameters[0] << std::endl
        << "  current params:   " << parameters[n] << std::endl
        << "  ref<true> = " << reference_logprob_true << std::endl
        << "  cur<true> = " << logprob_true << std::endl
        << "  ref<false> = " << reference_logprob_false << std::endl
        << "  cur<false> = " << logprob_false;
    }
  }

  void add_finite_diff(const vector<double>& params, 
                       vector<double>& finite_diff, 
                       const size_t n) {
    const double e = 1e-8;
    const double e2 = 2 * e;

    vector<double> plus(10);
    vector<double> minus(10);
    for (size_t i = 0; i < 10; i++) {
      plus[i] = get_param<double>(params, i);
      minus[i] = get_param<double>(params, i);
    }
    plus[n] += e;
    minus[n] -= e;
    
    double lp_plus = TestClass.log_prob
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5],plus[6],plus[7],plus[8],plus[9]);
    double lp_minus = TestClass.log_prob
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5],minus[6],minus[7],minus[8],minus[9]);
    
    finite_diff.push_back((lp_plus - lp_minus) / e2);
  }
  

  void calculate_finite_diff(const vector<double>& params, vector<double>& finite_diff) {
    if (!is_constant_struct<Scalar0>::value && !is_empty<Scalar0>::value)
      add_finite_diff(params, finite_diff, 0);
    if (!is_constant_struct<Scalar1>::value && !is_empty<Scalar1>::value)
      add_finite_diff(params, finite_diff, 1);
    if (!is_constant_struct<Scalar2>::value && !is_empty<Scalar2>::value)
      add_finite_diff(params, finite_diff, 2);
    if (!is_constant_struct<Scalar3>::value && !is_empty<Scalar3>::value)
      add_finite_diff(params, finite_diff, 3);
    if (!is_constant_struct<Scalar4>::value && !is_empty<Scalar4>::value)
      add_finite_diff(params, finite_diff, 4);
    if (!is_constant_struct<Scalar5>::value && !is_empty<Scalar5>::value)
      add_finite_diff(params, finite_diff, 5);
    if (!is_constant_struct<Scalar6>::value && !is_empty<Scalar6>::value)
      add_finite_diff(params, finite_diff, 6);
    if (!is_constant_struct<Scalar7>::value && !is_empty<Scalar7>::value)
      add_finite_diff(params, finite_diff, 7);
    if (!is_constant_struct<Scalar8>::value && !is_empty<Scalar8>::value)
      add_finite_diff(params, finite_diff, 8);
    if (!is_constant_struct<Scalar9>::value && !is_empty<Scalar9>::value)
      add_finite_diff(params, finite_diff, 9);
  }

  double calculate_gradients(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var logprob = TestClass.template log_prob
      <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    logprob.grad(x, grad);
    return logprob.val();
  }
  
  double calculate_gradients_with_function(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var logprob = TestClass.template log_prob_function
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    logprob.grad(x, grad);
    return logprob.val();
  }

  void test_finite_diff() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> finite_diffs;
      vector<double> gradients;

      calculate_finite_diff(parameters[n], finite_diffs);
      calculate_gradients(parameters[n], gradients);

      ASSERT_EQ(finite_diffs.size(), gradients.size()) 
        << "Number of finite diff gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < finite_diffs.size(); i++) {
        EXPECT_NEAR(finite_diffs[i], gradients[i], 1e-4)
          << "Comparison of finite diff to calculated gradient failed for i=" << i 
          << ": " << parameters[n] << std::endl 
          << "  finite diffs: " << finite_diffs << std::endl
          << "  grads:        " << gradients;
      }
    }
  }

  void test_gradient_function() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> expected_gradients;
      vector<double> gradients;

      calculate_gradients_with_function(parameters[n], expected_gradients);
      calculate_gradients(parameters[n], gradients);

      ASSERT_EQ(expected_gradients.size(), gradients.size()) 
        << "Number of expected gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < expected_gradients.size(); i++) {
        EXPECT_FLOAT_EQ(expected_gradients[i], gradients[i])
          << "Comparison of expected gradient to calculated gradient failed";
      }
    }
  }

  void test_multiple_gradient_values(const bool is_vec,
                                     const vector<double>& single_gradients, size_t& pos_single,
                                     const vector<double>& multiple_gradients, size_t& pos_multiple,
                                     const size_t N_REPEAT) {
    if (is_vec) {
      for (size_t i = 0; i < N_REPEAT; i++) {
        EXPECT_FLOAT_EQ(single_gradients[pos_single],
                        multiple_gradients[pos_multiple])
          << "Comparison of single_gradient value to vectorized gradient failed";
        pos_multiple++;
      }
      pos_single++; 
    } else {
      EXPECT_FLOAT_EQ(single_gradients[pos_single]*double(N_REPEAT), 
                      multiple_gradients[pos_multiple])
        << "Comparison of single_gradient value to vectorized gradient failed";
      pos_single++; pos_multiple++;
    }
  }

  void test_repeat_as_vector() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 3;
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> single_gradients;
      double single_lp = calculate_gradients(parameters[n], single_gradients);
      
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);

      var multiple_lp = TestClass.template log_prob
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      vector<double> multiple_gradients;
      vector<var> x;
      add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
      multiple_lp.grad(x, multiple_gradients);
      

      EXPECT_FLOAT_EQ(N_REPEAT * single_lp, multiple_lp.val())
        << "log prob with repeated vector input should match "
        << "a multiple of log prob of single input";

      size_t pos_single = 0;
      size_t pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value)
        test_multiple_gradient_values(is_vector<T0>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value)
        test_multiple_gradient_values(is_vector<T1>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value)
        test_multiple_gradient_values(is_vector<T2>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value)
        test_multiple_gradient_values(is_vector<T3>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value)
        test_multiple_gradient_values(is_vector<T4>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value)
        test_multiple_gradient_values(is_vector<T5>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T6>::value && !is_empty<T6>::value)
        test_multiple_gradient_values(is_vector<T6>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T7>::value && !is_empty<T7>::value)
        test_multiple_gradient_values(is_vector<T7>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T8>::value && !is_empty<T8>::value)
        test_multiple_gradient_values(is_vector<T8>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T9>::value && !is_empty<T9>::value)
        test_multiple_gradient_values(is_vector<T9>::value, 
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 0;
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    T0 p0 = get_repeated_params<T0>(parameters[0], 0, N_REPEAT);
    T1 p1 = get_repeated_params<T1>(parameters[0], 1, N_REPEAT);
    T2 p2 = get_repeated_params<T2>(parameters[0], 2, N_REPEAT);
    T3 p3 = get_repeated_params<T3>(parameters[0], 3, N_REPEAT);
    T4 p4 = get_repeated_params<T4>(parameters[0], 4, N_REPEAT);
    T5 p5 = get_repeated_params<T5>(parameters[0], 5, N_REPEAT);
    T6 p6 = get_repeated_params<T6>(parameters[0], 6, N_REPEAT);
    T7 p7 = get_repeated_params<T7>(parameters[0], 7, N_REPEAT);
    T8 p8 = get_repeated_params<T8>(parameters[0], 8, N_REPEAT);
    T9 p9 = get_repeated_params<T9>(parameters[0], 9, N_REPEAT);

    var lp = TestClass.template log_prob
      <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);

    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "log prob with an empty vector should return 0.0";
  }
  
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> log_prob;

    TestClass.valid_values(params, log_prob); 
    return params[0];
  }
};
TYPED_TEST_CASE_P(AgradDistributionTestFixture);

TYPED_TEST_P(AgradDistributionTestFixture, CallAllVersions) {
  this->call_all_versions();
}

TYPED_TEST_P(AgradDistributionTestFixture, ValidValues) {
  this->test_valid_values();
}

TYPED_TEST_P(AgradDistributionTestFixture, InvalidValues) {
  this->test_invalid_values();
}

TYPED_TEST_P(AgradDistributionTestFixture, Propto) {
  this->test_propto();
}

TYPED_TEST_P(AgradDistributionTestFixture, FiniteDiff) {
  this->test_finite_diff();
}

TYPED_TEST_P(AgradDistributionTestFixture, Function) {
  this->test_gradient_function();
}

TYPED_TEST_P(AgradDistributionTestFixture, RepeatAsVector) {
  this->test_repeat_as_vector();
}

TYPED_TEST_P(AgradDistributionTestFixture, Length0Vector) {
  this->test_length_0_vector();
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
                           CallAllVersions,
                           ValidValues,
                           InvalidValues,
                           Propto,
                           FiniteDiff,
                           Function,
                           RepeatAsVector,
                           Length0Vector);


template<class T>
class AgradCdfTestFixture : public ::testing::Test {
public:
  typename at_c<T,0>::type TestClass;
  typedef typename at_c<typename at_c<T,1>::type, 0>::type T0;
  typedef typename at_c<typename at_c<T,1>::type, 1>::type T1;
  typedef typename at_c<typename at_c<T,1>::type, 2>::type T2;
  typedef typename at_c<typename at_c<T,1>::type, 3>::type T3;
  typedef typename at_c<typename at_c<T,1>::type, 4>::type T4;
  typedef typename at_c<typename at_c<T,1>::type, 5>::type T5;
  typedef typename at_c<typename at_c<T,1>::type, 6>::type T6;
  typedef typename at_c<typename at_c<T,1>::type, 7>::type T7;
  typedef typename at_c<typename at_c<T,1>::type, 8>::type T8;
  typedef typename at_c<typename at_c<T,1>::type, 9>::type T9;

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;
  typedef typename scalar_type<T6>::type Scalar6;
  typedef typename scalar_type<T7>::type Scalar7;
  typedef typename scalar_type<T8>::type Scalar8;
  typedef typename scalar_type<T9>::type Scalar9;
  
  void call_all_versions() {
    vector<double> cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, cdf);
    
    T0 p0 = get_params<T0>(parameters, 0);
    T1 p1 = get_params<T1>(parameters, 1);
    T2 p2 = get_params<T2>(parameters, 2);
    T3 p3 = get_params<T3>(parameters, 3);
    T4 p4 = get_params<T4>(parameters, 4);
    T5 p5 = get_params<T5>(parameters, 5);
    T6 p6 = get_params<T6>(parameters, 6);
    T7 p7 = get_params<T7>(parameters, 7);
    T8 p8 = get_params<T8>(parameters, 8);
    T9 p9 = get_params<T9>(parameters, 9);
    
    EXPECT_NO_THROW(({ TestClass.template cdf
            <T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling cdf throws exception with default parameters";
  }

  void test_valid_values() {
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_params<T0>(parameters, n, 0);
      T1 p1 = get_params<T1>(parameters, n, 1);
      T2 p2 = get_params<T2>(parameters, n, 2);
      T3 p3 = get_params<T3>(parameters, n, 3);
      T4 p4 = get_params<T4>(parameters, n, 4);
      T5 p5 = get_params<T5>(parameters, n, 5);
      T6 p6 = get_params<T6>(parameters, n, 6);
      T7 p7 = get_params<T7>(parameters, n, 7);
      T8 p8 = get_params<T8>(parameters, n, 8);
      T9 p9 = get_params<T9>(parameters, n, 9);

      var cdf(0);
      EXPECT_NO_THROW(({ cdf = TestClass.template cdf
              <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];
      EXPECT_TRUE(cdf.val() >= 0)
        << "cdf value must be greater than or equal to 0. cdf value: " 
        << cdf;
      EXPECT_TRUE(cdf.val() <= 1)
        << "cdf value must be less than or equal to 1. cdf value: "
        << cdf;

      if (all_scalar<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
        EXPECT_FLOAT_EQ(expected_cdf[n], cdf.val())
          << "For all scalar inputs cdf should match the provided value. Failed at index: " << n;
      }
    }
  }

  void test_nan_value(const vector<double>& parameters, const size_t n) {
    var cdf(0);
    vector<double> invalid_params(parameters);
    invalid_params[n] = std::numeric_limits<double>::quiet_NaN();
    
    Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
    Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
    Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
    Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
    Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
    Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
    Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
    Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
    Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
    Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);
      
    EXPECT_THROW(({ TestClass.template cdf
            <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
            (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
      std::domain_error) 
      << "NaN value at index " << n << " should have failed" << std::endl
      << invalid_params;
  }

  void test_invalid_values() {
    vector<double> parameters = this->first_valid_params();
    
    vector<size_t> index;
    vector<double> invalid_values;
    TestClass.invalid_values(index, invalid_values);

    for (size_t n = 0; n < index.size(); n++) {
      var cdf(0);
      vector<double> invalid_params(parameters);
      invalid_params[index[n]] = invalid_values[n];
      
      Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
      Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
      Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
      Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
      Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
      Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
      Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
      Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
      Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
      Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);

      EXPECT_THROW(({ TestClass.template cdf
              <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
        std::domain_error) 
        << "Invalid value " << n << " should have failed" << std::endl
        << invalid_params;      
    }
    if (std::numeric_limits<Scalar0>::has_quiet_NaN && parameters.size() > 0) 
      test_nan_value(parameters, 0);
    if (std::numeric_limits<Scalar1>::has_quiet_NaN && parameters.size() > 1) 
      test_nan_value(parameters, 1);
    if (std::numeric_limits<Scalar2>::has_quiet_NaN && parameters.size() > 2) 
      test_nan_value(parameters, 2);
    if (std::numeric_limits<Scalar3>::has_quiet_NaN && parameters.size() > 3) 
      test_nan_value(parameters, 3);
    if (std::numeric_limits<Scalar4>::has_quiet_NaN && parameters.size() > 4) 
      test_nan_value(parameters, 4);
    if (std::numeric_limits<Scalar5>::has_quiet_NaN && parameters.size() > 5) 
      test_nan_value(parameters, 5);
    if (std::numeric_limits<Scalar6>::has_quiet_NaN && parameters.size() > 6) 
      test_nan_value(parameters, 6);
    if (std::numeric_limits<Scalar7>::has_quiet_NaN && parameters.size() > 7) 
      test_nan_value(parameters, 7);
    if (std::numeric_limits<Scalar8>::has_quiet_NaN && parameters.size() > 8) 
      test_nan_value(parameters, 8);
    if (std::numeric_limits<Scalar9>::has_quiet_NaN && parameters.size() > 9) 
      test_nan_value(parameters, 9);
  }

  void add_finite_diff(const vector<double>& params, 
                       vector<double>& finite_diff, 
                       const size_t n) {
    const double e = 1e-8;
    const double e2 = 2 * e;

    vector<double> plus(10);
    vector<double> minus(10);
    for (size_t i = 0; i < 10; i++) {
      plus[i] = get_param<double>(params, i);
      minus[i] = get_param<double>(params, i);
    }
    plus[n] += e;
    minus[n] -= e;
    
    double cdf_plus = TestClass.cdf
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5],plus[6],plus[7],plus[8],plus[9]);
    double cdf_minus = TestClass.cdf
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5],minus[6],minus[7],minus[8],minus[9]);
    
    finite_diff.push_back((cdf_plus - cdf_minus) / e2);
  }

  void calculate_finite_diff(const vector<double>& params, vector<double>& finite_diff) {
    if (!is_constant_struct<Scalar0>::value && !is_empty<Scalar0>::value)
      add_finite_diff(params, finite_diff, 0);
    if (!is_constant_struct<Scalar1>::value && !is_empty<Scalar1>::value)
      add_finite_diff(params, finite_diff, 1);
    if (!is_constant_struct<Scalar2>::value && !is_empty<Scalar2>::value)
      add_finite_diff(params, finite_diff, 2);
    if (!is_constant_struct<Scalar3>::value && !is_empty<Scalar3>::value)
      add_finite_diff(params, finite_diff, 3);
    if (!is_constant_struct<Scalar4>::value && !is_empty<Scalar4>::value)
      add_finite_diff(params, finite_diff, 4);
    if (!is_constant_struct<Scalar5>::value && !is_empty<Scalar5>::value)
      add_finite_diff(params, finite_diff, 5);
    if (!is_constant_struct<Scalar6>::value && !is_empty<Scalar6>::value)
      add_finite_diff(params, finite_diff, 6);
    if (!is_constant_struct<Scalar7>::value && !is_empty<Scalar7>::value)
      add_finite_diff(params, finite_diff, 7);
    if (!is_constant_struct<Scalar8>::value && !is_empty<Scalar8>::value)
      add_finite_diff(params, finite_diff, 8);
    if (!is_constant_struct<Scalar9>::value && !is_empty<Scalar9>::value)
      add_finite_diff(params, finite_diff, 9);
  }

  double calculate_gradients(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var cdf = TestClass.template cdf
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    cdf.grad(x, grad);
    return cdf.val();
  }
  
  double calculate_gradients_with_function(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var cdf = TestClass.template cdf_function
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    cdf.grad(x, grad);
    return cdf.val();
  }
  
  void test_finite_diff() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> finite_diffs;
      vector<double> gradients;

      calculate_finite_diff(parameters[n], finite_diffs);
      calculate_gradients(parameters[n], gradients);

      ASSERT_EQ(finite_diffs.size(), gradients.size()) 
        << "Number of finite diff gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < finite_diffs.size(); i++) {
        EXPECT_NEAR(finite_diffs[i], gradients[i], 1e-4)
          << "Comparison of finite diff to calculated gradient failed for i=" << i 
          << ": " << parameters[n] << std::endl 
          << "  finite diffs: " << finite_diffs << std::endl
          << "  grads:        " << gradients;
      }
    }
  }

  void test_gradient_function() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> expected_gradients;
      vector<double> gradients;

      double expected_cdf = calculate_gradients_with_function(parameters[n], expected_gradients);
      double cdf = calculate_gradients(parameters[n], gradients);

      EXPECT_FLOAT_EQ(expected_cdf, cdf);

      ASSERT_EQ(expected_gradients.size(), gradients.size()) 
        << "Number of expected gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < expected_gradients.size(); i++) {
        EXPECT_NEAR(expected_gradients[i], gradients[i], 1e-6)
          << "Comparison of expected gradient to calculated gradient failed";
      }
    }
  }

  void test_multiple_gradient_values(const bool is_vec,
                                     const double single_cdf,
                                     const vector<double>& single_gradients, size_t& pos_single,
                                     const vector<double>& multiple_gradients, size_t& pos_multiple,
                                     const size_t N_REPEAT) {
    if (is_vec) {
      for (size_t i = 0; i < N_REPEAT; i++) {
        EXPECT_FLOAT_EQ(single_gradients[pos_single] * pow(single_cdf, N_REPEAT-1),
                        multiple_gradients[pos_multiple])
          << "Comparison of single_gradient value to vectorized gradient failed";
        pos_multiple++;
      }
      pos_single++; 
    } else {
      EXPECT_FLOAT_EQ(N_REPEAT * single_gradients[pos_single] * pow(single_cdf, N_REPEAT-1), 
                      multiple_gradients[pos_multiple])
        << "Comparison of single_gradient value to vectorized gradient failed";
      pos_single++; pos_multiple++;
    }
  }

  void test_repeat_as_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> single_gradients;
      double single_cdf = calculate_gradients(parameters[n], single_gradients);
      
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);

      var multiple_cdf = TestClass.template cdf
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      vector<double> multiple_gradients;
      vector<var> x;
      add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
      multiple_cdf.grad(x, multiple_gradients);
      

      EXPECT_FLOAT_EQ(pow(single_cdf, N_REPEAT), multiple_cdf.val())
        << "cdf with repeated vector input should match "
        << "a multiple of cdf of single input";

      size_t pos_single = 0;
      size_t pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value)
        test_multiple_gradient_values(is_vector<T0>::value,
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value)
        test_multiple_gradient_values(is_vector<T1>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value)
        test_multiple_gradient_values(is_vector<T2>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value)
        test_multiple_gradient_values(is_vector<T3>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value)
        test_multiple_gradient_values(is_vector<T4>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,    
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value)
        test_multiple_gradient_values(is_vector<T5>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T6>::value && !is_empty<T6>::value)
        test_multiple_gradient_values(is_vector<T6>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T7>::value && !is_empty<T7>::value)
        test_multiple_gradient_values(is_vector<T7>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T8>::value && !is_empty<T8>::value)
        test_multiple_gradient_values(is_vector<T8>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T9>::value && !is_empty<T9>::value)
        test_multiple_gradient_values(is_vector<T9>::value, 
                                      single_cdf,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
    }
  }

  void test_lower_bound() {
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    if (!TestClass.has_lower_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = stan::agrad::value_of(std::numeric_limits<Scalar0>::min());
      } else {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = -std::numeric_limits<double>::infinity();
      }
    } else {
      for (size_t n = 0; n < parameters.size(); n++)
        parameters[n][0] = TestClass.lower_bound();
    }

    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);
      
      var cdf_at_lower_bound = TestClass.template cdf
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      EXPECT_FLOAT_EQ(0.0, cdf_at_lower_bound.val())
        << "CDF evaluated at lower bound should equal 0";
    }
  }
  
  void test_upper_bound() {
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);

    if (!TestClass.has_upper_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = stan::agrad::value_of(std::numeric_limits<Scalar0>::max());
      } else {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = std::numeric_limits<double>::infinity();
      }
    } else {
      for (size_t n = 0; n < parameters.size(); n++)
        parameters[n][0] = TestClass.upper_bound();
    }

    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);
      
      var cdf_at_upper_bound = TestClass.template cdf
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      EXPECT_FLOAT_EQ(1.0, cdf_at_upper_bound.val())
        << "CDF evaluated at upper bound should equal 1";
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 0;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);

    T0 p0 = get_repeated_params<T0>(parameters[0], 0, N_REPEAT);
    T1 p1 = get_repeated_params<T1>(parameters[0], 1, N_REPEAT);
    T2 p2 = get_repeated_params<T2>(parameters[0], 2, N_REPEAT);
    T3 p3 = get_repeated_params<T3>(parameters[0], 3, N_REPEAT);
    T4 p4 = get_repeated_params<T4>(parameters[0], 4, N_REPEAT);
    T5 p5 = get_repeated_params<T5>(parameters[0], 5, N_REPEAT);
    T6 p6 = get_repeated_params<T6>(parameters[0], 6, N_REPEAT);
    T7 p7 = get_repeated_params<T7>(parameters[0], 7, N_REPEAT);
    T8 p8 = get_repeated_params<T8>(parameters[0], 8, N_REPEAT);
    T9 p9 = get_repeated_params<T9>(parameters[0], 9, N_REPEAT);

    var cdf = TestClass.template cdf
      <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);

    EXPECT_FLOAT_EQ(1.0, cdf.val())
      << "cdf with an empty vector should return 1.0";
  }

  vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> cdf;

    TestClass.valid_values(params, cdf); 
    return params[0];
  }
};

TYPED_TEST_CASE_P(AgradCdfTestFixture);

TYPED_TEST_P(AgradCdfTestFixture, CallAllVersions) {
  this->call_all_versions();
}

TYPED_TEST_P(AgradCdfTestFixture, ValidValues) {
  this->test_valid_values();
}

TYPED_TEST_P(AgradCdfTestFixture, InvalidValues) {
  this->test_invalid_values();
}

TYPED_TEST_P(AgradCdfTestFixture, FiniteDiff) {
  this->test_finite_diff();
}

TYPED_TEST_P(AgradCdfTestFixture, Function) {
  this->test_gradient_function();
}

TYPED_TEST_P(AgradCdfTestFixture, RepeatAsVector) {
  this->test_repeat_as_vector();
}

TYPED_TEST_P(AgradCdfTestFixture, LowerBound) {
  this->test_lower_bound();
}

TYPED_TEST_P(AgradCdfTestFixture, UpperBound) {
  this->test_upper_bound();
}

TYPED_TEST_P(AgradCdfTestFixture, Length0Vector) {
  this->test_length_0_vector();
}

REGISTER_TYPED_TEST_CASE_P(AgradCdfTestFixture,
                           CallAllVersions,
                           ValidValues,
                           InvalidValues,
                           FiniteDiff,
                           Function,
                           RepeatAsVector,
                           LowerBound,
                           UpperBound,
                           Length0Vector);

template<class T>
class AgradCdfLogTestFixture : public ::testing::Test {
public:
  typename at_c<T,0>::type TestClass;
  typedef typename at_c<typename at_c<T,1>::type, 0>::type T0;
  typedef typename at_c<typename at_c<T,1>::type, 1>::type T1;
  typedef typename at_c<typename at_c<T,1>::type, 2>::type T2;
  typedef typename at_c<typename at_c<T,1>::type, 3>::type T3;
  typedef typename at_c<typename at_c<T,1>::type, 4>::type T4;
  typedef typename at_c<typename at_c<T,1>::type, 5>::type T5;
  typedef typename at_c<typename at_c<T,1>::type, 6>::type T6;
  typedef typename at_c<typename at_c<T,1>::type, 7>::type T7;
  typedef typename at_c<typename at_c<T,1>::type, 8>::type T8;
  typedef typename at_c<typename at_c<T,1>::type, 9>::type T9;

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;
  typedef typename scalar_type<T6>::type Scalar6;
  typedef typename scalar_type<T7>::type Scalar7;
  typedef typename scalar_type<T8>::type Scalar8;
  typedef typename scalar_type<T9>::type Scalar9;
  
  void call_all_versions() {
    vector<double> cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, cdf_log);
    
    T0 p0 = get_params<T0>(parameters, 0);
    T1 p1 = get_params<T1>(parameters, 1);
    T2 p2 = get_params<T2>(parameters, 2);
    T3 p3 = get_params<T3>(parameters, 3);
    T4 p4 = get_params<T4>(parameters, 4);
    T5 p5 = get_params<T5>(parameters, 5);
    T6 p6 = get_params<T6>(parameters, 6);
    T7 p7 = get_params<T7>(parameters, 7);
    T8 p8 = get_params<T8>(parameters, 8);
    T9 p9 = get_params<T9>(parameters, 9);
    
    EXPECT_NO_THROW(({ TestClass.template cdf_log
            <T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling cdf_log throws exception with default parameters";
  }

  void test_valid_values() {
    vector<double> expected_cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf_log);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_params<T0>(parameters, n, 0);
      T1 p1 = get_params<T1>(parameters, n, 1);
      T2 p2 = get_params<T2>(parameters, n, 2);
      T3 p3 = get_params<T3>(parameters, n, 3);
      T4 p4 = get_params<T4>(parameters, n, 4);
      T5 p5 = get_params<T5>(parameters, n, 5);
      T6 p6 = get_params<T6>(parameters, n, 6);
      T7 p7 = get_params<T7>(parameters, n, 7);
      T8 p8 = get_params<T8>(parameters, n, 8);
      T9 p9 = get_params<T9>(parameters, n, 9);

      var cdf_log(0);
      EXPECT_NO_THROW(({ cdf_log = TestClass.template cdf_log
              <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];
      EXPECT_TRUE(cdf_log.val() <= 0)
        << "cdf_log value must be less than or equal to 0. cdf_log value: " 
        << cdf_log;
      EXPECT_TRUE(cdf_log.val() <= 0)
        << "cdf_log value must be less than or equal to 0. cdf_log value: "
        << cdf_log;

      if (all_scalar<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
        EXPECT_FLOAT_EQ(expected_cdf_log[n], cdf_log.val())
          << "For all scalar inputs cdf_log should match the provided value. Failed at index: " << n;
      }
    }
  }

  void test_nan_value(const vector<double>& parameters, const size_t n) {
    var cdf_log(0);
    vector<double> invalid_params(parameters);
    invalid_params[n] = std::numeric_limits<double>::quiet_NaN();
    
    Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
    Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
    Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
    Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
    Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
    Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
    Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
    Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
    Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
    Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);
      
    EXPECT_THROW(({ TestClass.template cdf_log
            <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
            (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
      std::domain_error) 
      << "NaN value at index " << n << " should have failed" << std::endl
      << invalid_params;
  }

  void test_invalid_values() {
    vector<double> parameters = this->first_valid_params();
    
    vector<size_t> index;
    vector<double> invalid_values;
    TestClass.invalid_values(index, invalid_values);

    for (size_t n = 0; n < index.size(); n++) {
      var cdf_log(0);
      vector<double> invalid_params(parameters);
      invalid_params[index[n]] = invalid_values[n];
      
      Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
      Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
      Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
      Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
      Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
      Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
      Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
      Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
      Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
      Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);

      EXPECT_THROW(({ TestClass.template cdf_log
              <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
        std::domain_error) 
        << "Invalid value " << n << " should have failed" << std::endl
        << invalid_params;      
    }
    if (std::numeric_limits<Scalar0>::has_quiet_NaN && parameters.size() > 0) 
      test_nan_value(parameters, 0);
    if (std::numeric_limits<Scalar1>::has_quiet_NaN && parameters.size() > 1) 
      test_nan_value(parameters, 1);
    if (std::numeric_limits<Scalar2>::has_quiet_NaN && parameters.size() > 2) 
      test_nan_value(parameters, 2);
    if (std::numeric_limits<Scalar3>::has_quiet_NaN && parameters.size() > 3) 
      test_nan_value(parameters, 3);
    if (std::numeric_limits<Scalar4>::has_quiet_NaN && parameters.size() > 4) 
      test_nan_value(parameters, 4);
    if (std::numeric_limits<Scalar5>::has_quiet_NaN && parameters.size() > 5) 
      test_nan_value(parameters, 5);
    if (std::numeric_limits<Scalar6>::has_quiet_NaN && parameters.size() > 6) 
      test_nan_value(parameters, 6);
    if (std::numeric_limits<Scalar7>::has_quiet_NaN && parameters.size() > 7) 
      test_nan_value(parameters, 7);
    if (std::numeric_limits<Scalar8>::has_quiet_NaN && parameters.size() > 8) 
      test_nan_value(parameters, 8);
    if (std::numeric_limits<Scalar9>::has_quiet_NaN && parameters.size() > 9) 
      test_nan_value(parameters, 9);
  }

  void add_finite_diff(const vector<double>& params, 
                       vector<double>& finite_diff, 
                       const size_t n) {
    const double e = 1e-8;
    const double e2 = 2 * e;

    vector<double> plus(10);
    vector<double> minus(10);
    for (size_t i = 0; i < 10; i++) {
      plus[i] = get_param<double>(params, i);
      minus[i] = get_param<double>(params, i);
    }
    plus[n] += e;
    minus[n] -= e;
    
    double cdf_log_plus = TestClass.cdf_log
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5],plus[6],plus[7],plus[8],plus[9]);
    double cdf_log_minus = TestClass.cdf_log
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5],minus[6],minus[7],minus[8],minus[9]);
    
    finite_diff.push_back((cdf_log_plus - cdf_log_minus) / e2);
  }

  void calculate_finite_diff(const vector<double>& params, vector<double>& finite_diff) {
    if (!is_constant_struct<Scalar0>::value && !is_empty<Scalar0>::value)
      add_finite_diff(params, finite_diff, 0);
    if (!is_constant_struct<Scalar1>::value && !is_empty<Scalar1>::value)
      add_finite_diff(params, finite_diff, 1);
    if (!is_constant_struct<Scalar2>::value && !is_empty<Scalar2>::value)
      add_finite_diff(params, finite_diff, 2);
    if (!is_constant_struct<Scalar3>::value && !is_empty<Scalar3>::value)
      add_finite_diff(params, finite_diff, 3);
    if (!is_constant_struct<Scalar4>::value && !is_empty<Scalar4>::value)
      add_finite_diff(params, finite_diff, 4);
    if (!is_constant_struct<Scalar5>::value && !is_empty<Scalar5>::value)
      add_finite_diff(params, finite_diff, 5);
    if (!is_constant_struct<Scalar6>::value && !is_empty<Scalar6>::value)
      add_finite_diff(params, finite_diff, 6);
    if (!is_constant_struct<Scalar7>::value && !is_empty<Scalar7>::value)
      add_finite_diff(params, finite_diff, 7);
    if (!is_constant_struct<Scalar8>::value && !is_empty<Scalar8>::value)
      add_finite_diff(params, finite_diff, 8);
    if (!is_constant_struct<Scalar9>::value && !is_empty<Scalar9>::value)
      add_finite_diff(params, finite_diff, 9);
  }

  double calculate_gradients(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var cdf_log = TestClass.template cdf_log
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    cdf_log.grad(x, grad);
    return cdf_log.val();
  }
  
  double calculate_gradients_with_function(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var cdf_log = TestClass.template cdf_log_function
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    cdf_log.grad(x, grad);
    return cdf_log.val();
  }
  
  void test_finite_diff() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> expected_cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf_log);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> finite_diffs;
      vector<double> gradients;

      calculate_finite_diff(parameters[n], finite_diffs);
      calculate_gradients(parameters[n], gradients);

      ASSERT_EQ(finite_diffs.size(), gradients.size()) 
        << "Number of finite diff gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < finite_diffs.size(); i++) {
        EXPECT_NEAR(finite_diffs[i], gradients[i], 1e-4)
          << "Comparison of finite diff to calculated gradient failed for i=" << i 
          << ": " << parameters[n] << std::endl 
          << "  finite diffs: " << finite_diffs << std::endl
          << "  grads:        " << gradients;
      }
    }
  }

  void test_gradient_function() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> expected_gradients;
      vector<double> gradients;

      double expected_cdf_log = calculate_gradients_with_function(parameters[n], expected_gradients);
      double cdf_log = calculate_gradients(parameters[n], gradients);

      EXPECT_FLOAT_EQ(expected_cdf_log, cdf_log);

      ASSERT_EQ(expected_gradients.size(), gradients.size()) 
        << "Number of expected gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < expected_gradients.size(); i++) {
        EXPECT_NEAR(expected_gradients[i], gradients[i], 1e-6)
          << "Comparison of expected gradient to calculated gradient failed";
      }
    }
  }

  void test_multiple_gradient_values(const bool is_vec,
                                     const double single_cdf_log,
                                     const vector<double>& single_gradients, size_t& pos_single,
                                     const vector<double>& multiple_gradients, size_t& pos_multiple,
                                     const size_t N_REPEAT) {
    if (is_vec) {
      for (size_t i = 0; i < N_REPEAT; i++) {
        EXPECT_FLOAT_EQ(single_gradients[pos_single],
                        multiple_gradients[pos_multiple])
          << "Comparison of single_gradient value to vectorized gradient failed";
        pos_multiple++;
      }
      pos_single++; 
    } else {
      EXPECT_FLOAT_EQ(N_REPEAT * single_gradients[pos_single], 
                      multiple_gradients[pos_multiple])
        << "Comparison of single_gradient value to vectorized gradient failed";
      pos_single++; pos_multiple++;
    }
  }

  void test_repeat_as_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf_log);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> single_gradients;
      double single_cdf_log = calculate_gradients(parameters[n], single_gradients);
      
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);

      var multiple_cdf_log = TestClass.template cdf_log
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      vector<double> multiple_gradients;
      vector<var> x;
      add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
      multiple_cdf_log.grad(x, multiple_gradients);
      

      EXPECT_FLOAT_EQ(single_cdf_log * N_REPEAT, multiple_cdf_log.val())
        << "cdf_log with repeated vector input should match "
        << "a multiple of cdf_log of single input";

      size_t pos_single = 0;
      size_t pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value)
        test_multiple_gradient_values(is_vector<T0>::value,
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value)
        test_multiple_gradient_values(is_vector<T1>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value)
        test_multiple_gradient_values(is_vector<T2>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value)
        test_multiple_gradient_values(is_vector<T3>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value)
        test_multiple_gradient_values(is_vector<T4>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,    
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value)
        test_multiple_gradient_values(is_vector<T5>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T6>::value && !is_empty<T6>::value)
        test_multiple_gradient_values(is_vector<T6>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T7>::value && !is_empty<T7>::value)
        test_multiple_gradient_values(is_vector<T7>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T8>::value && !is_empty<T8>::value)
        test_multiple_gradient_values(is_vector<T8>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T9>::value && !is_empty<T9>::value)
        test_multiple_gradient_values(is_vector<T9>::value, 
                                      single_cdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
    }
  }

  void test_lower_bound() {
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf_log);
    
    if (!TestClass.has_lower_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = stan::agrad::value_of(std::numeric_limits<Scalar0>::min());
      } else {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = -std::numeric_limits<double>::infinity();
      }
    } else {
      for (size_t n = 0; n < parameters.size(); n++)
        parameters[n][0] = TestClass.lower_bound();
    }

    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);
      
      var cdf_log_at_lower_bound = TestClass.template cdf_log
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      EXPECT_FLOAT_EQ(stan::math::negative_infinity(), cdf_log_at_lower_bound.val())
        << "cdf_log evaluated at lower bound should equal negative infinity";
    }
  }
  
  void test_upper_bound() {
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf_log);

    if (!TestClass.has_upper_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = stan::agrad::value_of(std::numeric_limits<Scalar0>::max());
      } else {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = std::numeric_limits<double>::infinity();
      }
    } else {
      for (size_t n = 0; n < parameters.size(); n++)
        parameters[n][0] = TestClass.upper_bound();
    }

    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);
      
      var cdf_log_at_upper_bound = TestClass.template cdf_log
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      EXPECT_FLOAT_EQ(0.0, cdf_log_at_upper_bound.val())
        << "cdf_log evaluated at upper bound should equal 0";
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 0;
    vector<double> expected_cdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf_log);

    T0 p0 = get_repeated_params<T0>(parameters[0], 0, N_REPEAT);
    T1 p1 = get_repeated_params<T1>(parameters[0], 1, N_REPEAT);
    T2 p2 = get_repeated_params<T2>(parameters[0], 2, N_REPEAT);
    T3 p3 = get_repeated_params<T3>(parameters[0], 3, N_REPEAT);
    T4 p4 = get_repeated_params<T4>(parameters[0], 4, N_REPEAT);
    T5 p5 = get_repeated_params<T5>(parameters[0], 5, N_REPEAT);
    T6 p6 = get_repeated_params<T6>(parameters[0], 6, N_REPEAT);
    T7 p7 = get_repeated_params<T7>(parameters[0], 7, N_REPEAT);
    T8 p8 = get_repeated_params<T8>(parameters[0], 8, N_REPEAT);
    T9 p9 = get_repeated_params<T9>(parameters[0], 9, N_REPEAT);

    var cdf_log = TestClass.template cdf_log
      <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);

    EXPECT_FLOAT_EQ(0.0, cdf_log.val())
      << "cdf_log with an empty vector should return 0.0";
  }

  vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> cdf_log;

    TestClass.valid_values(params, cdf_log); 
    return params[0];
  }
};

TYPED_TEST_CASE_P(AgradCdfLogTestFixture);

TYPED_TEST_P(AgradCdfLogTestFixture, CallAllVersions) {
  this->call_all_versions();
}

TYPED_TEST_P(AgradCdfLogTestFixture, ValidValues) {
  this->test_valid_values();
}

TYPED_TEST_P(AgradCdfLogTestFixture, InvalidValues) {
  this->test_invalid_values();
}

TYPED_TEST_P(AgradCdfLogTestFixture, FiniteDiff) {
  this->test_finite_diff();
}

TYPED_TEST_P(AgradCdfLogTestFixture, Function) {
  this->test_gradient_function();
}

TYPED_TEST_P(AgradCdfLogTestFixture, RepeatAsVector) {
  this->test_repeat_as_vector();
}

TYPED_TEST_P(AgradCdfLogTestFixture, LowerBound) {
  this->test_lower_bound();
}

TYPED_TEST_P(AgradCdfLogTestFixture, UpperBound) {
  this->test_upper_bound();
}

TYPED_TEST_P(AgradCdfLogTestFixture, Length0Vector) {
  this->test_length_0_vector();
}

REGISTER_TYPED_TEST_CASE_P(AgradCdfLogTestFixture,
                           CallAllVersions,
                           ValidValues,
                           InvalidValues,
                           FiniteDiff,
                           Function,
                           RepeatAsVector,
                           LowerBound,
                           UpperBound,
                           Length0Vector);

template<class T>
class AgradCcdfLogTestFixture : public ::testing::Test {
public:
  typename at_c<T,0>::type TestClass;
  typedef typename at_c<typename at_c<T,1>::type, 0>::type T0;
  typedef typename at_c<typename at_c<T,1>::type, 1>::type T1;
  typedef typename at_c<typename at_c<T,1>::type, 2>::type T2;
  typedef typename at_c<typename at_c<T,1>::type, 3>::type T3;
  typedef typename at_c<typename at_c<T,1>::type, 4>::type T4;
  typedef typename at_c<typename at_c<T,1>::type, 5>::type T5;
  typedef typename at_c<typename at_c<T,1>::type, 6>::type T6;
  typedef typename at_c<typename at_c<T,1>::type, 7>::type T7;
  typedef typename at_c<typename at_c<T,1>::type, 8>::type T8;
  typedef typename at_c<typename at_c<T,1>::type, 9>::type T9;

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;
  typedef typename scalar_type<T6>::type Scalar6;
  typedef typename scalar_type<T7>::type Scalar7;
  typedef typename scalar_type<T8>::type Scalar8;
  typedef typename scalar_type<T9>::type Scalar9;
  
  void call_all_versions() {
    vector<double> ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, ccdf_log);
    
    T0 p0 = get_params<T0>(parameters, 0);
    T1 p1 = get_params<T1>(parameters, 1);
    T2 p2 = get_params<T2>(parameters, 2);
    T3 p3 = get_params<T3>(parameters, 3);
    T4 p4 = get_params<T4>(parameters, 4);
    T5 p5 = get_params<T5>(parameters, 5);
    T6 p6 = get_params<T6>(parameters, 6);
    T7 p7 = get_params<T7>(parameters, 7);
    T8 p8 = get_params<T8>(parameters, 8);
    T9 p9 = get_params<T9>(parameters, 9);
    
    EXPECT_NO_THROW(({ TestClass.template ccdf_log
            <T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>
            (p0, p1, p2, p3, p4, p5, p6, p7, p8, p9); }))
      << "Calling ccdf_log throws exception with default parameters";
  }

  void test_valid_values() {
    vector<double> expected_ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_ccdf_log);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_params<T0>(parameters, n, 0);
      T1 p1 = get_params<T1>(parameters, n, 1);
      T2 p2 = get_params<T2>(parameters, n, 2);
      T3 p3 = get_params<T3>(parameters, n, 3);
      T4 p4 = get_params<T4>(parameters, n, 4);
      T5 p5 = get_params<T5>(parameters, n, 5);
      T6 p6 = get_params<T6>(parameters, n, 6);
      T7 p7 = get_params<T7>(parameters, n, 7);
      T8 p8 = get_params<T8>(parameters, n, 8);
      T9 p9 = get_params<T9>(parameters, n, 9);

      var ccdf_log(0);
      EXPECT_NO_THROW(({ ccdf_log = TestClass.template ccdf_log
              <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];
      EXPECT_TRUE(ccdf_log.val() <= 0)
        << "ccdf_log value must be less than or equal to 0. ccdf_log value: " 
        << ccdf_log;
      EXPECT_TRUE(ccdf_log.val() <= 0)
        << "ccdf_log value must be less than or equal to 0. ccdf_log value: "
        << ccdf_log;

      if (all_scalar<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
        EXPECT_FLOAT_EQ(expected_ccdf_log[n], ccdf_log.val())
          << "For all scalar inputs ccdf_log should match the provided value. Failed at index: " << n;
      }
    }
  }

  void test_nan_value(const vector<double>& parameters, const size_t n) {
    var ccdf_log(0);
    vector<double> invalid_params(parameters);
    invalid_params[n] = std::numeric_limits<double>::quiet_NaN();
    
    Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
    Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
    Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
    Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
    Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
    Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
    Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
    Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
    Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
    Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);
      
    EXPECT_THROW(({ TestClass.template ccdf_log
            <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
            (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
      std::domain_error) 
      << "NaN value at index " << n << " should have failed" << std::endl
      << invalid_params;
  }

  void test_invalid_values() {
    vector<double> parameters = this->first_valid_params();
    
    vector<size_t> index;
    vector<double> invalid_values;
    TestClass.invalid_values(index, invalid_values);

    for (size_t n = 0; n < index.size(); n++) {
      var ccdf_log(0);
      vector<double> invalid_params(parameters);
      invalid_params[index[n]] = invalid_values[n];
      
      Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
      Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
      Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
      Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
      Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
      Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
      Scalar6 p6 = get_param<Scalar6>(invalid_params, 6);
      Scalar7 p7 = get_param<Scalar7>(invalid_params, 7);
      Scalar8 p8 = get_param<Scalar8>(invalid_params, 8);
      Scalar9 p9 = get_param<Scalar9>(invalid_params, 9);

      EXPECT_THROW(({ TestClass.template ccdf_log
              <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
              (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9); }),
        std::domain_error) 
        << "Invalid value " << n << " should have failed" << std::endl
        << invalid_params;      
    }
    if (std::numeric_limits<Scalar0>::has_quiet_NaN && parameters.size() > 0) 
      test_nan_value(parameters, 0);
    if (std::numeric_limits<Scalar1>::has_quiet_NaN && parameters.size() > 1) 
      test_nan_value(parameters, 1);
    if (std::numeric_limits<Scalar2>::has_quiet_NaN && parameters.size() > 2) 
      test_nan_value(parameters, 2);
    if (std::numeric_limits<Scalar3>::has_quiet_NaN && parameters.size() > 3) 
      test_nan_value(parameters, 3);
    if (std::numeric_limits<Scalar4>::has_quiet_NaN && parameters.size() > 4) 
      test_nan_value(parameters, 4);
    if (std::numeric_limits<Scalar5>::has_quiet_NaN && parameters.size() > 5) 
      test_nan_value(parameters, 5);
    if (std::numeric_limits<Scalar6>::has_quiet_NaN && parameters.size() > 6) 
      test_nan_value(parameters, 6);
    if (std::numeric_limits<Scalar7>::has_quiet_NaN && parameters.size() > 7) 
      test_nan_value(parameters, 7);
    if (std::numeric_limits<Scalar8>::has_quiet_NaN && parameters.size() > 8) 
      test_nan_value(parameters, 8);
    if (std::numeric_limits<Scalar9>::has_quiet_NaN && parameters.size() > 9) 
      test_nan_value(parameters, 9);
  }

  void add_finite_diff(const vector<double>& params, 
                       vector<double>& finite_diff, 
                       const size_t n) {
    const double e = 1e-8;
    const double e2 = 2 * e;

    vector<double> plus(10);
    vector<double> minus(10);
    for (size_t i = 0; i < 10; i++) {
      plus[i] = get_param<double>(params, i);
      minus[i] = get_param<double>(params, i);
    }
    plus[n] += e;
    minus[n] -= e;
    
    double ccdf_log_plus = TestClass.ccdf_log
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5],plus[6],plus[7],plus[8],plus[9]);
    double ccdf_log_minus = TestClass.ccdf_log
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5],minus[6],minus[7],minus[8],minus[9]);
    
    finite_diff.push_back((ccdf_log_plus - ccdf_log_minus) / e2);
  }

  void calculate_finite_diff(const vector<double>& params, vector<double>& finite_diff) {
    if (!is_constant_struct<Scalar0>::value && !is_empty<Scalar0>::value)
      add_finite_diff(params, finite_diff, 0);
    if (!is_constant_struct<Scalar1>::value && !is_empty<Scalar1>::value)
      add_finite_diff(params, finite_diff, 1);
    if (!is_constant_struct<Scalar2>::value && !is_empty<Scalar2>::value)
      add_finite_diff(params, finite_diff, 2);
    if (!is_constant_struct<Scalar3>::value && !is_empty<Scalar3>::value)
      add_finite_diff(params, finite_diff, 3);
    if (!is_constant_struct<Scalar4>::value && !is_empty<Scalar4>::value)
      add_finite_diff(params, finite_diff, 4);
    if (!is_constant_struct<Scalar5>::value && !is_empty<Scalar5>::value)
      add_finite_diff(params, finite_diff, 5);
    if (!is_constant_struct<Scalar6>::value && !is_empty<Scalar6>::value)
      add_finite_diff(params, finite_diff, 6);
    if (!is_constant_struct<Scalar7>::value && !is_empty<Scalar7>::value)
      add_finite_diff(params, finite_diff, 7);
    if (!is_constant_struct<Scalar8>::value && !is_empty<Scalar8>::value)
      add_finite_diff(params, finite_diff, 8);
    if (!is_constant_struct<Scalar9>::value && !is_empty<Scalar9>::value)
      add_finite_diff(params, finite_diff, 9);
  }

  double calculate_gradients(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var ccdf_log = TestClass.template ccdf_log
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    ccdf_log.grad(x, grad);
    return ccdf_log.val();
  }
  
  double calculate_gradients_with_function(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    Scalar6 p6 = get_param<Scalar6>(params, 6);
    Scalar7 p7 = get_param<Scalar7>(params, 7);
    Scalar8 p8 = get_param<Scalar8>(params, 8);
    Scalar9 p9 = get_param<Scalar9>(params, 9);
    
    var ccdf_log = TestClass.template ccdf_log_function
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5,Scalar6,Scalar7,Scalar8,Scalar9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
    ccdf_log.grad(x, grad);
    return ccdf_log.val();
  }
  
  void test_finite_diff() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> expected_ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_ccdf_log);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> finite_diffs;
      vector<double> gradients;

      calculate_finite_diff(parameters[n], finite_diffs);
      calculate_gradients(parameters[n], gradients);

      ASSERT_EQ(finite_diffs.size(), gradients.size()) 
        << "Number of finite diff gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < finite_diffs.size(); i++) {
        EXPECT_NEAR(finite_diffs[i], gradients[i], 1e-4)
          << "Comparison of finite diff to calculated gradient failed for i=" << i 
          << ": " << parameters[n] << std::endl 
          << "  finite diffs: " << finite_diffs << std::endl
          << "  grads:        " << gradients;
      }
    }
  }

  void test_gradient_function() {
    if (all_constant<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> expected_gradients;
      vector<double> gradients;

      double expected_ccdf_log = calculate_gradients_with_function(parameters[n], expected_gradients);
      double ccdf_log = calculate_gradients(parameters[n], gradients);

      EXPECT_FLOAT_EQ(expected_ccdf_log, ccdf_log);

      ASSERT_EQ(expected_gradients.size(), gradients.size()) 
        << "Number of expected gradients and calculated gradients must match -- error in test fixture";
      for (size_t i = 0; i < expected_gradients.size(); i++) {
        EXPECT_NEAR(expected_gradients[i], gradients[i], 1e-6)
          << "Comparison of expected gradient to calculated gradient failed";
      }
    }
  }

  void test_multiple_gradient_values(const bool is_vec,
                                     const double single_ccdf_log,
                                     const vector<double>& single_gradients, size_t& pos_single,
                                     const vector<double>& multiple_gradients, size_t& pos_multiple,
                                     const size_t N_REPEAT) {
    if (is_vec) {
      for (size_t i = 0; i < N_REPEAT; i++) {
        EXPECT_FLOAT_EQ(single_gradients[pos_single],
                        multiple_gradients[pos_multiple])
          << "Comparison of single_gradient value to vectorized gradient failed";
        pos_multiple++;
      }
      pos_single++; 
    } else {
      EXPECT_FLOAT_EQ(N_REPEAT * single_gradients[pos_single], 
                      multiple_gradients[pos_multiple])
        << "Comparison of single_gradient value to vectorized gradient failed";
      pos_single++; pos_multiple++;
    }
  }

  void test_repeat_as_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 3;
    vector<double> expected_ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_ccdf_log);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> single_gradients;
      double single_ccdf_log = calculate_gradients(parameters[n], single_gradients);
      
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);

      var multiple_ccdf_log = TestClass.template ccdf_log
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      vector<double> multiple_gradients;
      vector<var> x;
      add_vars(x, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
      multiple_ccdf_log.grad(x, multiple_gradients);
      

      EXPECT_FLOAT_EQ(single_ccdf_log * N_REPEAT, multiple_ccdf_log.val())
        << "ccdf_log with repeated vector input should match "
        << "a multiple of ccdf_log of single input";

      size_t pos_single = 0;
      size_t pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value)
        test_multiple_gradient_values(is_vector<T0>::value,
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value)
        test_multiple_gradient_values(is_vector<T1>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value)
        test_multiple_gradient_values(is_vector<T2>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value)
        test_multiple_gradient_values(is_vector<T3>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value)
        test_multiple_gradient_values(is_vector<T4>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,    
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value)
        test_multiple_gradient_values(is_vector<T5>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T6>::value && !is_empty<T6>::value)
        test_multiple_gradient_values(is_vector<T6>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T7>::value && !is_empty<T7>::value)
        test_multiple_gradient_values(is_vector<T7>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T8>::value && !is_empty<T8>::value)
        test_multiple_gradient_values(is_vector<T8>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T9>::value && !is_empty<T9>::value)
        test_multiple_gradient_values(is_vector<T9>::value, 
                                      single_ccdf_log,
                                      single_gradients, pos_single,
                                      multiple_gradients, pos_multiple,
                                      N_REPEAT);
    }
  }

  void test_lower_bound() {
    const size_t N_REPEAT = 3;
    vector<double> expected_ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_ccdf_log);
    
    if (!TestClass.has_lower_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = stan::agrad::value_of(std::numeric_limits<Scalar0>::min());
      } else {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = -std::numeric_limits<double>::infinity();
      }
    } else {
      for (size_t n = 0; n < parameters.size(); n++)
        parameters[n][0] = TestClass.lower_bound();
    }

    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);
      
      var ccdf_log_at_lower_bound = TestClass.template ccdf_log
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      EXPECT_FLOAT_EQ(0.0, ccdf_log_at_lower_bound.val())
        << "ccdf_log evaluated at lower bound should equal 0.0";
    }
  }
  
  void test_upper_bound() {
    const size_t N_REPEAT = 3;
    vector<double> expected_ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_ccdf_log);

    if (!TestClass.has_upper_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = stan::agrad::value_of(std::numeric_limits<Scalar0>::max());
      } else {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = std::numeric_limits<double>::infinity();
      }
    } else {
      for (size_t n = 0; n < parameters.size(); n++)
        parameters[n][0] = TestClass.upper_bound();
    }

    for (size_t n = 0; n < parameters.size(); n++) {
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);
      T6 p6 = get_repeated_params<T6>(parameters[n], 6, N_REPEAT);
      T7 p7 = get_repeated_params<T7>(parameters[n], 7, N_REPEAT);
      T8 p8 = get_repeated_params<T8>(parameters[n], 8, N_REPEAT);
      T9 p9 = get_repeated_params<T9>(parameters[n], 9, N_REPEAT);
      
      var ccdf_log_at_upper_bound = TestClass.template ccdf_log
        <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
        (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);
      EXPECT_FLOAT_EQ(stan::math::negative_infinity(), ccdf_log_at_upper_bound.val())
        << "ccdf_log evaluated at upper bound should equal negative infinity";
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 0;
    vector<double> expected_ccdf_log;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_ccdf_log);

    T0 p0 = get_repeated_params<T0>(parameters[0], 0, N_REPEAT);
    T1 p1 = get_repeated_params<T1>(parameters[0], 1, N_REPEAT);
    T2 p2 = get_repeated_params<T2>(parameters[0], 2, N_REPEAT);
    T3 p3 = get_repeated_params<T3>(parameters[0], 3, N_REPEAT);
    T4 p4 = get_repeated_params<T4>(parameters[0], 4, N_REPEAT);
    T5 p5 = get_repeated_params<T5>(parameters[0], 5, N_REPEAT);
    T6 p6 = get_repeated_params<T6>(parameters[0], 6, N_REPEAT);
    T7 p7 = get_repeated_params<T7>(parameters[0], 7, N_REPEAT);
    T8 p8 = get_repeated_params<T8>(parameters[0], 8, N_REPEAT);
    T9 p9 = get_repeated_params<T9>(parameters[0], 9, N_REPEAT);

    var ccdf_log = TestClass.template ccdf_log
      <T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>
      (p0,p1,p2,p3,p4,p5,p6,p7,p8,p9);

    EXPECT_FLOAT_EQ(0.0, ccdf_log.val())
      << "ccdf_log with an empty vector should return 0.0";
  }

  vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> ccdf_log;

    TestClass.valid_values(params, ccdf_log); 
    return params[0];
  }
};

TYPED_TEST_CASE_P(AgradCcdfLogTestFixture);

TYPED_TEST_P(AgradCcdfLogTestFixture, CallAllVersions) {
  this->call_all_versions();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, ValidValues) {
  this->test_valid_values();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, InvalidValues) {
  this->test_invalid_values();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, FiniteDiff) {
  this->test_finite_diff();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, Function) {
  this->test_gradient_function();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, RepeatAsVector) {
  this->test_repeat_as_vector();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, LowerBound) {
  this->test_lower_bound();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, UpperBound) {
  this->test_upper_bound();
}

TYPED_TEST_P(AgradCcdfLogTestFixture, Length0Vector) {
  this->test_length_0_vector();
}

REGISTER_TYPED_TEST_CASE_P(AgradCcdfLogTestFixture,
                           CallAllVersions,
                           ValidValues,
                           InvalidValues,
                           FiniteDiff,
                           Function,
                           RepeatAsVector,
                           LowerBound,
                           UpperBound,
                           Length0Vector);
#endif
