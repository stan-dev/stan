#ifndef __TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_CCDF_LOG_HPP___
#define __TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_CCDF_LOG_HPP___

#include <stdexcept>
#include <stan/math/error_handling.hpp>
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

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;
  
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
    
    EXPECT_NO_THROW(({ TestClass.template ccdf_log
            <T0, T1, T2, T3, T4, T5>
            (p0, p1, p2, p3, p4, p5); }))
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

      var ccdf_log(0);
      EXPECT_NO_THROW(({ ccdf_log = TestClass.template ccdf_log
              <T0,T1,T2,T3,T4,T5>
              (p0,p1,p2,p3,p4,p5); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];
      EXPECT_TRUE(ccdf_log.val() <= 0)
        << "ccdf_log value must be less than or equal to 0. ccdf_log value: " 
        << ccdf_log;
      EXPECT_TRUE(ccdf_log.val() <= 0)
        << "ccdf_log value must be less than or equal to 0. ccdf_log value: "
        << ccdf_log;

      if (all_scalar<T0,T1,T2,T3,T4,T5>::value) {
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
      
    EXPECT_THROW(({ TestClass.template ccdf_log
            <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
            (p0,p1,p2,p3,p4,p5); }),
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

      EXPECT_THROW(({ TestClass.template ccdf_log
              <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
              (p0,p1,p2,p3,p4,p5); }),
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
  }

  void add_finite_diff(const vector<double>& params, 
                       vector<double>& finite_diff, 
                       const size_t n) {
    const double e = 1e-8;
    const double e2 = 2 * e;

    vector<double> plus(10);
    vector<double> minus(10);
    for (size_t i = 0; i < 6; i++) {
      plus[i] = get_param<double>(params, i);
      minus[i] = get_param<double>(params, i);
    }
    plus[n] += e;
    minus[n] -= e;
    
    double ccdf_log_plus = TestClass.ccdf_log
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5]);
    double ccdf_log_minus = TestClass.ccdf_log
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5]);
    
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
  }

  double calculate_gradients(const vector<double>& params, vector<double>& grad) {
    Scalar0 p0 = get_param<Scalar0>(params, 0);
    Scalar1 p1 = get_param<Scalar1>(params, 1);
    Scalar2 p2 = get_param<Scalar2>(params, 2);
    Scalar3 p3 = get_param<Scalar3>(params, 3);
    Scalar4 p4 = get_param<Scalar4>(params, 4);
    Scalar5 p5 = get_param<Scalar5>(params, 5);
    
    var ccdf_log = TestClass.template ccdf_log
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
      (p0,p1,p2,p3,p4,p5);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5);
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
    
    var ccdf_log = TestClass.template ccdf_log_function
      <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
      (p0,p1,p2,p3,p4,p5);
    vector<var> x;
    add_vars(x, p0, p1, p2, p3, p4, p5);
    ccdf_log.grad(x, grad);
    return ccdf_log.val();
  }
  
  void test_finite_diff() {
    if (all_constant<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5>::value) {
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
    if (all_constant<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5>::value) {
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
    if (!any_vector<T0,T1,T2,T3,T4,T5>::value) {
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

      var multiple_ccdf_log = TestClass.template ccdf_log
        <T0,T1,T2,T3,T4,T5>
        (p0,p1,p2,p3,p4,p5);
      vector<double> multiple_gradients;
      vector<var> x;
      add_vars(x, p0, p1, p2, p3, p4, p5);
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
      
      var ccdf_log_at_lower_bound = TestClass.template ccdf_log
        <T0,T1,T2,T3,T4,T5>
        (p0,p1,p2,p3,p4,p5);
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
      
      var ccdf_log_at_upper_bound = TestClass.template ccdf_log
        <T0,T1,T2,T3,T4,T5>
        (p0,p1,p2,p3,p4,p5);
      EXPECT_FLOAT_EQ(stan::math::negative_infinity(), ccdf_log_at_upper_bound.val())
        << "ccdf_log evaluated at upper bound should equal negative infinity";
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5>::value) {
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

    var ccdf_log = TestClass.template ccdf_log
      <T0,T1,T2,T3,T4,T5>
      (p0,p1,p2,p3,p4,p5);

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
