#ifndef TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_CDF_HPP
#define TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_CDF_HPP

#include <stan/math/rev/core.hpp>
#include <stdexcept>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <test/prob/utility.hpp>
#include <boost/type_traits/is_same.hpp>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using stan::scalar_type;
using stan::is_vector;
using stan::is_constant;
using stan::is_constant_struct;
using stan::math::value_of;


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
    return stan::math::normal_cdf(y, mu, sigma);
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

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;
  
  typedef typename stan::math::fvar<typename stan::partials_return_type<T0,T1,T2,T3,T4,T5>::type> T_fvar_return;
  typedef typename stan::return_type<T0,T1,T2,T3,T4,T5>::type T_return_type;

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
    
    EXPECT_NO_THROW(({ TestClass.template cdf
            <T0, T1, T2, T3, T4, T5>
            (p0, p1, p2, p3, p4, p5); }))
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

      T_return_type cdf(0);
      EXPECT_NO_THROW(({ cdf = TestClass.template cdf
              <T0,T1,T2,T3,T4,T5>
              (p0,p1,p2,p3,p4,p5); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];
      EXPECT_TRUE(cdf >= 0)
        << "cdf value must be greater than or equal to 0. cdf value: " 
        << cdf;
      EXPECT_TRUE(cdf <= 1)
        << "cdf value must be less than or equal to 1. cdf value: "
        << cdf;

      if (all_scalar<T0,T1,T2,T3,T4,T5>::value) {
        EXPECT_TRUE(stan::math::abs(expected_cdf[n] - cdf) < 1e-8)
          << "For all scalar inputs cdf should match the provided value. Failed at index: " << n << std::endl
          << "expected: " << expected_cdf[n] << std::endl
          << "actual:   " << cdf;
      }
    }
  }

  void test_nan_value(const vector<double>& parameters, const size_t n) {
    vector<double> invalid_params(parameters);
    invalid_params[n] = std::numeric_limits<double>::quiet_NaN();
    
    Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
    Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
    Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
    Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
    Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
    Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);
      
    EXPECT_THROW(({ TestClass.template cdf
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
      vector<double> invalid_params(parameters);
      invalid_params[index[n]] = invalid_values[n];
      
      Scalar0 p0 = get_param<Scalar0>(invalid_params, 0);
      Scalar1 p1 = get_param<Scalar1>(invalid_params, 1);
      Scalar2 p2 = get_param<Scalar2>(invalid_params, 2);
      Scalar3 p3 = get_param<Scalar3>(invalid_params, 3);
      Scalar4 p4 = get_param<Scalar4>(invalid_params, 4);
      Scalar5 p5 = get_param<Scalar5>(invalid_params, 5);

      EXPECT_THROW(({ TestClass.template cdf
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

 void add_finite_diff_1storder(const vector<double>& params, 
                                vector<double>& finite_diff, 
                                const size_t n) {
    const double e = 1e-8;
    const double e2 = 2 * e;

    vector<double> plus(6);
    vector<double> minus(6);
    for (size_t i = 0; i < 6; i++) {
      plus[i] = get_param<double>(params, i);
      minus[i] = get_param<double>(params, i);
    }
    plus[n] += e;
    minus[n] -= e;
    
    double cdf_plus = TestClass.cdf
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5]);
    double cdf_minus = TestClass.cdf
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5]);

    finite_diff.push_back((cdf_plus - cdf_minus) / e2);
  }

   void calculate_finite_diff(const vector<double>& params, 
                              vector<double>& finite_diff) {
    if (!is_constant_struct<Scalar0>::value && !is_empty<Scalar0>::value)
      add_finite_diff_1storder(params, finite_diff, 0);
    if (!is_constant_struct<Scalar1>::value && !is_empty<Scalar1>::value)
      add_finite_diff_1storder(params, finite_diff, 1);
    if (!is_constant_struct<Scalar2>::value && !is_empty<Scalar2>::value)
      add_finite_diff_1storder(params, finite_diff, 2);
    if (!is_constant_struct<Scalar3>::value && !is_empty<Scalar3>::value)
      add_finite_diff_1storder(params, finite_diff, 3);
    if (!is_constant_struct<Scalar4>::value && !is_empty<Scalar4>::value)
      add_finite_diff_1storder(params, finite_diff, 4);
    if (!is_constant_struct<Scalar5>::value && !is_empty<Scalar5>::value)
      add_finite_diff_1storder(params, finite_diff, 5);
  }

   // works for <double>
  double calculate_gradients_1storder(vector<double>& grad,
                                      double& cdf,
                                      vector<var>& x) {
    return cdf;
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      double& cdf,
                                      vector<var>& x) {
    return cdf;
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      double& cdf, 
                                      vector<var>& x) {
    return cdf;
  }

  // works for <var>
  double calculate_gradients_1storder(vector<double>& grad,
                                      var& cdf,
                                      vector<var>& x) {
    cdf.grad(x, grad);
    stan::math::recover_memory();
    return cdf.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      var& cdf,
                                      vector<var>& x) {
    return cdf.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      var& cdf, 
                                      vector<var>& x) {
    return cdf.val();
  }

  //works for fvar<double>
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<double>& cdf,
                                      vector<var>& x) {
    x.push_back(cdf.d_);
    return cdf.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      fvar<double>& cdf, 
                                      vector<var>& x) {
    return cdf.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      fvar<double>& cdf, 
                                      vector<var>& x) {
    return cdf.val();
  }

  //works for fvar<fvar<double> >
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<fvar<double> >& cdf, 
                                      vector<var>& x) {
    x.push_back(cdf.d_.val_);
    return cdf.val().val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      fvar<fvar<double> >& cdf,
                                      vector<var>& x) {
    return cdf.val().val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      fvar<fvar<double> >& cdf, 
                                      vector<var>& x) {
    return cdf.val().val();
  }

  // works for fvar<var>
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<var>& cdf, 
                                      vector<var>& x) {
    cdf.val_.grad(x, grad);
    stan::math::recover_memory();
    return cdf.val_.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad, 
                                      fvar<var>& cdf, 
                                      vector<var>& x) {
    cdf.d_.grad(x, grad);
    stan::math::recover_memory();
    return cdf.val_.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad, 
                                      fvar<var>& cdf, 
                                      vector<var>& x) {
    return cdf.val_.val();
  }

  // works for fvar<fvar<var> >
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<fvar<var> >& cdf,
                                      vector<var>& x) {
    cdf.val_.val_.grad(x, grad);
    stan::math::recover_memory();
    return cdf.val_.val_.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      fvar<fvar<var> >& cdf, 
                                      vector<var>& x) {
    cdf.d_.val_.grad(x, grad);
    stan::math::recover_memory();
    return cdf.val_.val_.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      fvar<fvar<var> >& cdf, 
                                      vector<var>& x) {
    cdf.d_.d_.grad(x, grad);
    stan::math::recover_memory();
    return cdf.val_.val_.val();
  }

  void test_finite_diffs_equal(const vector<double>& parameters,
                               const vector<double>& finite_dif,
                               const vector<double>& gradients) {

    ASSERT_EQ(finite_dif.size(), gradients.size()) 
      << "Number of finite diff gradients and calculated gradients must match -- error in test fixture";
    for (size_t i = 0; i < finite_dif.size(); i++) {
      EXPECT_NEAR(finite_dif[i], gradients[i], 1e-4)
        << "Comparison of finite diff to calculated gradient failed for i=" << i 
        << ": " << parameters << std::endl 
        << "  finite diffs: " << finite_dif << std::endl
        << "  grads:        " << gradients;
      }
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
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> finite_diffs;
      vector<double> gradients;

      if (!boost::is_same<Scalar0, fvar<double> >::value &&
          !boost::is_same<Scalar0, fvar<fvar<double> > >::value &&
          !boost::is_same<Scalar1, fvar<double> >::value &&
          !boost::is_same<Scalar1, fvar<fvar<double> > >::value &&
          !boost::is_same<Scalar2, fvar<double> >::value &&
          !boost::is_same<Scalar2, fvar<fvar<double> > >::value &&
          !boost::is_same<Scalar3, fvar<double> >::value &&
          !boost::is_same<Scalar3, fvar<fvar<double> > >::value &&
          !boost::is_same<Scalar4, fvar<double> >::value &&
          !boost::is_same<Scalar4, fvar<fvar<double> > >::value &&
          !boost::is_same<Scalar5, fvar<double> >::value &&
          !boost::is_same<Scalar5, fvar<fvar<double> > >::value) {

        calculate_finite_diff(parameters[n], finite_diffs);
        Scalar0 p0 = get_param<Scalar0>(parameters[n], 0);
        Scalar1 p1 = get_param<Scalar1>(parameters[n], 1);
        Scalar2 p2 = get_param<Scalar2>(parameters[n], 2);
        Scalar3 p3 = get_param<Scalar3>(parameters[n], 3);
        Scalar4 p4 = get_param<Scalar4>(parameters[n], 4);
        Scalar5 p5 = get_param<Scalar5>(parameters[n], 5);
    
        vector<var> x1;
        add_vars(x1, p0, p1, p2, p3, p4, p5);
          
        T_return_type cdf = TestClass.template cdf
          <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
          (p0,p1,p2,p3,p4,p5);

        calculate_gradients_1storder(gradients, cdf, x1);

        test_finite_diffs_equal(parameters[n], finite_diffs, gradients);
      }
    }
  }

  void  test_gradients_equal(const vector<double>& expected_gradients,
                             const vector<double>& gradients ) {

    ASSERT_EQ(expected_gradients.size(), gradients.size()) 
      << "Number of expected gradients and calculated gradients must match -- error in test fixture";
    for (size_t i = 0; i < expected_gradients.size(); i++) {
      EXPECT_FLOAT_EQ(expected_gradients[i], gradients[i])
        << "Comparison of expected gradient to calculated gradient failed";
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
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> expected_gradients1;
      vector<double> expected_gradients2;
      vector<double> expected_gradients3;
      vector<double> gradients1;
      vector<double> gradients2;
      vector<double> gradients3;

      Scalar0 p0 = get_param<Scalar0>(parameters[n], 0);
      Scalar1 p1 = get_param<Scalar1>(parameters[n], 1);
      Scalar2 p2 = get_param<Scalar2>(parameters[n], 2);
      Scalar3 p3 = get_param<Scalar3>(parameters[n], 3);
      Scalar4 p4 = get_param<Scalar4>(parameters[n], 4);
      Scalar5 p5 = get_param<Scalar5>(parameters[n], 5);
    
      vector<var> x1;
      vector<var> x2;
      vector<var> x3;
      vector<var> y1;
      vector<var> y2;
      vector<var> y3;
      add_vars(x1, p0, p1, p2, p3, p4, p5);
      add_vars(x2, p0, p1, p2, p3, p4, p5);
      add_vars(x3, p0, p1, p2, p3, p4, p5);
      add_vars(y1, p0, p1, p2, p3, p4, p5);
      add_vars(y2, p0, p1, p2, p3, p4, p5);
      add_vars(y3, p0, p1, p2, p3, p4, p5);
      
          
      T_return_type cdf = TestClass.template cdf
        <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);

      T_return_type cdf_funct = TestClass.template cdf_function
        <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);

      calculate_gradients_1storder(expected_gradients1, cdf_funct, x1);
      calculate_gradients_1storder(gradients1, cdf, y1);
      calculate_gradients_2ndorder(expected_gradients2, cdf_funct, x2);
      calculate_gradients_2ndorder(gradients2, cdf, y2);
      calculate_gradients_3rdorder(expected_gradients3, cdf_funct, x3);
      calculate_gradients_3rdorder(gradients3, cdf, y3);
      
      test_gradients_equal(expected_gradients1,gradients1);
      test_gradients_equal(expected_gradients2,gradients2);
      test_gradients_equal(expected_gradients3,gradients3);
    }
  }

  void test_multiple_gradient_values(const bool is_vec,
                                     const double single_cdf,
                                     const vector<double>& single_gradients, 
                                     size_t& pos_single,
                                     const vector<double>& multiple_gradients, 
                                     size_t& pos_multiple,
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
    if (!any_vector<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }

    const size_t N_REPEAT = 3;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    for (size_t n = 0; n < parameters.size(); n++) {
      vector<double> single_gradients1;
      vector<double> single_gradients2;
      vector<double> single_gradients3;

      Scalar0 p0_ = get_param<Scalar0>(parameters[n], 0);
      Scalar1 p1_ = get_param<Scalar1>(parameters[n], 1);
      Scalar2 p2_ = get_param<Scalar2>(parameters[n], 2);
      Scalar3 p3_ = get_param<Scalar3>(parameters[n], 3);
      Scalar4 p4_ = get_param<Scalar4>(parameters[n], 4);
      Scalar5 p5_ = get_param<Scalar5>(parameters[n], 5);
      vector<var> s1;
      vector<var> s2;
      vector<var> s3;
      add_vars(s1,p0_,p1_,p2_,p3_,p4_,p5_);
      add_vars(s2,p0_,p1_,p2_,p3_,p4_,p5_);
      add_vars(s3,p0_,p1_,p2_,p3_,p4_,p5_);

      T_return_type cdf = TestClass.template cdf
        <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0_,p1_,p2_,p3_,p4_,p5_);

      double single_cdf = calculate_gradients_1storder(single_gradients1,cdf,s1);
      calculate_gradients_2ndorder(single_gradients2,cdf,s2);
      calculate_gradients_3rdorder(single_gradients3,cdf,s3);
      
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);

      T_return_type multiple_cdf = TestClass.template cdf
        <T0,T1,T2,T3,T4,T5>
        (p0,p1,p2,p3,p4,p5);
      vector<double> multiple_gradients1;
      vector<double> multiple_gradients2;
      vector<double> multiple_gradients3;
      vector<var> x1;
      vector<var> x2;
      vector<var> x3;
      add_vars(x1, p0, p1, p2, p3, p4, p5);
      add_vars(x2, p0, p1, p2, p3, p4, p5);
      add_vars(x3, p0, p1, p2, p3, p4, p5);

      calculate_gradients_1storder(multiple_gradients1,multiple_cdf,x1);
      calculate_gradients_1storder(multiple_gradients2,multiple_cdf,x1);
      calculate_gradients_1storder(multiple_gradients3,multiple_cdf,x1);
      
      EXPECT_TRUE(pow(single_cdf, N_REPEAT) - multiple_cdf < 1e-8)
        << "cdf with repeated vector input should match "
        << "a multiple of cdf of single input";


      size_t pos_single = 0;
      size_t pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value &&
          !boost::is_same<Scalar0, fvar<double> >::value &&
          !boost::is_same<Scalar0, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T0>::value, single_cdf,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value &&
          !boost::is_same<Scalar1, fvar<double> >::value &&
          !boost::is_same<Scalar1, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T1>::value, single_cdf,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value &&
          !boost::is_same<Scalar2, fvar<double> >::value &&
          !boost::is_same<Scalar2, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T2>::value, single_cdf,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value &&
          !boost::is_same<Scalar3, fvar<double> >::value &&
          !boost::is_same<Scalar3, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T3>::value, single_cdf, 
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value &&
          !boost::is_same<Scalar4, fvar<double> >::value &&
          !boost::is_same<Scalar4, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T4>::value, single_cdf, 
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value &&
          !boost::is_same<Scalar5, fvar<double> >::value &&
          !boost::is_same<Scalar5, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T5>::value, single_cdf, 
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);

      pos_single = 0;
      pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value &&
          (boost::is_same<Scalar0, fvar<var> >::value || 
           boost::is_same<Scalar0, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T0>::value, single_cdf, 
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value &&
          (boost::is_same<Scalar1, fvar<var> >::value || 
           boost::is_same<Scalar1, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T1>::value, single_cdf, 
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value &&
          (boost::is_same<Scalar2, fvar<var> >::value || 
           boost::is_same<Scalar2, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T2>::value, single_cdf, 
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value&&
          (boost::is_same<Scalar3, fvar<var> >::value || 
           boost::is_same<Scalar3, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T3>::value, single_cdf, 
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value &&
          (boost::is_same<Scalar4, fvar<var> >::value || 
           boost::is_same<Scalar4, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T4>::value, single_cdf, 
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value &&
          (boost::is_same<Scalar5, fvar<var> >::value || 
           boost::is_same<Scalar5, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T5>::value, single_cdf, 
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);

      pos_single = 0;
      pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value &&
          boost::is_same<Scalar0, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T0>::value, single_cdf, 
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value &&
          boost::is_same<Scalar1, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T1>::value, single_cdf, 
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value &&
          boost::is_same<Scalar2, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T2>::value, single_cdf, 
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value &&
          boost::is_same<Scalar3, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T3>::value, single_cdf, 
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value &&
          boost::is_same<Scalar4, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T4>::value, single_cdf, 
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value &&
          boost::is_same<Scalar5, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T5>::value, single_cdf, 
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
    }
  }

  void test_lower_bound() {
    using stan::math::value_of;
    using stan::math::value_of;
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);
    
    if (!TestClass.has_lower_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = value_of(value_of(value_of(std::numeric_limits<Scalar0>::min())));
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
      
      T_return_type cdf_at_lower_bound = TestClass.template cdf
        <T0,T1,T2,T3,T4,T5>
        (p0,p1,p2,p3,p4,p5);
      EXPECT_TRUE(0.0 == cdf_at_lower_bound)
        << "CDF evaluated at lower bound should equal 0";
    }
  }
  
  void test_upper_bound() {
    using stan::math::value_of;
    using stan::math::value_of;
    const size_t N_REPEAT = 3;
    vector<double> expected_cdf;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, expected_cdf);

    if (!TestClass.has_upper_bound()) {
      if (!std::numeric_limits<Scalar0>::has_infinity) {
        for (size_t n = 0; n < parameters.size(); n++)
          parameters[n][0] = value_of(value_of(value_of(std::numeric_limits<Scalar0>::max())));
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
      
      T_return_type cdf_at_upper_bound = TestClass.template cdf
        <T0,T1,T2,T3,T4,T5>
        (p0,p1,p2,p3,p4,p5);
      EXPECT_TRUE(1.0 == cdf_at_upper_bound)
        << "CDF evaluated at upper bound is " 
        << cdf_at_upper_bound
        <<" but should equal 1";
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5>::value) {
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

    T_return_type cdf = TestClass.template cdf
      <T0,T1,T2,T3,T4,T5>
      (p0,p1,p2,p3,p4,p5);

    EXPECT_TRUE(1.0 == cdf)
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

#endif
