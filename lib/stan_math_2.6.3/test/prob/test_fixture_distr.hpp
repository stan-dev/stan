#ifndef TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_DISTR_HPP
#define TEST__UNIT_DISTRIBUTION__TEST_FIXTURE_DISTR_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <test/prob/utility.hpp>
#include <boost/type_traits/is_same.hpp>
#include <stdexcept>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using stan::math::fvar;
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

  typedef typename scalar_type<T0>::type Scalar0;
  typedef typename scalar_type<T1>::type Scalar1;
  typedef typename scalar_type<T2>::type Scalar2;
  typedef typename scalar_type<T3>::type Scalar3;
  typedef typename scalar_type<T4>::type Scalar4;
  typedef typename scalar_type<T5>::type Scalar5;

  typedef typename stan::math::fvar<typename stan::partials_return_type<T0,T1,T2,T3,T4,T5>::type> T_fvar_return;
  typedef typename stan::return_type<T0,T1,T2,T3,T4,T5>::type T_return_type;

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

    EXPECT_NO_THROW(({ TestClass.template log_prob
            <T0, T1, T2, T3, T4, T5>
            (p0, p1, p2, p3, p4, p5); }))
      << "Calling log_prob throws exception with default parameters";

    EXPECT_NO_THROW(({ TestClass.template log_prob
            <true, T0, T1, T2, T3, T4, T5>
            (p0, p1, p2, p3, p4, p5); }))
      << "Calling log_prob throws exception with propto=true";

    EXPECT_NO_THROW(({ TestClass.template log_prob
            <false, T0, T1, T2, T3, T4, T5>
            (p0, p1, p2, p3, p4, p5); }))
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

      T_return_type lp(0);
      EXPECT_NO_THROW(({ lp = TestClass.template log_prob
              <true,T0,T1,T2,T3,T4,T5>
              (p0,p1,p2,p3,p4,p5); }))
        << "Valid parameters failed at index: " << n << " -- " 
        << parameters[n];

      if (all_constant<T0,T1,T2,T3,T4,T5>::value) {
        // all double inputs should result in a log probability of 0
        EXPECT_TRUE(lp == 0.0)
          << "All constant inputs should result in 0 log probability. Failed at index: " << n;
      }
      if (all_scalar<T0,T1,T2,T3,T4,T5>::value) {
        lp = TestClass.template log_prob
          <false,T0,T1,T2,T3,T4,T5>
          (p0,p1,p2,p3,p4,p5);
        EXPECT_TRUE(stan::math::abs(lp - log_prob[n]) < 1e-8)
          << "For all scalar inputs, when propto is false, log_prob should match the provided value. Failed at index: "
          << n << std::endl
          << "expected: " << log_prob[n] << std::endl
          << "actual:   " << lp;
      }
      if (all_constant<T0,T1,T2,T3,T4,T5>::value
          && all_scalar<T0,T1,T2,T3,T4,T5>::value) {
        lp = TestClass.template log_prob
          <T0,T1,T2,T3,T4,T5>
          (p0,p1,p2,p3,p4,p5);
        EXPECT_TRUE(stan::math::abs(lp - log_prob[n]) < 1e-8)
          << "For all scalar and all constant inputs log_prob should match the provided value. Failed at index: "
          << n << std::endl
          << "expected: " << log_prob[n] << std::endl
          << "actual:   " << lp;
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
      
    EXPECT_THROW(({ TestClass.template log_prob
            <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
            (p0,p1,p2,p3,p4,p5); }),
      std::domain_error) 
      << "NaN value at index " << n << " should have failed" << std::endl
      << invalid_params;
  }
  
  void test_invalid_values() {
    if (!all_scalar<T0,T1,T2,T3,T4,T5>::value)
      return;

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

      EXPECT_THROW(({ TestClass.template log_prob
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

  void test_propto() {
    if (all_constant<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (any_vector<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << " No test for vector arguments";
      return;
    }
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    T_return_type reference_logprob_true;
    T_return_type reference_logprob_false;
    {
      Scalar0 p0 = get_param<Scalar0>(parameters[0], 0);
      Scalar1 p1 = get_param<Scalar1>(parameters[0], 1);
      Scalar2 p2 = get_param<Scalar2>(parameters[0], 2);
      Scalar3 p3 = get_param<Scalar3>(parameters[0], 3);
      Scalar4 p4 = get_param<Scalar4>(parameters[0], 4);
      Scalar5 p5 = get_param<Scalar5>(parameters[0], 5);

      reference_logprob_true 
        = TestClass.template log_prob
        <true,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);
      reference_logprob_false 
        = TestClass.template log_prob
        <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);
    }
    
    for (size_t n = 0; n < parameters.size(); n++) {
      Scalar0 p0 = select_var_param<T0>(parameters, n, 0);
      Scalar1 p1 = select_var_param<T1>(parameters, n, 1);
      Scalar2 p2 = select_var_param<T2>(parameters, n, 2);
      Scalar3 p3 = select_var_param<T3>(parameters, n, 3);
      Scalar4 p4 = select_var_param<T4>(parameters, n, 4);
      Scalar5 p5 = select_var_param<T5>(parameters, n, 5);

      T_return_type logprob_true
        = TestClass.template log_prob
        <true,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);
      T_return_type logprob_false
        = TestClass.template log_prob
        <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);

      EXPECT_TRUE(reference_logprob_false - logprob_false ==
                  reference_logprob_true - logprob_true)
        << "Proportional test failed at index: " << n << std::endl
        << "  reference params: " << parameters[0] << std::endl
        << "  current params:   " << parameters[n] << std::endl
        << "  ref<true> = " << reference_logprob_true << std::endl
        << "  cur<true> = " << logprob_true << std::endl
        << "  ref<false> = " << reference_logprob_false << std::endl
        << "  cur<false> = " << logprob_false;
    }
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
    
    double lp_plus = TestClass.log_prob
      (plus[0],plus[1],plus[2],plus[3],plus[4],plus[5]);
    double lp_minus = TestClass.log_prob
      (minus[0],minus[1],minus[2],minus[3],minus[4],minus[5]);
    
    finite_diff.push_back((lp_plus - lp_minus) / e2);
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
                                      double& logprob,
                                      vector<var>& x) {
    return logprob;
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      double& logprob,
                                      vector<var>& x) {
    return logprob;
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      double& logprob, 
                                      vector<var>& x) {
    return logprob;
  }

  // works for <var>
  double calculate_gradients_1storder(vector<double>& grad,
                                      var& logprob,
                                      vector<var>& x) {
    logprob.grad(x, grad);
    stan::math::recover_memory();
    return logprob.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      var& logprob,
                                      vector<var>& x) {
    return logprob.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      var& logprob, 
                                      vector<var>& x) {
    return logprob.val();
  }

  //works for fvar<double>
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<double>& logprob,
                                      vector<var>& x) {
    x.push_back(logprob.d_);
    return logprob.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      fvar<double>& logprob, 
                                      vector<var>& x) {
    return logprob.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      fvar<double>& logprob, 
                                      vector<var>& x) {
    return logprob.val();
  }

  //works for fvar<fvar<double> >
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<fvar<double> >& logprob, 
                                      vector<var>& x) {
    x.push_back(logprob.d_.val_);
    return logprob.val().val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      fvar<fvar<double> >& logprob,
                                      vector<var>& x) {
    return logprob.val().val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      fvar<fvar<double> >& logprob, 
                                      vector<var>& x) {
    return logprob.val().val();
  }

  // works for fvar<var>
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<var>& logprob, 
                                      vector<var>& x) {
    logprob.val_.grad(x, grad);
    stan::math::recover_memory();
    return logprob.val_.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad, 
                                      fvar<var>& logprob, 
                                      vector<var>& x) {
    logprob.d_.grad(x, grad);
    stan::math::recover_memory();
    return logprob.val_.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad, 
                                      fvar<var>& logprob, 
                                      vector<var>& x) {
    return logprob.val_.val();
  }

  // works for fvar<fvar<var> >
  double calculate_gradients_1storder(vector<double>& grad,
                                      fvar<fvar<var> >& logprob,
                                      vector<var>& x) {
    logprob.val_.val_.grad(x, grad);
    stan::math::recover_memory();
    return logprob.val_.val_.val();
  }
  double calculate_gradients_2ndorder(vector<double>& grad,
                                      fvar<fvar<var> >& logprob, 
                                      vector<var>& x) {
    logprob.d_.val_.grad(x, grad);
    stan::math::recover_memory();
    return logprob.val_.val_.val();
  }
  double calculate_gradients_3rdorder(vector<double>& grad,
                                      fvar<fvar<var> >& logprob, 
                                      vector<var>& x) {
    logprob.d_.d_.grad(x, grad);
    stan::math::recover_memory();
    return logprob.val_.val_.val();
  }

  void  test_finite_diffs_equal(const vector<double>& parameters,
                                const vector<double>& finite_diffs,
                                const vector<double>& gradients ) {

    ASSERT_EQ(finite_diffs.size(), gradients.size()) 
      << "Number of first order finite diff gradients and calculated gradients must match -- error in test fixture";
    for (size_t i = 0; i < finite_diffs.size(); i++) {
      EXPECT_NEAR(finite_diffs[i], gradients[i], 1e-4)
        << "Comparison of first order finite diff to calculated gradient failed for i=" << i 
        << ": " << parameters << std::endl 
        << "  finite diffs: " << finite_diffs << std::endl
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

    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
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

        Scalar0 p0_ = get_param<Scalar0>(parameters[n], 0);
        Scalar1 p1_ = get_param<Scalar1>(parameters[n], 1);
        Scalar2 p2_ = get_param<Scalar2>(parameters[n], 2);
        Scalar3 p3_ = get_param<Scalar3>(parameters[n], 3);
        Scalar4 p4_ = get_param<Scalar4>(parameters[n], 4);
        Scalar5 p5_ = get_param<Scalar5>(parameters[n], 5);
        vector<var> x;
        add_vars(x,p0_,p1_,p2_,p3_,p4_,p5_);

        T_return_type logprob = TestClass.template log_prob
          <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
          (p0_,p1_,p2_,p3_,p4_,p5_);
        calculate_gradients_1storder(gradients, logprob, x);

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

  void  test_gradients() {
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
      
          
      T_return_type logprob = TestClass.template log_prob
        <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);

      T_return_type logprob_funct = TestClass.template log_prob_function
        <Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0,p1,p2,p3,p4,p5);

      calculate_gradients_1storder(expected_gradients1, logprob_funct, x1);
      calculate_gradients_1storder(gradients1, logprob, y1);
      calculate_gradients_2ndorder(expected_gradients2, logprob_funct, x2);
      calculate_gradients_2ndorder(gradients2, logprob, y2);
      calculate_gradients_3rdorder(expected_gradients3, logprob_funct, x3);
      calculate_gradients_3rdorder(gradients3, logprob, y3);
      
      test_gradients_equal(expected_gradients1,gradients1);
      test_gradients_equal(expected_gradients2,gradients2);
      test_gradients_equal(expected_gradients3,gradients3);

    }
  }

  void test_multiple_gradient_values(const bool is_vec,
                                     const vector<double>& single_gradients, 
                                     size_t& pos_single,
                                     const vector<double>& multiple_gradients, 
                                     size_t& pos_multiple,
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
    if (all_constant<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << "No test for all double arguments";
      return;
    }
    if (!any_vector<T0,T1,T2,T3,T4,T5>::value) {
      SUCCEED() << "No test for non-vector arguments";
      return;
    }
    const size_t N_REPEAT = 3;
    vector<double> log_prob;
    vector<vector<double> > parameters;
    TestClass.valid_values(parameters, log_prob);
    
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

      T_return_type logprob = TestClass.template log_prob
        <false,Scalar0,Scalar1,Scalar2,Scalar3,Scalar4,Scalar5>
        (p0_,p1_,p2_,p3_,p4_,p5_);

      double single_lp = calculate_gradients_1storder(single_gradients1,logprob,s1);
      calculate_gradients_2ndorder(single_gradients2,logprob,s2);
      calculate_gradients_3rdorder(single_gradients3,logprob,s3);
      
      T0 p0 = get_repeated_params<T0>(parameters[n], 0, N_REPEAT);
      T1 p1 = get_repeated_params<T1>(parameters[n], 1, N_REPEAT);
      T2 p2 = get_repeated_params<T2>(parameters[n], 2, N_REPEAT);
      T3 p3 = get_repeated_params<T3>(parameters[n], 3, N_REPEAT);
      T4 p4 = get_repeated_params<T4>(parameters[n], 4, N_REPEAT);
      T5 p5 = get_repeated_params<T5>(parameters[n], 5, N_REPEAT);

      T_return_type multiple_lp = TestClass.template log_prob
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

      calculate_gradients_1storder(multiple_gradients1,multiple_lp,x1);
      calculate_gradients_1storder(multiple_gradients2,multiple_lp,x1);
      calculate_gradients_1storder(multiple_gradients3,multiple_lp,x1);
      

      EXPECT_TRUE(N_REPEAT * single_lp - multiple_lp < 1e-8)
        << "log prob with repeated vector input should match "
        << "a multiple of log prob of single input";


      size_t pos_single = 0;
      size_t pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value &&
          !boost::is_same<Scalar0, fvar<double> >::value &&
          !boost::is_same<Scalar0, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T0>::value,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value &&
          !boost::is_same<Scalar1, fvar<double> >::value &&
          !boost::is_same<Scalar1, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T1>::value,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value &&
          !boost::is_same<Scalar2, fvar<double> >::value &&
          !boost::is_same<Scalar2, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T2>::value,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value &&
          !boost::is_same<Scalar3, fvar<double> >::value &&
          !boost::is_same<Scalar3, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T3>::value,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value &&
          !boost::is_same<Scalar4, fvar<double> >::value &&
          !boost::is_same<Scalar4, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T4>::value,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value &&
          !boost::is_same<Scalar5, fvar<double> >::value &&
          !boost::is_same<Scalar5, fvar<fvar<double> > >::value)
        test_multiple_gradient_values(is_vector<T5>::value,
                                      single_gradients1, pos_single,
                                      multiple_gradients1, pos_multiple,
                                      N_REPEAT);

      pos_single = 0;
      pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value &&
          (boost::is_same<Scalar0, fvar<var> >::value || 
           boost::is_same<Scalar0, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T0>::value,
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value &&
          (boost::is_same<Scalar1, fvar<var> >::value || 
           boost::is_same<Scalar1, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T1>::value,
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value &&
          (boost::is_same<Scalar2, fvar<var> >::value || 
           boost::is_same<Scalar2, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T2>::value,
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value&&
          (boost::is_same<Scalar3, fvar<var> >::value || 
           boost::is_same<Scalar3, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T3>::value,
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value &&
          (boost::is_same<Scalar4, fvar<var> >::value || 
           boost::is_same<Scalar4, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T4>::value,
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value &&
          (boost::is_same<Scalar5, fvar<var> >::value || 
           boost::is_same<Scalar5, fvar<fvar<var> > >::value) )
        test_multiple_gradient_values(is_vector<T5>::value,
                                      single_gradients2, pos_single,
                                      multiple_gradients2, pos_multiple,
                                      N_REPEAT);

      pos_single = 0;
      pos_multiple = 0;
      if (!is_constant_struct<T0>::value && !is_empty<T0>::value &&
          boost::is_same<Scalar0, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T0>::value,
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T1>::value && !is_empty<T1>::value &&
          boost::is_same<Scalar1, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T1>::value,
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T2>::value && !is_empty<T2>::value &&
          boost::is_same<Scalar2, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T2>::value,
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T3>::value && !is_empty<T3>::value &&
          boost::is_same<Scalar3, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T3>::value,
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T4>::value && !is_empty<T4>::value &&
          boost::is_same<Scalar4, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T4>::value,
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
      if (!is_constant_struct<T5>::value && !is_empty<T5>::value &&
          boost::is_same<Scalar5, fvar<fvar<var> > >::value)
        test_multiple_gradient_values(is_vector<T5>::value,
                                      single_gradients3, pos_single,
                                      multiple_gradients3, pos_multiple,
                                      N_REPEAT);
    }
  }

  void test_length_0_vector() {
    if (!any_vector<T0,T1,T2,T3,T4,T5>::value) {
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

    T_return_type lp = TestClass.template log_prob
      <T0,T1,T2,T3,T4,T5>
      (p0,p1,p2,p3,p4,p5);

    EXPECT_TRUE(0.0 == lp)
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
  this->test_gradients();
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


#endif
