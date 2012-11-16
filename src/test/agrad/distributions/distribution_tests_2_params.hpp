#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>
template<class T0, class T1, class T2, class T3,
	 class T4, class T5, class T6,
	 class T7, class T8, class T9>
class CALL_LOG_PROB {
public:
  var call(T0& p0, T1& p1, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<true>(p0, p1);
  }
};

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

  var param1, param2;
  var logprob;
  param1 = parameters[0];
  param2 = parameters[1];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dd) {
  test_valid<TypeParam, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dv) {
  test_valid<TypeParam, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vd) {
  test_valid<TypeParam, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vv) {
  test_valid<TypeParam, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dd) {
  test_invalid<TypeParam, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dv) {  
  test_invalid<TypeParam, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vd) {
  test_invalid<TypeParam, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vv) {
  test_invalid<TypeParam, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dd) {
  var logprob_true = _LOG_PROB_<true>(this->first_valid_params()[0],
				      this->first_valid_params()[1]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]));
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vd) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1]);
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vv) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]));
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dd) {
  SUCCEED() << "No op for all double" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e) - _LOG_PROB_<false>(p[0], p[1]-e)) / e_times_2;
      
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(p[0], p1);
    vector<var> v_params(1);
    v_params[0] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1]) - _LOG_PROB_<false>(p[0]-e, p[1])) / e_times_2;
      
    var p0(p[0]);
    var lp = _LOG_PROB_<true>(p0, p[1]);
    vector<var> v_params(1);
    v_params[0] = p0;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1]) - _LOG_PROB_<false>(p[0]-e, p[1])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e) - _LOG_PROB_<false>(p[0], p[1]-e)) / e_times_2;
      
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dd) {
  SUCCEED() << "No op for (d,d,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p[0], p1);
    var expected_lp = TypeParam().log_prob(p[0], p1);
    vector<var> v_params(1);
    v_params[0] = p1;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    
    var lp = _LOG_PROB_<true>(p0, p[1]);
    var expected_lp = TypeParam().log_prob(p0, p[1]);
    vector<var> v_params(1);
    v_params[0] = p0;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1);
    var expected_lp = TypeParam().log_prob(p0, p1);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dd) {
  test_vectorized<TypeParam, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dD) {
  test_vectorized<TypeParam, double, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dv) {
  test_vectorized<TypeParam, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dV) {
  test_vectorized<TypeParam, double, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Dd) {
  test_vectorized<TypeParam, vector<double>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DD) {
  test_vectorized<TypeParam, vector<double>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Dv) {
  test_vectorized<TypeParam, vector<double>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DV) {
  test_vectorized<TypeParam, vector<double>, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vd) {
  test_vectorized<TypeParam, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vD) {
  test_vectorized<TypeParam, var, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vv) {
  test_vectorized<TypeParam, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vV) {
  test_vectorized<TypeParam, var, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Vd) {
  test_vectorized<TypeParam, vector<var>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VD) {
  test_vectorized<TypeParam, vector<var>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Vv) {
  test_vectorized<TypeParam, vector<var>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VV) {
  test_vectorized<TypeParam, vector<var>, vector<var> >();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_dd,
			   check_valid_dv,
			   check_valid_vd,
			   check_valid_vv,
			   check_invalid_dd,
			   check_invalid_dv,
			   check_invalid_vd,
			   check_invalid_vv,
			   logprob_propto_dd,
			   logprob_propto_dv,
			   logprob_propto_vd,
			   logprob_propto_vv,
			   gradient_finite_diff_dd,
			   gradient_finite_diff_dv,
			   gradient_finite_diff_vd,
			   gradient_finite_diff_vv,
			   gradient_function_dd,
			   gradient_function_dv,
			   gradient_function_vd,
			   gradient_function_vv,
			   vectorized_dd,
			   vectorized_dD,
			   vectorized_dv,
			   vectorized_dV,
			   vectorized_Dd,
			   vectorized_DD,
			   vectorized_Dv,
			   vectorized_DV,		    
			   vectorized_vd,
			   vectorized_vD,
			   vectorized_vv,
			   vectorized_vV,
			   vectorized_Vd,
			   vectorized_VD,
			   vectorized_Vv,
			   vectorized_VV);
#endif
