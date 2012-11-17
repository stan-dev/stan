#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_2_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_2_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>

// i: int
// I: vector

template<class T0, class T1, class T2, class T3,
	 class T4, class T5, class T6,
	 class T7, class T8, class T9>
class CALL_LOG_PROB {
public:
  var call(T0& p0, T1& p1, T2& p2, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<true>(p0, p1, p2);
  }
  var call_nopropto(T0& p0, T1& p1, T2& p2, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<false>(p0, p1, p2);
  }
};

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();
  ASSERT_EQ(3U, parameters.size());

  
  int param0;
  var param1, param2;
  var logprob;
  param0 = int(parameters[0]);
  param1 = parameters[1];
  param2 = parameters[2];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_idd) {
  test_valid<TypeParam, int, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_idv) {
  test_valid<TypeParam, int, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ivd) {
  test_valid<TypeParam, int, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ivv) {
  test_valid<TypeParam, int, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_idd) {
  test_invalid<TypeParam, int, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_idv) {  
  test_invalid<TypeParam, int, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ivd) {
  test_invalid<TypeParam, int, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ivv) {
  test_invalid<TypeParam, int, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_idd) {
  test_propto<TypeParam, int, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_idv) {
  test_propto<TypeParam, int, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ivd) { 
  test_propto<TypeParam, int, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ivv) { 
  test_propto<TypeParam, int, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_idd) {
  SUCCEED() << "No op for (i,d,d)" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_idv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g2 = (_LOG_PROB_<false>(int(p[0]), p[1], p[2]+e) - _LOG_PROB_<false>(int(p[0]), p[1], p[2]-e)) / e_times_2;
      
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(int(p[0]), p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g2,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ivd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(int(p[0]), p[1]+e, p[2]) - _LOG_PROB_<false>(int(p[0]), p[1]-e, p[2])) / e_times_2;
      
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p[2]);
    vector<var> v_params(1);
    v_params[0] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ivv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(int(p[0]), p[1]+e, p[2]) - _LOG_PROB_<false>(int(p[0]), p[1]-e, p[2])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(int(p[0]), p[1], p[2]+e) - _LOG_PROB_<false>(int(p[0]), p[1], p[2]-e)) / e_times_2;
      
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p2);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_idd) {
  SUCCEED() << "No op for (i,d,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_idv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p[1], p2);
    var expected_lp = TypeParam().log_prob(int(p[0]), p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ivd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p[2]);
    var expected_lp = TypeParam().log_prob(int(p[0]), p1, p[2]);
    vector<var> v_params(1);
    v_params[0] = p1;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ivv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p2);
    var expected_lp = TypeParam().log_prob(int(p[0]), p1, p2);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idd) {
  test_vectorized<TypeParam, int, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idD) {
  test_vectorized<TypeParam, int, double, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idv) {
  test_vectorized<TypeParam, int, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idV) {
  test_vectorized<TypeParam, int, double, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDd) {
  test_vectorized<TypeParam, int, vector<double>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDD) {
  test_vectorized<TypeParam, int, vector<double>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDv) {
  test_vectorized<TypeParam, int, vector<double>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDV) {
  test_vectorized<TypeParam, int, vector<double>, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivd) {
  test_vectorized<TypeParam, int, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivD) {
  test_vectorized<TypeParam, int, var, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivv) {
  test_vectorized<TypeParam, int, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivV) {
  test_vectorized<TypeParam, int, var, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVd) {
  test_vectorized<TypeParam, int, vector<var>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVD) {
  test_vectorized<TypeParam, int, vector<var>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVv) {
  test_vectorized<TypeParam, int, vector<var>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVV) {
  test_vectorized<TypeParam, int, vector<var>, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Idd) {
  test_vectorized<TypeParam, vector<int>, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IdD) {
  test_vectorized<TypeParam, vector<int>, double, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Idv) {
  test_vectorized<TypeParam, vector<int>, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IdV) {
  test_vectorized<TypeParam, vector<int>, double, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDd) {
  test_vectorized<TypeParam, vector<int>, vector<double>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDD) {
  test_vectorized<TypeParam, vector<int>, vector<double>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDv) {
  test_vectorized<TypeParam, vector<int>, vector<double>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDV) {
  test_vectorized<TypeParam, vector<int>, vector<double>, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ivd) {
  test_vectorized<TypeParam, vector<int>, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IvD) {
  test_vectorized<TypeParam, vector<int>, var, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ivv) {
  test_vectorized<TypeParam, vector<int>, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IvV) {
  test_vectorized<TypeParam, vector<int>, var, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVd) {
  test_vectorized<TypeParam, vector<int>, vector<var>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVD) {
  test_vectorized<TypeParam, vector<int>, vector<var>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVv) {
  test_vectorized<TypeParam, vector<int>, vector<var>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVV) {
  test_vectorized<TypeParam, vector<int>, vector<var>, vector<var> >();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_idd,
			   check_valid_idv,
			   check_valid_ivd,
			   check_valid_ivv,
			   check_invalid_idd,
			   check_invalid_idv,
			   check_invalid_ivd,
			   check_invalid_ivv,
			   logprob_propto_idd,
			   logprob_propto_idv,
			   logprob_propto_ivd,
			   logprob_propto_ivv,
			   gradient_finite_diff_idd,
			   gradient_finite_diff_idv,
			   gradient_finite_diff_ivd,
			   gradient_finite_diff_ivv,
			   gradient_function_idd,
			   gradient_function_idv,
			   gradient_function_ivd,
			   gradient_function_ivv);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture2,
			   vectorized_idd,
			   vectorized_idD,
			   vectorized_idv,
			   vectorized_idV,
			   vectorized_iDd,
			   vectorized_iDD,
			   vectorized_iDv,
			   vectorized_iDV,
			   vectorized_ivd,
			   vectorized_ivD,
			   vectorized_ivv,
			   vectorized_ivV,
			   vectorized_iVd,
			   vectorized_iVD,
			   vectorized_iVv,
			   vectorized_iVV,
			   vectorized_Idd,
			   vectorized_IdD,
			   vectorized_Idv,
			   vectorized_IdV,
			   vectorized_IDd,
			   vectorized_IDD,
			   vectorized_IDv,
			   vectorized_IDV,
			   vectorized_Ivd,
			   vectorized_IvD,
			   vectorized_Ivv,
			   vectorized_IvV,
			   vectorized_IVd,
			   vectorized_IVD,
			   vectorized_IVv,
			   vectorized_IVV);
#endif
