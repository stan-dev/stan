#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_1_PARAM_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_1_PARAM_HPP___

// i: int
// v: var
// d: double
// I: vector<int>
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
  var call_nopropto(T0& p0, T1& p1, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<false>(p0, p1);
  }
};

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

  int param0;
  var param1;
  var logprob;
  param0 = parameters[0];
  param1 = parameters[1];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_id) {
  test_valid<TypeParam, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iv) {
  test_valid<TypeParam, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_id) {
  test_invalid<TypeParam, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iv) {
  test_invalid<TypeParam, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_id) {
  test_propto<TypeParam, int, double>();  
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iv) {
  test_propto<TypeParam, int, var>();  
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_id) {
  test_finite_diff<TypeParam, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iv) {
  test_finite_diff<TypeParam, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_id) {
  SUCCEED() << "No op for (i,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iv) {
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_id) {
  test_vectorized<TypeParam, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iD) {
  test_vectorized<TypeParam, int, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iv) {
  test_vectorized<TypeParam, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iV) {
  test_vectorized<TypeParam, int, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Id) {
  test_vectorized<TypeParam, vector<int>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_ID) {
  test_vectorized<TypeParam, vector<int>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iv) {
  test_vectorized<TypeParam, vector<int>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IV) {
  test_vectorized<TypeParam, vector<int>, vector<var> >();
}
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_id,
			   check_valid_iv,
			   check_invalid_id,
			   check_invalid_iv,
			   logprob_propto_id,
			   logprob_propto_iv,
			   gradient_finite_diff_id,
			   gradient_finite_diff_iv,
			   gradient_function_id,
			   gradient_function_iv,
			   vectorized_id,
			   vectorized_iD,
			   vectorized_iv,
			   vectorized_iV,
			   vectorized_Id,
			   vectorized_ID,
			   vectorized_Iv,
			   vectorized_IV);
#endif
