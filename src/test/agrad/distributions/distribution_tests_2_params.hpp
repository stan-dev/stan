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
  var call_nopropto(T0& p0, T1& p1, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<false>(p0, p1);
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
  test_propto<TypeParam, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dv) {
  test_propto<TypeParam, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vd) { 
  test_propto<TypeParam, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vv) { 
  test_propto<TypeParam, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dd) {
  test_finite_diff<TypeParam, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dv) {
  test_finite_diff<TypeParam, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vd) {  
  test_finite_diff<TypeParam, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vv) {
  test_finite_diff<TypeParam, var, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dd) {
  test_gradient_function<TypeParam, double, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dv) {  
  test_gradient_function<TypeParam, double, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vd) {  
  test_gradient_function<TypeParam, var, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vv) {  
  test_gradient_function<TypeParam, var, var>();
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
