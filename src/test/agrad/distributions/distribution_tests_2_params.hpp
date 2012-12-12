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
  AgradTest<TypeParam, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dv) {
  AgradTest<TypeParam, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vd) {
  AgradTest<TypeParam, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vv) {
  AgradTest<TypeParam, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dd) {
  AgradTest<TypeParam, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dv) {  
  AgradTest<TypeParam, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vd) {
  AgradTest<TypeParam, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vv) {
  AgradTest<TypeParam, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dd) {
  AgradTest<TypeParam, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dv) {
  AgradTest<TypeParam, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vd) { 
  AgradTest<TypeParam, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vv) { 
  AgradTest<TypeParam, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dd) {
  AgradTest<TypeParam, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dv) {
  AgradTest<TypeParam, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vd) {  
  AgradTest<TypeParam, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vv) {
  AgradTest<TypeParam, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dd) {
  AgradTest<TypeParam, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dv) {  
  AgradTest<TypeParam, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vd) {  
  AgradTest<TypeParam, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vv) {  
  AgradTest<TypeParam, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dd) {
  AgradTest<TypeParam, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dD) {
  AgradTest<TypeParam, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dv) {
  AgradTest<TypeParam, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dV) {
  AgradTest<TypeParam, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Dd) {
  AgradTest<TypeParam, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DD) {
  AgradTest<TypeParam, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Dv) {
  AgradTest<TypeParam, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DV) {
  AgradTest<TypeParam, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vd) {
  AgradTest<TypeParam, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vD) {
  AgradTest<TypeParam, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vv) {
  AgradTest<TypeParam, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vV) {
  AgradTest<TypeParam, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Vd) {
  AgradTest<TypeParam, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VD) {
  AgradTest<TypeParam, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Vv) {
  AgradTest<TypeParam, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VV) {
  AgradTest<TypeParam, vector<var>, vector<var> >::test_vectorized();
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
