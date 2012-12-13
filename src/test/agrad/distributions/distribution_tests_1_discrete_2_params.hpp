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
  AgradTest<TypeParam, int, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_idv) {
  AgradTest<TypeParam, int, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ivd) {
  AgradTest<TypeParam, int, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ivv) {
  AgradTest<TypeParam, int, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_idd) {
  AgradTest<TypeParam, int, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_idv) {  
  AgradTest<TypeParam, int, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ivd) {
  AgradTest<TypeParam, int, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ivv) {
  AgradTest<TypeParam, int, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_idd) {
  AgradTest<TypeParam, int, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_idv) {
  AgradTest<TypeParam, int, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ivd) { 
  AgradTest<TypeParam, int, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ivv) { 
  AgradTest<TypeParam, int, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_idd) {
  AgradTest<TypeParam, int, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_idv) {
  AgradTest<TypeParam, int, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ivd) {  
  AgradTest<TypeParam, int, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ivv) {
  AgradTest<TypeParam, int, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_idd) {
  AgradTest<TypeParam, int, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_idv) {  
  AgradTest<TypeParam, int, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ivd) {  
  AgradTest<TypeParam, int, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ivv) {  
  AgradTest<TypeParam, int, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idd) {
  AgradTest<TypeParam, int, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idD) {
  AgradTest<TypeParam, int, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idv) {
  AgradTest<TypeParam, int, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_idV) {
  AgradTest<TypeParam, int, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDd) {
  AgradTest<TypeParam, int, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDD) {
  AgradTest<TypeParam, int, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDv) {
  AgradTest<TypeParam, int, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iDV) {
  AgradTest<TypeParam, int, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivd) {
  AgradTest<TypeParam, int, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivD) {
  AgradTest<TypeParam, int, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivv) {
  AgradTest<TypeParam, int, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ivV) {
  AgradTest<TypeParam, int, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVd) {
  AgradTest<TypeParam, int, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVD) {
  AgradTest<TypeParam, int, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVv) {
  AgradTest<TypeParam, int, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iVV) {
  AgradTest<TypeParam, int, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Idd) {
  AgradTest<TypeParam, vector<int>, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IdD) {
  AgradTest<TypeParam, vector<int>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Idv) {
  AgradTest<TypeParam, vector<int>, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IdV) {
  AgradTest<TypeParam, vector<int>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDd) {
  AgradTest<TypeParam, vector<int>, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDD) {
  AgradTest<TypeParam, vector<int>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDv) {
  AgradTest<TypeParam, vector<int>, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IDV) {
  AgradTest<TypeParam, vector<int>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ivd) {
  AgradTest<TypeParam, vector<int>, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IvD) {
  AgradTest<TypeParam, vector<int>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ivv) {
  AgradTest<TypeParam, vector<int>, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IvV) {
  AgradTest<TypeParam, vector<int>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVd) {
  AgradTest<TypeParam, vector<int>, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVD) {
  AgradTest<TypeParam, vector<int>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVv) {
  AgradTest<TypeParam, vector<int>, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_IVV) {
  AgradTest<TypeParam, vector<int>, vector<var>, vector<var> >::test_vectorized();
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
