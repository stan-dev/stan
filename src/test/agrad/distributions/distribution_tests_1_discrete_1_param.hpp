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
  AgradTest<TypeParam, int, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iv) {
  AgradTest<TypeParam, int, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_id) {
  AgradTest<TypeParam, int, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iv) {
  AgradTest<TypeParam, int, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_id) {
  AgradTest<TypeParam, int, double>::test_propto();  
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iv) {
  AgradTest<TypeParam, int, var>::test_propto();  
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_id) {
  AgradTest<TypeParam, int, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iv) {
  AgradTest<TypeParam, int, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_id) {
  AgradTest<TypeParam, int, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iv) {
  AgradTest<TypeParam, int, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_id) {
  AgradTest<TypeParam, int, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iD) {
  AgradTest<TypeParam, int, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iv) {
  AgradTest<TypeParam, int, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iV) {
  AgradTest<TypeParam, int, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Id) {
  AgradTest<TypeParam, vector<int>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_ID) {
  AgradTest<TypeParam, vector<int>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iv) {
  AgradTest<TypeParam, vector<int>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IV) {
  AgradTest<TypeParam, vector<int>, vector<var> >::test_vectorized();
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
