#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_1_PARAM_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_1_PARAM_HPP___

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
  var call(T0& p0, T1& p1, T2& p2, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<true>(p0, p1, p2);
  }
  var call_nopropto(T0& p0, T1& p1, T2& p2, T3&, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<false>(p0, p1, p2);
  }
};

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

  int param0, param1;
  var param2;
  var logprob;
  param0 = parameters[0];
  param1 = parameters[1];
  param2 = parameters[2];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iid) {
  AgradTest<TypeParam, int, int, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iiv) {
  AgradTest<TypeParam, int, int, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iid) {
  AgradTest<TypeParam, int, int, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iiv) {
  AgradTest<TypeParam, int, int, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iid) {
  AgradTest<TypeParam, int, int, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iiv) {
  AgradTest<TypeParam, int, int, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iid) {
  AgradTest<TypeParam, int, int, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iiv) {
  AgradTest<TypeParam, int, int, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iid) {
  AgradTest<TypeParam, int, int, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iiv) {
  AgradTest<TypeParam, int, int, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iid) {
  AgradTest<TypeParam, int, int, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiv) {
  AgradTest<TypeParam, int, int, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiD) {
  AgradTest<TypeParam, int, int, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiV) {
  AgradTest<TypeParam, int, int, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iId) {
  AgradTest<TypeParam, int, vector<int>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iIv) {
  AgradTest<TypeParam, int, vector<int>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iID) {
  AgradTest<TypeParam, int, vector<int>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iIV) {
  AgradTest<TypeParam, int, vector<int>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iid) {
  AgradTest<TypeParam, vector<int>, int, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iiv) {
  AgradTest<TypeParam, vector<int>, int, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IiD) {
  AgradTest<TypeParam, vector<int>, int, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IiV) {
  AgradTest<TypeParam, vector<int>, int, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IId) {
  AgradTest<TypeParam, vector<int>, vector<int>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IIv) {
  AgradTest<TypeParam, vector<int>, vector<int>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IID) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IIV) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<var> >::test_vectorized();
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_iid,
			   check_valid_iiv,
			   check_invalid_iid,
			   check_invalid_iiv,
			   logprob_propto_iid,
			   logprob_propto_iiv,
			   gradient_finite_diff_iid,
			   gradient_finite_diff_iiv,
			   gradient_function_iid,
			   gradient_function_iiv,
			   vectorized_iid,
			   vectorized_iiv,
			   vectorized_iiD,
			   vectorized_iiV,
			   vectorized_iId,
			   vectorized_iIv,
			   vectorized_iID,
			   vectorized_iIV,
			   vectorized_Iid,
			   vectorized_Iiv,
			   vectorized_IiD,
			   vectorized_IiV,
			   vectorized_IId,			   
			   vectorized_IIv,
			   vectorized_IID,
			   vectorized_IIV);
#endif
