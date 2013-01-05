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
TYPED_TEST_P(AgradDistributionTestFixture, test_iid) {
  typedef AgradTest<TypeParam, int, int, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iiv) {
  typedef AgradTest<TypeParam, int, int, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iiD) {
  typedef AgradTest<TypeParam, int, int, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iiV) {
  typedef AgradTest<TypeParam, int, int, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iId) {
  typedef AgradTest<TypeParam, int, vector<int>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iIv) {
  typedef AgradTest<TypeParam, int, vector<int>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iID) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iIV) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Iid) {
  typedef AgradTest<TypeParam, vector<int>, int, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Iiv) {
  typedef AgradTest<TypeParam, vector<int>, int, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IiD) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IiV) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IId) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IIv) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IID) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IIV) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   test_iid,
			   test_iiv,
			   test_iiD,
			   test_iiV,
			   test_iId,
			   test_iIv,
			   test_iID,
			   test_iIV,
			   test_Iid,
			   test_Iiv,
			   test_IiD,
			   test_IiV,
			   test_IId,			   
			   test_IIv,
			   test_IID,
			   test_IIV);
#endif
