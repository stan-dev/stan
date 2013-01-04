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

TYPED_TEST_P(AgradDistributionTestFixture, test_id) {
  typedef AgradTest<TypeParam, int, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iv) {
  typedef AgradTest<TypeParam, int, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iD) {
  typedef AgradTest<TypeParam, int, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_iV) {
  typedef AgradTest<TypeParam, int, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Id) {
  typedef AgradTest<TypeParam, vector<int>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Iv) {
  typedef AgradTest<TypeParam, vector<int>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_ID) {
  typedef AgradTest<TypeParam, vector<int>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_IV) {
  typedef AgradTest<TypeParam, vector<int>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   test_id,
			   test_iv,
			   test_iD,
			   test_iV,			   
			   test_Id,
			   test_Iv,
			   test_ID,
			   test_IV);
#endif
