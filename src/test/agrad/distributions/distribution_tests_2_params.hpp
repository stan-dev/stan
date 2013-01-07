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

TYPED_TEST_P(AgradDistributionTestFixture, test_dd) {
  typedef AgradTest<TypeParam, double, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_dD) {
  typedef AgradTest<TypeParam, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_dv) {
  typedef AgradTest<TypeParam, double, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_dV) {
  typedef AgradTest<TypeParam, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Dd) {
  typedef AgradTest<TypeParam, vector<double>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_DD) {
  typedef AgradTest<TypeParam, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Dv) {
  typedef AgradTest<TypeParam, vector<double>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_DV) {
  typedef AgradTest<TypeParam, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_vd) {
  typedef AgradTest<TypeParam, var, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_vD) {
  typedef AgradTest<TypeParam, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_vv) {
  typedef AgradTest<TypeParam, var, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_vV) {
  typedef AgradTest<TypeParam, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Vd) {
  typedef AgradTest<TypeParam, vector<var>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_VD) {
  typedef AgradTest<TypeParam, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_Vv) {
  typedef AgradTest<TypeParam, vector<var>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture, test_VV) {
  typedef AgradTest<TypeParam, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   test_dd,
			   test_dD,
			   test_dv,
			   test_dV,
			   test_Dd,
			   test_DD,
			   test_Dv,
			   test_DV,		    
			   test_vd,
			   test_vD,
			   test_vv,
			   test_vV,
			   test_Vd,
			   test_VD,
			   test_Vv,
			   test_VV);
#endif
