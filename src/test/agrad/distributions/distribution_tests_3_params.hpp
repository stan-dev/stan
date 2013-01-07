#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___

// v: var
// d: double
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
  ASSERT_EQ(parameters.size(), 3U);

  var param1, param2, param3;
  var logprob;
  param1 = parameters[0];
  param2 = parameters[1];
  param3 = parameters[2];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, param3, errno_policy()));
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddd) {
  typedef AgradTest<TypeParam, double, double, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddD) {
  typedef AgradTest<TypeParam, double, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddv) {
  typedef AgradTest<TypeParam, double, double, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddV) {
  typedef AgradTest<TypeParam, double, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDd) {
  typedef AgradTest<TypeParam, double, vector<double>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDD) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDv) {
  typedef AgradTest<TypeParam, double, vector<double>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDV) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dvd) {
  typedef AgradTest<TypeParam, double, var, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dvD) {
  typedef AgradTest<TypeParam, double, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dvv) {
  typedef AgradTest<TypeParam, double, var, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dvV) {
  typedef AgradTest<TypeParam, double, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVd) {
  typedef AgradTest<TypeParam, double, vector<var>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVD) {
  typedef AgradTest<TypeParam, double, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVv) {
  typedef AgradTest<TypeParam, double, vector<var>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVV) {
  typedef AgradTest<TypeParam, double, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Ddd) {
  typedef AgradTest<TypeParam, vector<double>, double, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DdD) {
  typedef AgradTest<TypeParam, vector<double>, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Ddv) {
  typedef AgradTest<TypeParam, vector<double>, double, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DdV) {
  typedef AgradTest<TypeParam, vector<double>, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DDd) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DDD) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DDv) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DDV) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Dvd) {
  typedef AgradTest<TypeParam, vector<double>, var, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DvD) {
  typedef AgradTest<TypeParam, vector<double>, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Dvv) {
  typedef AgradTest<TypeParam, vector<double>, var, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DvV) {
  typedef AgradTest<TypeParam, vector<double>, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DVd) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DVD) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DVv) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_DVV) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Vdd) {
  typedef AgradTest<TypeParam, vector<double>, double, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VdD) {
  typedef AgradTest<TypeParam, vector<double>, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Vdv) {
  typedef AgradTest<TypeParam, vector<double>, double, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VdV) {
  typedef AgradTest<TypeParam, vector<double>, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VDd) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VDD) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VDv) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VDV) {
  typedef AgradTest<TypeParam, vector<double>, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Vvd) {
  typedef AgradTest<TypeParam, vector<double>, var, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VvD) {
  typedef AgradTest<TypeParam, vector<double>, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_Vvv) {
  typedef AgradTest<TypeParam, vector<double>, var, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VvV) {
  typedef AgradTest<TypeParam, vector<double>, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VVd) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VVD) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VVv) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_VVV) {
  typedef AgradTest<TypeParam, vector<double>, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture2,
			   test_ddd,
			   test_ddD,
			   test_ddv,
			   test_ddV,
			   test_dDd,
			   test_dDD,
			   test_dDv,
			   test_dDV,
			   test_dvd,
			   test_dvD,
			   test_dvv,
			   test_dvV,
			   test_dVd,
			   test_dVD,
			   test_dVv,
			   test_dVV,
			   test_Ddd,
			   test_DdD,
			   test_Ddv,
			   test_DdV,
			   test_DDd,
			   test_DDD,
			   test_DDv,
			   test_DDV,
			   test_Dvd,
			   test_DvD,
			   test_Dvv,
			   test_DvV,
			   test_DVd,
			   test_DVD,
			   test_DVv,
			   test_DVV,
			   test_Vdd,
			   test_VdD,
			   test_Vdv,
			   test_VdV,
			   test_VDd,
			   test_VDD,
			   test_VDv,
			   test_VDV,
			   test_Vvd,
			   test_VvD,
			   test_Vvv,
			   test_VvV,
			   test_VVd,
			   test_VVD,
			   test_VVv,
			   test_VVV);
#endif
