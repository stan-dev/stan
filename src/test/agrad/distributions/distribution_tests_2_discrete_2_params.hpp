#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_2_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_2_PARAMS_HPP___

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
  var call(T0& p0, T1& p1, T2& p2, T3& p3, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<true>(p0, p1, p2, p3);
  }
  var call_nopropto(T0& p0, T1& p1, T2& p2, T3& p3, T4&, T5&, T6&, T7&, T8&, T9&) {
    return _LOG_PROB_<false>(p0, p1, p2, p3);
  }
};

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();
  ASSERT_EQ(4U, parameters.size());

  int param0;
  int param1;
  var param2, param3;
  var logprob;
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3 = parameters[3];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, param3, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture2, test_iidd) {
  typedef AgradTest<TypeParam, int, int, double, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iidD) {
  typedef AgradTest<TypeParam, int, int, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iidv) {
  typedef AgradTest<TypeParam, int, int, double, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iidV) {
  typedef AgradTest<TypeParam, int, int, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiDd) {
  typedef AgradTest<TypeParam, int, int, vector<double>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiDD) {
  typedef AgradTest<TypeParam, int, int, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiDv) {
  typedef AgradTest<TypeParam, int, int, vector<double>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiDV) {
  typedef AgradTest<TypeParam, int, int, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iivd) {
  typedef AgradTest<TypeParam, int, int, var, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iivD) {
  typedef AgradTest<TypeParam, int, int, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iivv) {
  typedef AgradTest<TypeParam, int, int, var, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iivV) {
  typedef AgradTest<TypeParam, int, int, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiVd) {
  typedef AgradTest<TypeParam, int, int, vector<var>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiVD) {
  typedef AgradTest<TypeParam, int, int, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiVv) {
  typedef AgradTest<TypeParam, int, int, vector<var>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iiVV) {
  typedef AgradTest<TypeParam, int, int, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIdd) {
  typedef AgradTest<TypeParam, int, vector<int>, double, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIdD) {
  typedef AgradTest<TypeParam, int, vector<int>, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIdv) {
  typedef AgradTest<TypeParam, int, vector<int>, double, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIdV) {
  typedef AgradTest<TypeParam, int, vector<int>, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIDd) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<double>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIDD) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIDv) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<double>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIDV) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIvd) {
  typedef AgradTest<TypeParam, int, vector<int>, var, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIvD) {
  typedef AgradTest<TypeParam, int, vector<int>, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIvv) {
  typedef AgradTest<TypeParam, int, vector<int>, var, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIvV) {
  typedef AgradTest<TypeParam, int, vector<int>, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIVd) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<var>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIVD) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIVv) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<var>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_iIVV) {
  typedef AgradTest<TypeParam, int, vector<int>, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Iidd) {
  typedef AgradTest<TypeParam, vector<int>, int, double, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IidD) {
  typedef AgradTest<TypeParam, vector<int>, int, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Iidv) {
  typedef AgradTest<TypeParam, vector<int>, int, double, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IidV) {
  typedef AgradTest<TypeParam, vector<int>, int, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiDd) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<double>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiDD) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiDv) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<double>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiDV) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Iivd) {
  typedef AgradTest<TypeParam, vector<int>, int, var, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IivD) {
  typedef AgradTest<TypeParam, vector<int>, int, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Iivv) {
  typedef AgradTest<TypeParam, vector<int>, int, var, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IivV) {
  typedef AgradTest<TypeParam, vector<int>, int, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiVd) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<var>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiVD) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiVv) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<var>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IiVV) {
  typedef AgradTest<TypeParam, vector<int>, int, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIdd) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, double, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIdD) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIdv) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, double, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIdV) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIDd) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIDD) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIDv) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIDV) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIvd) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, var, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIvD) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIvv) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, var, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIvV) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIVd) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, double> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIVD) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIVv) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, var> Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_IIVV) {
  typedef AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, vector<var> > Test;
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
			   test_iidd,
			   test_iidD,
			   test_iidv,
			   test_iidV,
			   test_iiDd,
			   test_iiDD,
			   test_iiDv,
			   test_iiDV,
			   test_iivd,
			   test_iivD,
			   test_iivv,
			   test_iivV,
			   test_iiVd,
			   test_iiVD,
			   test_iiVv,
			   test_iiVV,
			   test_iIdd,
			   test_iIdD,
			   test_iIdv,
			   test_iIdV,
			   test_iIDd,
			   test_iIDD,
			   test_iIDv,
			   test_iIDV,
			   test_iIvd,
			   test_iIvD,
			   test_iIvv,
			   test_iIvV,
			   test_iIVd,
			   test_iIVD,
			   test_iIVv,
			   test_iIVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture3,
			   test_Iidd,
			   test_IidD,
			   test_Iidv,
			   test_IidV,
			   test_IiDd,
			   test_IiDD,
			   test_IiDv,
			   test_IiDV,
			   test_Iivd,
			   test_IivD,
			   test_Iivv,
			   test_IivV,
			   test_IiVd,
			   test_IiVD,
			   test_IiVv,
			   test_IiVV,
			   test_IIdd,
			   test_IIdD,
			   test_IIdv,
			   test_IIdV,
			   test_IIDd,
			   test_IIDD,
			   test_IIDv,
			   test_IIDV,
			   test_IIvd,
			   test_IIvD,
			   test_IIvv,
			   test_IIvV,
			   test_IIVd,
			   test_IIVD,
			   test_IIVv,
			   test_IIVV);
#endif
