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

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iidd) {
  AgradTest<TypeParam, int, int, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iidv) {
  AgradTest<TypeParam, int, int, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iivd) {
  AgradTest<TypeParam, int, int, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iivv) {
  AgradTest<TypeParam, int, int, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iidd) {
  AgradTest<TypeParam, int, int, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iidv) {  
  AgradTest<TypeParam, int, int, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iivd) {
  AgradTest<TypeParam, int, int, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iivv) {
  AgradTest<TypeParam, int, int, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iidd) {
  AgradTest<TypeParam, int, int, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iidv) {
  AgradTest<TypeParam, int, int, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iivd) { 
  AgradTest<TypeParam, int, int, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iivv) { 
  AgradTest<TypeParam, int, int, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iidd) {
  AgradTest<TypeParam, int, int, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iidv) {
  AgradTest<TypeParam, int, int, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iivd) {  
  AgradTest<TypeParam, int, int, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iivv) {
  AgradTest<TypeParam, int, int, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iidd) {
  AgradTest<TypeParam, int, int, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iidv) {  
  AgradTest<TypeParam, int, int, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iivd) {  
  AgradTest<TypeParam, int, int, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iivv) {  
  AgradTest<TypeParam, int, int, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iidd) {
  AgradTest<TypeParam, int, int, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iidD) {
  AgradTest<TypeParam, int, int, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iidv) {
  AgradTest<TypeParam, int, int, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iidV) {
  AgradTest<TypeParam, int, int, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiDd) {
  AgradTest<TypeParam, int, int, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiDD) {
  AgradTest<TypeParam, int, int, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiDv) {
  AgradTest<TypeParam, int, int, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiDV) {
  AgradTest<TypeParam, int, int, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iivd) {
  AgradTest<TypeParam, int, int, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iivD) {
  AgradTest<TypeParam, int, int, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iivv) {
  AgradTest<TypeParam, int, int, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iivV) {
  AgradTest<TypeParam, int, int, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiVd) {
  AgradTest<TypeParam, int, int, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiVD) {
  AgradTest<TypeParam, int, int, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiVv) {
  AgradTest<TypeParam, int, int, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iiVV) {
  AgradTest<TypeParam, int, int, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIdd) {
  AgradTest<TypeParam, int, vector<int>, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIdD) {
  AgradTest<TypeParam, int, vector<int>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIdv) {
  AgradTest<TypeParam, int, vector<int>, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIdV) {
  AgradTest<TypeParam, int, vector<int>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIDd) {
  AgradTest<TypeParam, int, vector<int>, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIDD) {
  AgradTest<TypeParam, int, vector<int>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIDv) {
  AgradTest<TypeParam, int, vector<int>, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIDV) {
  AgradTest<TypeParam, int, vector<int>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIvd) {
  AgradTest<TypeParam, int, vector<int>, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIvD) {
  AgradTest<TypeParam, int, vector<int>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIvv) {
  AgradTest<TypeParam, int, vector<int>, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIvV) {
  AgradTest<TypeParam, int, vector<int>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIVd) {
  AgradTest<TypeParam, int, vector<int>, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIVD) {
  AgradTest<TypeParam, int, vector<int>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIVv) {
  AgradTest<TypeParam, int, vector<int>, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_iIVV) {
  AgradTest<TypeParam, int, vector<int>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_Iidd) {
  AgradTest<TypeParam, vector<int>, int, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IidD) {
  AgradTest<TypeParam, vector<int>, int, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_Iidv) {
  AgradTest<TypeParam, vector<int>, int, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IidV) {
  AgradTest<TypeParam, vector<int>, int, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiDd) {
  AgradTest<TypeParam, vector<int>, int, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiDD) {
  AgradTest<TypeParam, vector<int>, int, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiDv) {
  AgradTest<TypeParam, vector<int>, int, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiDV) {
  AgradTest<TypeParam, vector<int>, int, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_Iivd) {
  AgradTest<TypeParam, vector<int>, int, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IivD) {
  AgradTest<TypeParam, vector<int>, int, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_Iivv) {
  AgradTest<TypeParam, vector<int>, int, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IivV) {
  AgradTest<TypeParam, vector<int>, int, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiVd) {
  AgradTest<TypeParam, vector<int>, int, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiVD) {
  AgradTest<TypeParam, vector<int>, int, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiVv) {
  AgradTest<TypeParam, vector<int>, int, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IiVV) {
  AgradTest<TypeParam, vector<int>, int, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIdd) {
  AgradTest<TypeParam, vector<int>, vector<int>, double, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIdD) {
  AgradTest<TypeParam, vector<int>, vector<int>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIdv) {
  AgradTest<TypeParam, vector<int>, vector<int>, double, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIdV) {
  AgradTest<TypeParam, vector<int>, vector<int>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIDd) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIDD) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIDv) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIDV) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIvd) {
  AgradTest<TypeParam, vector<int>, vector<int>, var, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIvD) {
  AgradTest<TypeParam, vector<int>, vector<int>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIvv) {
  AgradTest<TypeParam, vector<int>, vector<int>, var, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIvV) {
  AgradTest<TypeParam, vector<int>, vector<int>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIVd) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, double>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIVD) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIVv) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, var>::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_IIVV) {
  AgradTest<TypeParam, vector<int>, vector<int>, vector<var>, vector<var> >::test_vectorized();
}


// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_iidd,
			   check_valid_iidv,
			   check_valid_iivd,
			   check_valid_iivv,
			   check_invalid_iidd,
			   check_invalid_iidv,
			   check_invalid_iivd,
			   check_invalid_iivv,
			   logprob_propto_iidd,
			   logprob_propto_iidv,
			   logprob_propto_iivd,
			   logprob_propto_iivv,
			   gradient_finite_diff_iidd,
			   gradient_finite_diff_iidv,
			   gradient_finite_diff_iivd,
			   gradient_finite_diff_iivv,
			   gradient_function_iidd,
			   gradient_function_iidv,
			   gradient_function_iivd,
			   gradient_function_iivv);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture2,
			   vectorized_iidd,
			   vectorized_iidD,
			   vectorized_iidv,
			   vectorized_iidV,
			   vectorized_iiDd,
			   vectorized_iiDD,
			   vectorized_iiDv,
			   vectorized_iiDV,
			   vectorized_iivd,
			   vectorized_iivD,
			   vectorized_iivv,
			   vectorized_iivV,
			   vectorized_iiVd,
			   vectorized_iiVD,
			   vectorized_iiVv,
			   vectorized_iiVV,
			   vectorized_iIdd,
			   vectorized_iIdD,
			   vectorized_iIdv,
			   vectorized_iIdV,
			   vectorized_iIDd,
			   vectorized_iIDD,
			   vectorized_iIDv,
			   vectorized_iIDV,
			   vectorized_iIvd,
			   vectorized_iIvD,
			   vectorized_iIvv,
			   vectorized_iIvV,
			   vectorized_iIVd,
			   vectorized_iIVD,
			   vectorized_iIVv,
			   vectorized_iIVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture3,
			   vectorized_Iidd,
			   vectorized_IidD,
			   vectorized_Iidv,
			   vectorized_IidV,
			   vectorized_IiDd,
			   vectorized_IiDD,
			   vectorized_IiDv,
			   vectorized_IiDV,
			   vectorized_Iivd,
			   vectorized_IivD,
			   vectorized_Iivv,
			   vectorized_IivV,
			   vectorized_IiVd,
			   vectorized_IiVD,
			   vectorized_IiVv,
			   vectorized_IiVV,
			   vectorized_IIdd,
			   vectorized_IIdD,
			   vectorized_IIdv,
			   vectorized_IIdV,
			   vectorized_IIDd,
			   vectorized_IIDD,
			   vectorized_IIDv,
			   vectorized_IIDV,
			   vectorized_IIvd,
			   vectorized_IIvD,
			   vectorized_IIvv,
			   vectorized_IIvV,
			   vectorized_IIVd,
			   vectorized_IIVD,
			   vectorized_IIVv,
			   vectorized_IIVV);
#endif
