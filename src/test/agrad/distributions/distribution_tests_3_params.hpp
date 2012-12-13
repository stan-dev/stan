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

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddd) {
  AgradTest<TypeParam, double, double, double >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddv) {
  AgradTest<TypeParam, double, double, var >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvd) {
  AgradTest<TypeParam, double, var, double >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvv) {
  AgradTest<TypeParam, double, var, var >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdd) {
  AgradTest<TypeParam, var, double, double >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdv) {
  AgradTest<TypeParam, var, double, var >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvd) {
  AgradTest<TypeParam, var, var, double >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvv) {
  AgradTest<TypeParam, var, var, var >::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddd) {
  AgradTest<TypeParam, double, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddv) {
  AgradTest<TypeParam, double, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvd) {  
  AgradTest<TypeParam, double, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvv) {  
  AgradTest<TypeParam, double, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdd) {
  AgradTest<TypeParam, var, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdv) {  
  AgradTest<TypeParam, var, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvd) {
  AgradTest<TypeParam, var, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvv) {
  AgradTest<TypeParam, var, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ddd) {
  AgradTest<TypeParam, double, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ddv) {
  AgradTest<TypeParam, double, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dvd) {
  AgradTest<TypeParam, double, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dvv) { 
  AgradTest<TypeParam, double, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vdd) { 
  AgradTest<TypeParam, var, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vdv) {
  AgradTest<TypeParam, var, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vvd) { 
  AgradTest<TypeParam, var, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vvv) { 
  AgradTest<TypeParam, var, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ddd) {
  AgradTest<TypeParam, double, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ddv) {
  AgradTest<TypeParam, double, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dvd) {
  AgradTest<TypeParam, double, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dvv) {
  AgradTest<TypeParam, double, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vdd) {  
  AgradTest<TypeParam, var, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vdv) {
  AgradTest<TypeParam, var, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vvd) {
  AgradTest<TypeParam, var, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vvv) {
  AgradTest<TypeParam, var, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ddd) {
  AgradTest<TypeParam, double, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ddv) {
  AgradTest<TypeParam, double, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dvd) {  
  AgradTest<TypeParam, double, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dvv) {  
  AgradTest<TypeParam, double, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vdd) {  
  AgradTest<TypeParam, var, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vdv) {
  AgradTest<TypeParam, var, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vvd) {  
  AgradTest<TypeParam, var, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vvv) {
  AgradTest<TypeParam, var, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddd) {
  AgradTest<TypeParam, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddD) {
  AgradTest<TypeParam, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddv) {
  AgradTest<TypeParam, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddV) {
  AgradTest<TypeParam, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDd) {
  AgradTest<TypeParam, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDD) {
  AgradTest<TypeParam, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDv) {
  AgradTest<TypeParam, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDV) {
  AgradTest<TypeParam, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvd) {
  AgradTest<TypeParam, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvD) {
  AgradTest<TypeParam, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvv) {
  AgradTest<TypeParam, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvV) {
  AgradTest<TypeParam, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVd) {
  AgradTest<TypeParam, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVD) {
  AgradTest<TypeParam, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVv) {
  AgradTest<TypeParam, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVV) {
  AgradTest<TypeParam, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ddd) {
  AgradTest<TypeParam, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DdD) {
  AgradTest<TypeParam, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ddv) {
  AgradTest<TypeParam, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DdV) {
  AgradTest<TypeParam, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDd) {
  AgradTest<TypeParam, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDD) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDv) {
  AgradTest<TypeParam, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDV) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Dvd) {
  AgradTest<TypeParam, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DvD) {
  AgradTest<TypeParam, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Dvv) {
  AgradTest<TypeParam, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DvV) {
  AgradTest<TypeParam, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVd) {
  AgradTest<TypeParam, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVD) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVv) {
  AgradTest<TypeParam, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVV) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vdd) {
  AgradTest<TypeParam, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VdD) {
  AgradTest<TypeParam, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vdv) {
  AgradTest<TypeParam, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VdV) {
  AgradTest<TypeParam, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDd) {
  AgradTest<TypeParam, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDD) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDv) {
  AgradTest<TypeParam, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDV) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vvd) {
  AgradTest<TypeParam, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VvD) {
  AgradTest<TypeParam, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vvv) {
  AgradTest<TypeParam, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VvV) {
  AgradTest<TypeParam, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVd) {
  AgradTest<TypeParam, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVD) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVv) {
  AgradTest<TypeParam, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVV) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var> >::test_vectorized();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_ddd,
			   check_valid_ddv,
			   check_valid_dvd,
			   check_valid_dvv,
			   check_valid_vdd,
			   check_valid_vdv,
			   check_valid_vvd,
			   check_valid_vvv,
			   check_invalid_ddd,
			   check_invalid_ddv,
			   check_invalid_dvd,
			   check_invalid_dvv,
			   check_invalid_vdd,
			   check_invalid_vdv,
			   check_invalid_vvd,
			   check_invalid_vvv,
			   logprob_propto_ddd,
			   logprob_propto_ddv,
			   logprob_propto_dvd,
			   logprob_propto_dvv,
			   logprob_propto_vdd,
			   logprob_propto_vdv,
			   logprob_propto_vvd,
			   logprob_propto_vvv,
			   gradient_finite_diff_ddd,
			   gradient_finite_diff_ddv,
			   gradient_finite_diff_dvd,
			   gradient_finite_diff_dvv,
			   gradient_finite_diff_vdd,
			   gradient_finite_diff_vdv,
			   gradient_finite_diff_vvd,
			   gradient_finite_diff_vvv,
			   gradient_function_ddd,
			   gradient_function_ddv,
			   gradient_function_dvd,
			   gradient_function_dvv,
			   gradient_function_vdd,
			   gradient_function_vdv,
			   gradient_function_vvd,
			   gradient_function_vvv);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture2,
			   vectorized_ddd,
			   vectorized_ddD,
			   vectorized_ddv,
			   vectorized_ddV,
			   vectorized_dDd,
			   vectorized_dDD,
			   vectorized_dDv,
			   vectorized_dDV,
			   vectorized_dvd,
			   vectorized_dvD,
			   vectorized_dvv,
			   vectorized_dvV,
			   vectorized_dVd,
			   vectorized_dVD,
			   vectorized_dVv,
			   vectorized_dVV,
			   vectorized_Ddd,
			   vectorized_DdD,
			   vectorized_Ddv,
			   vectorized_DdV,
			   vectorized_DDd,
			   vectorized_DDD,
			   vectorized_DDv,
			   vectorized_DDV,
			   vectorized_Dvd,
			   vectorized_DvD,
			   vectorized_Dvv,
			   vectorized_DvV,
			   vectorized_DVd,
			   vectorized_DVD,
			   vectorized_DVv,
			   vectorized_DVV,
			   vectorized_Vdd,
			   vectorized_VdD,
			   vectorized_Vdv,
			   vectorized_VdV,
			   vectorized_VDd,
			   vectorized_VDD,
			   vectorized_VDv,
			   vectorized_VDV,
			   vectorized_Vvd,
			   vectorized_VvD,
			   vectorized_Vvv,
			   vectorized_VvV,
			   vectorized_VVd,
			   vectorized_VVD,
			   vectorized_VVv,
			   vectorized_VVV);
#endif
