#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_4_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_4_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>
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
  ASSERT_EQ(parameters.size(), 4U);

  var param1, param2, param3, param4;
  var logprob;
  param1 = parameters[0];
  param2 = parameters[1];
  param3 = parameters[2];
  param4 = parameters[3];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, param3, param4));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, param3, param4));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, param3, param4, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, param3, param4, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, param3, param4));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, param3, param4, errno_policy()));
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dddd) {
  AgradTest<TypeParam, double, double, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dddv) {
  AgradTest<TypeParam, double, double, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddvd) {
  AgradTest<TypeParam, double, double, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddvv) {
  AgradTest<TypeParam, double, double, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvdd) {
  AgradTest<TypeParam, double, var, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvdv) {
  AgradTest<TypeParam, double, var, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvvd) {
  AgradTest<TypeParam, double, var, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvvv) {
  AgradTest<TypeParam, double, var, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vddd) {
  AgradTest<TypeParam, var, double, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vddv) {
  AgradTest<TypeParam, var, double, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdvd) {
  AgradTest<TypeParam, var, double, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdvv) {
  AgradTest<TypeParam, var, double, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvdd) {
  AgradTest<TypeParam, var, var, double, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvdv) {
  AgradTest<TypeParam, var, var, double, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvvd) {
  AgradTest<TypeParam, var, var, var, double>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvvv) {
  AgradTest<TypeParam, var, var, var, var>::test_valid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dddd) {
  AgradTest<TypeParam, double, double, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dddv) {
  AgradTest<TypeParam, double, double, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddvd) {
  AgradTest<TypeParam, double, double, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddvv) {
  AgradTest<TypeParam, double, double, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvdd) {
  AgradTest<TypeParam, double, var, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvdv) {
  AgradTest<TypeParam, double, var, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvvd) {
  AgradTest<TypeParam, double, var, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvvv) {
  AgradTest<TypeParam, double, var, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vddd) {
  AgradTest<TypeParam, var, double, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vddv) {
  AgradTest<TypeParam, var, double, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdvd) {
  AgradTest<TypeParam, var, double, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdvv) {
  AgradTest<TypeParam, var, double, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvdd) {
  AgradTest<TypeParam, var, var, double, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvdv) {
  AgradTest<TypeParam, var, var, double, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvvd) {
  AgradTest<TypeParam, var, var, var, double>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvvv) {
  AgradTest<TypeParam, var, var, var, var>::test_invalid();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dddd) {
  AgradTest<TypeParam, double, double, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dddv) {
  AgradTest<TypeParam, double, double, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_ddvd) {
  AgradTest<TypeParam, double, double, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_ddvv) {
  AgradTest<TypeParam, double, double, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvdd) {
  AgradTest<TypeParam, double, var, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvdv) {
  AgradTest<TypeParam, double, var, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvvd) {
  AgradTest<TypeParam, double, var, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvvv) {
  AgradTest<TypeParam, double, var, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vddd) {
  AgradTest<TypeParam, var, double, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vddv) {
  AgradTest<TypeParam, var, double, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vdvd) {
  AgradTest<TypeParam, var, double, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vdvv) {
  AgradTest<TypeParam, var, double, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvdd) {
  AgradTest<TypeParam, var, var, double, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvdv) {
  AgradTest<TypeParam, var, var, double, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvvd) {
  AgradTest<TypeParam, var, var, var, double>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvvv) {
  AgradTest<TypeParam, var, var, var, var>::test_propto();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dddd) {
  AgradTest<TypeParam, double, double, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dddv) {
  AgradTest<TypeParam, double, double, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_ddvd) {
  AgradTest<TypeParam, double, double, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_ddvv) {
  AgradTest<TypeParam, double, double, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvdd) {
  AgradTest<TypeParam, double, var, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvdv) {
  AgradTest<TypeParam, double, var, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvvd) {
  AgradTest<TypeParam, double, var, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvvv) {
  AgradTest<TypeParam, double, var, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vddd) {
  AgradTest<TypeParam, var, double, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vddv) {
  AgradTest<TypeParam, var, double, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vdvd) {
  AgradTest<TypeParam, var, double, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vdvv) {
  AgradTest<TypeParam, var, double, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvdd) {
  AgradTest<TypeParam, var, var, double, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvdv) {
  AgradTest<TypeParam, var, var, double, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvvd) {
  AgradTest<TypeParam, var, var, var, double>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvvv) {
  AgradTest<TypeParam, var, var, var, var>::test_finite_diff();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dddd) {
  AgradTest<TypeParam, double, double, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dddv) {
  AgradTest<TypeParam, double, double, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_ddvd) {  
  AgradTest<TypeParam, double, double, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_ddvv) {  
  AgradTest<TypeParam, double, double, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvdd) {  
  AgradTest<TypeParam, double, var, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvdv) {  
  AgradTest<TypeParam, double, var, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvvd) {  
  AgradTest<TypeParam, double, var, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvvv) {  
  AgradTest<TypeParam, double, var, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vddd) {
  AgradTest<TypeParam, var, double, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vddv) {
  AgradTest<TypeParam, var, double, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vdvd) {  
  AgradTest<TypeParam, var, double, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vdvv) {  
  AgradTest<TypeParam, var, double, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvdd) {  
  AgradTest<TypeParam, var, var, double, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvdv) {  
  AgradTest<TypeParam, var, var, double, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvvd) {  
  AgradTest<TypeParam, var, var, var, double>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvvv) {  
  AgradTest<TypeParam, var, var, var, var>::test_gradient_function();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddd) {
  AgradTest<TypeParam, double, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddD) {
  AgradTest<TypeParam, double, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddv) {
  AgradTest<TypeParam, double, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddV) {
  AgradTest<TypeParam, double, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDd) {
  AgradTest<TypeParam, double, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDD) {
  AgradTest<TypeParam, double, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDv) {
  AgradTest<TypeParam, double, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDV) {
  AgradTest<TypeParam, double, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvd) {
  AgradTest<TypeParam, double, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvD) {
  AgradTest<TypeParam, double, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvv) {
  AgradTest<TypeParam, double, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvV) {
  AgradTest<TypeParam, double, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVd) {
  AgradTest<TypeParam, double, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVD) {
  AgradTest<TypeParam, double, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVv) {
  AgradTest<TypeParam, double, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVV) {
  AgradTest<TypeParam, double, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdd) {
  AgradTest<TypeParam, double, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdD) {
  AgradTest<TypeParam, double, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdv) {
  AgradTest<TypeParam, double, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdV) {
  AgradTest<TypeParam, double, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDd) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDD) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDv) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDV) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvd) {
  AgradTest<TypeParam, double, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvD) {
  AgradTest<TypeParam, double, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvv) {
  AgradTest<TypeParam, double, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvV) {
  AgradTest<TypeParam, double, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVd) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVD) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVv) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVV) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdd) {
  AgradTest<TypeParam, double, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdD) {
  AgradTest<TypeParam, double, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdv) {
  AgradTest<TypeParam, double, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdV) {
  AgradTest<TypeParam, double, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDd) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDD) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDv) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDV) {
  AgradTest<TypeParam, double, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvd) {
  AgradTest<TypeParam, double, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvD) {
  AgradTest<TypeParam, double, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvv) {
  AgradTest<TypeParam, double, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvV) {
  AgradTest<TypeParam, double, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVd) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVD) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVv) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVV) {
  AgradTest<TypeParam, double, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Dddd) {
  AgradTest<TypeParam, vector<double>, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DddD) {
  AgradTest<TypeParam, vector<double>, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Dddv) {
  AgradTest<TypeParam, vector<double>, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DddV) {
  AgradTest<TypeParam, vector<double>, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDd) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDD) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDv) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDV) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Ddvd) {
  AgradTest<TypeParam, vector<double>, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdvD) {
  AgradTest<TypeParam, vector<double>, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Ddvv) {
  AgradTest<TypeParam, vector<double>, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdvV) {
  AgradTest<TypeParam, vector<double>, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVd) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVD) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVv) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVV) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdd) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdD) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdv) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdV) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDd) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDD) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDv) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDV) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvd) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvD) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvv) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvV) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVd) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVD) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVv) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVV) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdd) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdD) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdv) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdV) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDd) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDD) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDv) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDV) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvd) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvD) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvv) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvV) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVd) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVD) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVv) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVV) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddd) {
  AgradTest<TypeParam, var, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddD) {
  AgradTest<TypeParam, var, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddv) {
  AgradTest<TypeParam, var, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddV) {
  AgradTest<TypeParam, var, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDd) {
  AgradTest<TypeParam, var, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDD) {
  AgradTest<TypeParam, var, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDv) {
  AgradTest<TypeParam, var, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDV) {
  AgradTest<TypeParam, var, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvd) {
  AgradTest<TypeParam, var, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvD) {
  AgradTest<TypeParam, var, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvv) {
  AgradTest<TypeParam, var, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvV) {
  AgradTest<TypeParam, var, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVd) {
  AgradTest<TypeParam, var, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVD) {
  AgradTest<TypeParam, var, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVv) {
  AgradTest<TypeParam, var, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVV) {
  AgradTest<TypeParam, var, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdd) {
  AgradTest<TypeParam, var, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdD) {
  AgradTest<TypeParam, var, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdv) {
  AgradTest<TypeParam, var, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdV) {
  AgradTest<TypeParam, var, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDd) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDD) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDv) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDV) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvd) {
  AgradTest<TypeParam, var, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvD) {
  AgradTest<TypeParam, var, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvv) {
  AgradTest<TypeParam, var, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvV) {
  AgradTest<TypeParam, var, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVd) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVD) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVv) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVV) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdd) {
  AgradTest<TypeParam, var, vector<var>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdD) {
  AgradTest<TypeParam, var, vector<var>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdv) {
  AgradTest<TypeParam, var, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdV) {
  AgradTest<TypeParam, var, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDd) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDD) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDv) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDV) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvd) {
  AgradTest<TypeParam, var, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvD) {
  AgradTest<TypeParam, var, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvv) {
  AgradTest<TypeParam, var, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvV) {
  AgradTest<TypeParam, var, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVd) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVD) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVv) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVV) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vddd) {
  AgradTest<TypeParam, vector<var>, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VddD) {
  AgradTest<TypeParam, vector<var>, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vddv) {
  AgradTest<TypeParam, vector<var>, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VddV) {
  AgradTest<TypeParam, vector<var>, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDd) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDD) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDv) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDV) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vdvd) {
  AgradTest<TypeParam, vector<var>, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdvD) {
  AgradTest<TypeParam, vector<var>, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vdvv) {
  AgradTest<TypeParam, vector<var>, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdvV) {
  AgradTest<TypeParam, vector<var>, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVd) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVD) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVv) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVV) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdd) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdD) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdv) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdV) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDd) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDD) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDv) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDV) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvd) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvD) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvv) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvV) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVd) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVD) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVv) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVV) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdd) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdD) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdv) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdV) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDd) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDD) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDv) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDV) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvd) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvD) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvv) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvV) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVd) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVD) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVv) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVV) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, vector<var> >::test_vectorized();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_dddd,
			   check_valid_dddv,
			   check_valid_ddvd,
			   check_valid_ddvv,
			   check_valid_dvdd,
			   check_valid_dvdv,
			   check_valid_dvvd,
			   check_valid_dvvv,
			   check_valid_vddd,
			   check_valid_vddv,
			   check_valid_vdvd,
			   check_valid_vdvv,
			   check_valid_vvdd,
			   check_valid_vvdv,
			   check_valid_vvvd,
			   check_valid_vvvv,
			   check_invalid_dddd,
			   check_invalid_dddv,
			   check_invalid_ddvd,
			   check_invalid_ddvv,
			   check_invalid_dvdd,
			   check_invalid_dvdv,
			   check_invalid_dvvd,
			   check_invalid_dvvv,
			   check_invalid_vddd,
			   check_invalid_vddv,
			   check_invalid_vdvd,
			   check_invalid_vdvv,
			   check_invalid_vvdd,
			   check_invalid_vvdv,
			   check_invalid_vvvd,
			   check_invalid_vvvv);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture2,
			   logprob_propto_dddd,
			   logprob_propto_dddv,
			   logprob_propto_ddvd,
			   logprob_propto_ddvv,
			   logprob_propto_dvdd,
			   logprob_propto_dvdv,
			   logprob_propto_dvvd,
			   logprob_propto_dvvv,
			   logprob_propto_vddd,
			   logprob_propto_vddv,
			   logprob_propto_vdvd,
			   logprob_propto_vdvv,
			   logprob_propto_vvdd,
			   logprob_propto_vvdv,
			   logprob_propto_vvvd,
			   logprob_propto_vvvv,
			   gradient_finite_diff_dddd,
			   gradient_finite_diff_dddv,
			   gradient_finite_diff_ddvd,
			   gradient_finite_diff_ddvv,
			   gradient_finite_diff_dvdd,
			   gradient_finite_diff_dvdv,
			   gradient_finite_diff_dvvd,
			   gradient_finite_diff_dvvv,
			   gradient_finite_diff_vddd,
			   gradient_finite_diff_vddv,
			   gradient_finite_diff_vdvd,
			   gradient_finite_diff_vdvv,
			   gradient_finite_diff_vvdd,
			   gradient_finite_diff_vvdv,
			   gradient_finite_diff_vvvd,
			   gradient_finite_diff_vvvv,
			   gradient_function_dddd,
			   gradient_function_dddv,
			   gradient_function_ddvd,
			   gradient_function_ddvv,
			   gradient_function_dvdd,
			   gradient_function_dvdv,
			   gradient_function_dvvd,
			   gradient_function_dvvv,
			   gradient_function_vddd,
			   gradient_function_vddv,
			   gradient_function_vdvd,
			   gradient_function_vdvv,
			   gradient_function_vvdd,
			   gradient_function_vvdv,
			   gradient_function_vvvd,
			   gradient_function_vvvv);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture3,
			   vectorized_dddd,
			   vectorized_dddD,
			   vectorized_dddv,
			   vectorized_dddV,
			   vectorized_ddDd,
			   vectorized_ddDD,
			   vectorized_ddDv,
			   vectorized_ddDV,
			   vectorized_ddvd,
			   vectorized_ddvD,
			   vectorized_ddvv,
			   vectorized_ddvV,
			   vectorized_ddVd,
			   vectorized_ddVD,
			   vectorized_ddVv,
			   vectorized_ddVV,
			   vectorized_dDdd,
			   vectorized_dDdD,
			   vectorized_dDdv,
			   vectorized_dDdV,
			   vectorized_dDDd,
			   vectorized_dDDD,
			   vectorized_dDDv,
			   vectorized_dDDV,
			   vectorized_dDvd,
			   vectorized_dDvD,
			   vectorized_dDvv,
			   vectorized_dDvV,
			   vectorized_dDVd,
			   vectorized_dDVD,
			   vectorized_dDVv,
			   vectorized_dDVV,
			   vectorized_dVdd,
			   vectorized_dVdD,
			   vectorized_dVdv,
			   vectorized_dVdV,
			   vectorized_dVDd,
			   vectorized_dVDD,
			   vectorized_dVDv,
			   vectorized_dVDV,
			   vectorized_dVvd,
			   vectorized_dVvD,
			   vectorized_dVvv,
			   vectorized_dVvV,
			   vectorized_dVVd,
			   vectorized_dVVD,
			   vectorized_dVVv,
			   vectorized_dVVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture4,
			   vectorized_Dddd,
			   vectorized_DddD,
			   vectorized_Dddv,
			   vectorized_DddV,
			   vectorized_DdDd,
			   vectorized_DdDD,
			   vectorized_DdDv,
			   vectorized_DdDV,
			   vectorized_Ddvd,
			   vectorized_DdvD,
			   vectorized_Ddvv,
			   vectorized_DdvV,
			   vectorized_DdVd,
			   vectorized_DdVD,
			   vectorized_DdVv,
			   vectorized_DdVV,
			   vectorized_DDdd,
			   vectorized_DDdD,
			   vectorized_DDdv,
			   vectorized_DDdV,
			   vectorized_DDDd,
			   vectorized_DDDD,
			   vectorized_DDDv,
			   vectorized_DDDV,
			   vectorized_DDvd,
			   vectorized_DDvD,
			   vectorized_DDvv,
			   vectorized_DDvV,
			   vectorized_DDVd,
			   vectorized_DDVD,
			   vectorized_DDVv,
			   vectorized_DDVV,
			   vectorized_DVdd,
			   vectorized_DVdD,
			   vectorized_DVdv,
			   vectorized_DVdV,
			   vectorized_DVDd,
			   vectorized_DVDD,
			   vectorized_DVDv,
			   vectorized_DVDV,
			   vectorized_DVvd,
			   vectorized_DVvD,
			   vectorized_DVvv,
			   vectorized_DVvV,
			   vectorized_DVVd,
			   vectorized_DVVD,
			   vectorized_DVVv,
			   vectorized_DVVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture5,
			   vectorized_vddd,
			   vectorized_vddD,
			   vectorized_vddv,
			   vectorized_vddV,
			   vectorized_vdDd,
			   vectorized_vdDD,
			   vectorized_vdDv,
			   vectorized_vdDV,
			   vectorized_vdvd,
			   vectorized_vdvD,
			   vectorized_vdvv,
			   vectorized_vdvV,
			   vectorized_vdVd,
			   vectorized_vdVD,
			   vectorized_vdVv,
			   vectorized_vdVV,
			   vectorized_vDdd,
			   vectorized_vDdD,
			   vectorized_vDdv,
			   vectorized_vDdV,
			   vectorized_vDDd,
			   vectorized_vDDD,
			   vectorized_vDDv,
			   vectorized_vDDV,
			   vectorized_vDvd,
			   vectorized_vDvD,
			   vectorized_vDvv,
			   vectorized_vDvV,
			   vectorized_vDVd,
			   vectorized_vDVD,
			   vectorized_vDVv,
			   vectorized_vDVV,
			   vectorized_vVdd,
			   vectorized_vVdD,
			   vectorized_vVdv,
			   vectorized_vVdV,
			   vectorized_vVDd,
			   vectorized_vVDD,
			   vectorized_vVDv,
			   vectorized_vVDV,
			   vectorized_vVvd,
			   vectorized_vVvD,
			   vectorized_vVvv,
			   vectorized_vVvV,
			   vectorized_vVVd,
			   vectorized_vVVD,
			   vectorized_vVVv,
			   vectorized_vVVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture6,
			   vectorized_Vddd,
			   vectorized_VddD,
			   vectorized_Vddv,
			   vectorized_VddV,
			   vectorized_VdDd,
			   vectorized_VdDD,
			   vectorized_VdDv,
			   vectorized_VdDV,
			   vectorized_Vdvd,
			   vectorized_VdvD,
			   vectorized_Vdvv,
			   vectorized_VdvV,
			   vectorized_VdVd,
			   vectorized_VdVD,
			   vectorized_VdVv,
			   vectorized_VdVV,
			   vectorized_VDdd,
			   vectorized_VDdD,
			   vectorized_VDdv,
			   vectorized_VDdV,
			   vectorized_VDDd,
			   vectorized_VDDD,
			   vectorized_VDDv,
			   vectorized_VDDV,
			   vectorized_VDvd,
			   vectorized_VDvD,
			   vectorized_VDvv,
			   vectorized_VDvV,
			   vectorized_VDVd,
			   vectorized_VDVD,
			   vectorized_VDVv,
			   vectorized_VDVV,
			   vectorized_VVdd,
			   vectorized_VVdD,
			   vectorized_VVdv,
			   vectorized_VVdV,
			   vectorized_VVDd,
			   vectorized_VVDD,
			   vectorized_VVDv,
			   vectorized_VVDV,
			   vectorized_VVvd,
			   vectorized_VVvD,
			   vectorized_VVvv,
			   vectorized_VVvV,
			   vectorized_VVVd,
			   vectorized_VVVD,
			   vectorized_VVVv,
			   vectorized_VVVV);

#endif
