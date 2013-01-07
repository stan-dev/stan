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
TYPED_TEST_P(AgradDistributionTestFixture2, test_dddd) {
  typedef AgradTest<TypeParam, double, double, double, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dddD) {
  typedef AgradTest<TypeParam, double, double, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dddv) {
  typedef AgradTest<TypeParam, double, double, double, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dddV) {
  typedef AgradTest<TypeParam, double, double, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddDd) {
  typedef AgradTest<TypeParam, double, double, vector<double>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddDD) {
  typedef AgradTest<TypeParam, double, double, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddDv) {
  typedef AgradTest<TypeParam, double, double, vector<double>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddDV) {
  typedef AgradTest<TypeParam, double, double, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddvd) {
  typedef AgradTest<TypeParam, double, double, var, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddvD) {
  typedef AgradTest<TypeParam, double, double, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddvv) {
  typedef AgradTest<TypeParam, double, double, var, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddvV) {
  typedef AgradTest<TypeParam, double, double, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddVd) {
  typedef AgradTest<TypeParam, double, double, vector<var>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddVD) {
  typedef AgradTest<TypeParam, double, double, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddVv) {
  typedef AgradTest<TypeParam, double, double, vector<var>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_ddVV) {
  typedef AgradTest<TypeParam, double, double, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDdd) {
  typedef AgradTest<TypeParam, double, vector<double>, double, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDdD) {
  typedef AgradTest<TypeParam, double, vector<double>, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDdv) {
  typedef AgradTest<TypeParam, double, vector<double>, double, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDdV) {
  typedef AgradTest<TypeParam, double, vector<double>, double, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDDd) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<double>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDDD) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<double>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDDv) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<double>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDDV) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<double>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDvd) {
  typedef AgradTest<TypeParam, double, vector<double>, var, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDvD) {
  typedef AgradTest<TypeParam, double, vector<double>, var, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDvv) {
  typedef AgradTest<TypeParam, double, vector<double>, var, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDvV) {
  typedef AgradTest<TypeParam, double, vector<double>, var, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDVd) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<var>, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDVD) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<var>, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDVv) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<var>, var > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dDVV) {
  typedef AgradTest<TypeParam, double, vector<double>, vector<var>, vector<var> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVdd) {
  typedef AgradTest<TypeParam, double, vector<var>, double, double > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVdD) {
  typedef AgradTest<TypeParam, double, vector<var>, double, vector<double> > Test;
  Test::test_valid();
  Test::test_invalid();
  Test::test_propto();
  Test::test_finite_diff();
  Test::test_gradient_function();
  Test::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVdv) {
  AgradTest<TypeParam, double, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVdV) {
  AgradTest<TypeParam, double, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVDd) {
  AgradTest<TypeParam, double, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVDD) {
  AgradTest<TypeParam, double, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVDv) {
  AgradTest<TypeParam, double, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVDV) {
  AgradTest<TypeParam, double, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVvd) {
  AgradTest<TypeParam, double, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVvD) {
  AgradTest<TypeParam, double, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVvv) {
  AgradTest<TypeParam, double, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVvV) {
  AgradTest<TypeParam, double, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVVd) {
  AgradTest<TypeParam, double, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVVD) {
  AgradTest<TypeParam, double, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVVv) {
  AgradTest<TypeParam, double, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture2, test_dVVV) {
  AgradTest<TypeParam, double, vector<var>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Dddd) {
  AgradTest<TypeParam, vector<double>, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DddD) {
  AgradTest<TypeParam, vector<double>, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Dddv) {
  AgradTest<TypeParam, vector<double>, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DddV) {
  AgradTest<TypeParam, vector<double>, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdDd) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdDD) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdDv) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdDV) {
  AgradTest<TypeParam, vector<double>, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Ddvd) {
  AgradTest<TypeParam, vector<double>, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdvD) {
  AgradTest<TypeParam, vector<double>, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_Ddvv) {
  AgradTest<TypeParam, vector<double>, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdvV) {
  AgradTest<TypeParam, vector<double>, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdVd) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdVD) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdVv) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DdVV) {
  AgradTest<TypeParam, vector<double>, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDdd) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDdD) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDdv) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDdV) {
  AgradTest<TypeParam, vector<double>, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDDd) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDDD) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDDv) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDDV) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDvd) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDvD) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDvv) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDvV) {
  AgradTest<TypeParam, vector<double>, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDVd) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDVD) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDVv) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DDVV) {
  AgradTest<TypeParam, vector<double>, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVdd) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVdD) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVdv) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVdV) {
  AgradTest<TypeParam, vector<double>, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVDd) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVDD) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVDv) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVDV) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVvd) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVvD) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVvv) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVvV) {
  AgradTest<TypeParam, vector<double>, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVVd) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVVD) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVVv) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture3, test_DVVV) {
  AgradTest<TypeParam, vector<double>, vector<var>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vddd) {
  AgradTest<TypeParam, var, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vddD) {
  AgradTest<TypeParam, var, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vddv) {
  AgradTest<TypeParam, var, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vddV) {
  AgradTest<TypeParam, var, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdDd) {
  AgradTest<TypeParam, var, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdDD) {
  AgradTest<TypeParam, var, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdDv) {
  AgradTest<TypeParam, var, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdDV) {
  AgradTest<TypeParam, var, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdvd) {
  AgradTest<TypeParam, var, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdvD) {
  AgradTest<TypeParam, var, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdvv) {
  AgradTest<TypeParam, var, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdvV) {
  AgradTest<TypeParam, var, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdVd) {
  AgradTest<TypeParam, var, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdVD) {
  AgradTest<TypeParam, var, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdVv) {
  AgradTest<TypeParam, var, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vdVV) {
  AgradTest<TypeParam, var, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDdd) {
  AgradTest<TypeParam, var, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDdD) {
  AgradTest<TypeParam, var, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDdv) {
  AgradTest<TypeParam, var, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDdV) {
  AgradTest<TypeParam, var, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDDd) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDDD) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDDv) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDDV) {
  AgradTest<TypeParam, var, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDvd) {
  AgradTest<TypeParam, var, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDvD) {
  AgradTest<TypeParam, var, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDvv) {
  AgradTest<TypeParam, var, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDvV) {
  AgradTest<TypeParam, var, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDVd) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDVD) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDVv) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vDVV) {
  AgradTest<TypeParam, var, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVdd) {
  AgradTest<TypeParam, var, vector<var>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVdD) {
  AgradTest<TypeParam, var, vector<var>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVdv) {
  AgradTest<TypeParam, var, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVdV) {
  AgradTest<TypeParam, var, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVDd) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVDD) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVDv) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVDV) {
  AgradTest<TypeParam, var, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVvd) {
  AgradTest<TypeParam, var, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVvD) {
  AgradTest<TypeParam, var, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVvv) {
  AgradTest<TypeParam, var, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVvV) {
  AgradTest<TypeParam, var, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVVd) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVVD) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVVv) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture4, test_vVVV) {
  AgradTest<TypeParam, var, vector<var>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_Vddd) {
  AgradTest<TypeParam, vector<var>, double, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VddD) {
  AgradTest<TypeParam, vector<var>, double, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_Vddv) {
  AgradTest<TypeParam, vector<var>, double, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VddV) {
  AgradTest<TypeParam, vector<var>, double, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdDd) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdDD) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdDv) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdDV) {
  AgradTest<TypeParam, vector<var>, double, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_Vdvd) {
  AgradTest<TypeParam, vector<var>, double, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdvD) {
  AgradTest<TypeParam, vector<var>, double, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_Vdvv) {
  AgradTest<TypeParam, vector<var>, double, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdvV) {
  AgradTest<TypeParam, vector<var>, double, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdVd) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdVD) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdVv) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VdVV) {
  AgradTest<TypeParam, vector<var>, double, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDdd) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDdD) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDdv) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDdV) {
  AgradTest<TypeParam, vector<var>, vector<double>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDDd) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDDD) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDDv) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDDV) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDvd) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDvD) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDvv) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDvV) {
  AgradTest<TypeParam, vector<var>, vector<double>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDVd) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDVD) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDVv) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VDVV) {
  AgradTest<TypeParam, vector<var>, vector<double>, vector<var>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVdd) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVdD) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVdv) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVdV) {
  AgradTest<TypeParam, vector<var>, vector<var>, double, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVDd) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVDD) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVDv) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVDV) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<double>, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVvd) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVvD) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVvv) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVvV) {
  AgradTest<TypeParam, vector<var>, vector<var>, var, vector<var> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVVd) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, double >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVVD) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, vector<double> >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVVv) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, var >::test_vectorized();
}
TYPED_TEST_P(AgradDistributionTestFixture5, test_VVVV) {
  AgradTest<TypeParam, vector<var>, vector<var>, vector<var>, vector<var> >::test_vectorized();
}

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture2,
			   test_dddd,
			   test_dddD,
			   test_dddv,
			   test_dddV,
			   test_ddDd,
			   test_ddDD,
			   test_ddDv,
			   test_ddDV,
			   test_ddvd,
			   test_ddvD,
			   test_ddvv,
			   test_ddvV,
			   test_ddVd,
			   test_ddVD,
			   test_ddVv,
			   test_ddVV,
			   test_dDdd,
			   test_dDdD,
			   test_dDdv,
			   test_dDdV,
			   test_dDDd,
			   test_dDDD,
			   test_dDDv,
			   test_dDDV,
			   test_dDvd,
			   test_dDvD,
			   test_dDvv,
			   test_dDvV,
			   test_dDVd,
			   test_dDVD,
			   test_dDVv,
			   test_dDVV,
			   test_dVdd,
			   test_dVdD,
			   test_dVdv,
			   test_dVdV,
			   test_dVDd,
			   test_dVDD,
			   test_dVDv,
			   test_dVDV,
			   test_dVvd,
			   test_dVvD,
			   test_dVvv,
			   test_dVvV,
			   test_dVVd,
			   test_dVVD,
			   test_dVVv,
			   test_dVVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture3,
			   test_Dddd,
			   test_DddD,
			   test_Dddv,
			   test_DddV,
			   test_DdDd,
			   test_DdDD,
			   test_DdDv,
			   test_DdDV,
			   test_Ddvd,
			   test_DdvD,
			   test_Ddvv,
			   test_DdvV,
			   test_DdVd,
			   test_DdVD,
			   test_DdVv,
			   test_DdVV,
			   test_DDdd,
			   test_DDdD,
			   test_DDdv,
			   test_DDdV,
			   test_DDDd,
			   test_DDDD,
			   test_DDDv,
			   test_DDDV,
			   test_DDvd,
			   test_DDvD,
			   test_DDvv,
			   test_DDvV,
			   test_DDVd,
			   test_DDVD,
			   test_DDVv,
			   test_DDVV,
			   test_DVdd,
			   test_DVdD,
			   test_DVdv,
			   test_DVdV,
			   test_DVDd,
			   test_DVDD,
			   test_DVDv,
			   test_DVDV,
			   test_DVvd,
			   test_DVvD,
			   test_DVvv,
			   test_DVvV,
			   test_DVVd,
			   test_DVVD,
			   test_DVVv,
			   test_DVVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture4,
			   test_vddd,
			   test_vddD,
			   test_vddv,
			   test_vddV,
			   test_vdDd,
			   test_vdDD,
			   test_vdDv,
			   test_vdDV,
			   test_vdvd,
			   test_vdvD,
			   test_vdvv,
			   test_vdvV,
			   test_vdVd,
			   test_vdVD,
			   test_vdVv,
			   test_vdVV,
			   test_vDdd,
			   test_vDdD,
			   test_vDdv,
			   test_vDdV,
			   test_vDDd,
			   test_vDDD,
			   test_vDDv,
			   test_vDDV,
			   test_vDvd,
			   test_vDvD,
			   test_vDvv,
			   test_vDvV,
			   test_vDVd,
			   test_vDVD,
			   test_vDVv,
			   test_vDVV,
			   test_vVdd,
			   test_vVdD,
			   test_vVdv,
			   test_vVdV,
			   test_vVDd,
			   test_vVDD,
			   test_vVDv,
			   test_vVDV,
			   test_vVvd,
			   test_vVvD,
			   test_vVvv,
			   test_vVvV,
			   test_vVVd,
			   test_vVVD,
			   test_vVVv,
			   test_vVVV);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture5,
			   test_Vddd,
			   test_VddD,
			   test_Vddv,
			   test_VddV,
			   test_VdDd,
			   test_VdDD,
			   test_VdDv,
			   test_VdDV,
			   test_Vdvd,
			   test_VdvD,
			   test_Vdvv,
			   test_VdvV,
			   test_VdVd,
			   test_VdVD,
			   test_VdVv,
			   test_VdVV,
			   test_VDdd,
			   test_VDdD,
			   test_VDdv,
			   test_VDdV,
			   test_VDDd,
			   test_VDDD,
			   test_VDDv,
			   test_VDDV,
			   test_VDvd,
			   test_VDvD,
			   test_VDvv,
			   test_VDvV,
			   test_VDVd,
			   test_VDVD,
			   test_VDVv,
			   test_VDVV,
			   test_VVdd,
			   test_VVdD,
			   test_VVdv,
			   test_VVdV,
			   test_VVDd,
			   test_VVDD,
			   test_VVDv,
			   test_VVDV,
			   test_VVvd,
			   test_VVvD,
			   test_VVvv,
			   test_VVvV,
			   test_VVVd,
			   test_VVVD,
			   test_VVVv,
			   test_VVVV);

#endif
