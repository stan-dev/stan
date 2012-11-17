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
  test_valid<TypeParam, int, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iiv) {
  test_valid<TypeParam, int, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iid) {
  test_invalid<TypeParam, int, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iiv) {
  test_invalid<TypeParam, int, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iid) {
  test_propto<TypeParam, int, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iiv) {
  test_propto<TypeParam, int, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iid) {
  SUCCEED() << "No op for all constants" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iiv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g2 = (_LOG_PROB_<false>(int(p[0]), int(p[1]), p[2]+e) - _LOG_PROB_<false>(int(p[0]), int(p[1]), p[2]-e)) / e_times_2;
      
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(p[0], p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g2,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << int(p[1]) << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iid) {
  SUCCEED() << "No op for (i,i,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iiv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p[1], p2);
    var expected_lp = TypeParam().log_prob(p[0], p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iid) {
  test_vectorized<TypeParam, int, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiv) {
  test_vectorized<TypeParam, int, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiD) {
  test_vectorized<TypeParam, int, int, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiV) {
  test_vectorized<TypeParam, int, int, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iId) {
  test_vectorized<TypeParam, int, vector<int>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iIv) {
  test_vectorized<TypeParam, int, vector<int>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iID) {
  test_vectorized<TypeParam, int, vector<int>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iIV) {
  test_vectorized<TypeParam, int, vector<int>, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iid) {
  test_vectorized<TypeParam, vector<int>, int, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iiv) {
  test_vectorized<TypeParam, vector<int>, int, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IiD) {
  test_vectorized<TypeParam, vector<int>, int, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IiV) {
  test_vectorized<TypeParam, vector<int>, int, vector<var> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IId) {
  test_vectorized<TypeParam, vector<int>, vector<int>, double>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IIv) {
  test_vectorized<TypeParam, vector<int>, vector<int>, var>();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IID) {
  test_vectorized<TypeParam, vector<int>, vector<int>, vector<double> >();
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IIV) {
  test_vectorized<TypeParam, vector<int>, vector<int>, vector<var> >();
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
