#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>


template<class T0, class T1, class T2>
void update_expected_gradients(var& logprob,
			       vector<double>& grad0, vector<double>& grad1, vector<double>& grad2,
			       T0& p0, T1& p1, T2& p2) {
  vector<var> x;
  if (!is_constant<T0>::value)
    x.push_back(p0);
  if (!is_constant<T1>::value)
    x.push_back(p1);
  if (!is_constant<T2>::value)
    x.push_back(p2);
  vector<double> grad;
  logprob.grad(x, grad);
  if (!is_constant<T0>::value) {
    grad0.push_back(grad[0]);
    grad.erase(grad.begin());
  }
  if (!is_constant<T1>::value) {
    grad1.push_back(grad[0]);
    grad.erase(grad.begin());
  }
  if (!is_constant<T2>::value) {
    grad2.push_back(grad[0]);
    grad.erase(grad.begin());
  }
}
template<class T,
	 bool is_const>
void test_grad(vector<double>& e_grad, vector<double>& grad, size_t p) { }
template<>
void test_grad<var, false>(vector<double>& e_grad, vector<double>& grad, size_t p) {
  double expected_gradient = stan::math::sum(e_grad);
  double gradient = grad[0];

  EXPECT_FLOAT_EQ(expected_gradient, gradient)
    << "Gradient test failed for parameter " << p;
  grad.erase(grad.begin());
}
template<>
void test_grad<vector<var>, false>(vector<double>& e_grad, vector<double>& grad, size_t p) {
  for (size_t n = 0; n < e_grad.size(); n++)
    EXPECT_FLOAT_EQ(e_grad[n], grad[n])
      << "At index " << n << ". Gradient test failed for parameter " << p;
  grad.erase(grad.begin(), grad.begin() + e_grad.size());
}
template<class T0, class T1, class T2>
void test_gradients(var& logprob,
		    vector<double>& e_grad0, vector<double>& e_grad1, vector<double>& e_grad2,
		    T0& p0, T1& p1, T2& p2) {
  vector<var> x;
  add_params<T0, is_constant_struct<T0>::value>(x, p0);
  add_params<T1, is_constant_struct<T1>::value>(x, p1);
  add_params<T2, is_constant_struct<T2>::value>(x, p2);
  vector<double> grad;
  logprob.grad(x, grad);
  
  test_grad<T0, is_constant_struct<T0>::value>(e_grad0, grad, 0);
  test_grad<T1, is_constant_struct<T1>::value>(e_grad1, grad, 1);
  test_grad<T2, is_constant_struct<T2>::value>(e_grad2, grad, 2);
}

template<class T0, class T1, class T2, class TypeParam>
void test_vectorized() {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  ASSERT_EQ(parameters[0].size(), 3U);
  
  if (is_constant_struct<T0>::value && is_constant_struct<T1>::value && is_constant_struct<T2>::value) {
    SUCCEED() << "No need to test all double arguments";
    return;
  }
  if (!is_vector<T0>::value && !is_vector<T1>::value && !is_vector<T2>::value) {
    SUCCEED() << "No need to test all non-vector arguments";
    return;
  }

  double e_logprob(0.0);
  vector<double> e_grad_p0, e_grad_p1, e_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    typename scalar_type<T0>::type p0 = get_param<T0>(parameters, n, 0);
    typename scalar_type<T1>::type p1 = get_param<T1>(parameters, n, 1);
    typename scalar_type<T2>::type p2 = get_param<T2>(parameters, n, 2);
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    e_logprob += logprob.val();
    update_expected_gradients(logprob,
			      e_grad_p0, e_grad_p1, e_grad_p2,
			      p0, p1, p2);
  }
  T0 p0 = get_params<T0>(parameters, 0);
  T1 p1 = get_params<T1>(parameters, 1);
  T2 p2 = get_params<T2>(parameters, 2);
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(e_logprob, logprob.val())
    << "log probability does not match";

  test_gradients(logprob,
		 e_grad_p0, e_grad_p1, e_grad_p2,
		 p0, p1, p2);
  return;
}

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

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
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  params[2]))
      << "Failed with (d,d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (d,d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  var(params[2])))
      << "Failed with (d,d,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  params[2]))
      << "Failed with (d,v,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  var(params[2])))
      << "Failed with (d,v,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  params[2]))
      << "Failed with (v,d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  var(params[2])))
      << "Failed with (v,d,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  params[2]))
      << "Failed with (d,v,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  var(params[2])))
      << "Failed with (v,v,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  invalid_params[1],
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1],
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddv) {
  vector<size_t> index;
  vector<double> invalid_values;
  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvd) {  
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvv) {  
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  invalid_params[1],
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdv) {  
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvv) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i      
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ddd) {
  var logprob_true = _LOG_PROB_<true>(this->first_valid_params()[0],
				      this->first_valid_params()[1],
				      this->first_valid_params()[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1],
					 params[2]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ddv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					params[1],
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      params[1],
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   params[1],
					   var(params[2]));
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1],
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dvd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					params[2]);
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      params[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]),
					   params[2]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]),
					 params[2]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dvv) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]),
					   var(params[2]));
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]),
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vdd) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					params[2]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      params[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1],
					   params[2]);
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1],
					 params[2]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vdv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1],
					   var(params[2]));
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1],
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vvd) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					params[2]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      params[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]),
					   params[2]);
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]),
					 params[2]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vvv) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]),
					   var(params[2]));
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]),
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ddd) {
  SUCCEED() << "No op for all double" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e) - _LOG_PROB_<false>(p[0], p[1], p[2]-e)) / e_times_2;
      
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(p[0], p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g2,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;


  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2]) - _LOG_PROB_<false>(p[0], p[1]-e, p[2])) / e_times_2;
      
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(p[0], p1, p[2]);
    vector<var> v_params(1);
    v_params[0] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2]) - _LOG_PROB_<false>(p[0], p[1]-e, p[2])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e) - _LOG_PROB_<false>(p[0], p[1], p[2]-e)) / e_times_2;
      
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p2);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vdd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2]) - _LOG_PROB_<false>(p[0]-e, p[1], p[2])) / e_times_2;
      
    var p0(p[0]);
    var lp = _LOG_PROB_<true>(p0, p[1], p[2]);
    vector<var> v_params(1);
    v_params[0] = p0;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2]) - _LOG_PROB_<false>(p[0]-e, p[1], p[2])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e) - _LOG_PROB_<false>(p[0], p[1], p[2]-e)) / e_times_2;
      
    var p0(p[0]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p2);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2]) - _LOG_PROB_<false>(p[0]-e, p[1], p[2])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2]) - _LOG_PROB_<false>(p[0], p[1]-e, p[2])) / e_times_2;
      
    var p0(p[0]);
    var p1(p[1]);
    double p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2]) - _LOG_PROB_<false>(p[0]-e, p[1], p[2])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2]) - _LOG_PROB_<false>(p[0], p[1]-e, p[2])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e) - _LOG_PROB_<false>(p[0], p[1], p[2]-e)) / e_times_2;
      
    var p0(p[0]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> v_params(3);
    v_params[0] = p0;
    v_params[1] = p1;
    v_params[2] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ddd) {
  SUCCEED() << "No op for (d,d,d) input" << std::endl;
}

TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ddv) {
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
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dvd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p[2]);
    var expected_lp = TypeParam().log_prob(p[0], p1, p[2]);
    vector<var> v_params(1);
    v_params[0] = p1;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dvv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p2);
    var expected_lp = TypeParam().log_prob(p[0], p1, p2);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vdd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p[2]);
    var expected_lp = TypeParam().log_prob(p0, p[1], p[2]);
    vector<var> v_params(1);
    v_params[0] = p0;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p2);
    var expected_lp = TypeParam().log_prob(p0, p[1], p2);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p2;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vvd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p[2]);
    var expected_lp = TypeParam().log_prob(p0, p1, p[2]);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p2);
    var expected_lp = TypeParam().log_prob(p0, p1, p2);
    vector<var> v_params(3);
    v_params[0] = p0;
    v_params[1] = p1;
    v_params[2] = p2;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}

TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddd) {
  test_vectorized<double, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddD) {
  test_vectorized<double, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddv) {
  test_vectorized<double, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_ddV) {
  test_vectorized<double, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDd) {
 test_vectorized<double, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDD) {
  test_vectorized<double, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDv) {
  test_vectorized<double, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dDV) {
  test_vectorized<double, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvd) {
  test_vectorized<double, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvD) {
  test_vectorized<double, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvv) {
  test_vectorized<double, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dvV) {
  test_vectorized<double, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVd) {
  test_vectorized<double, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVD) {
  test_vectorized<double, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVv) {
  test_vectorized<double, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_dVV) {
  test_vectorized<double, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ddd) {
  test_vectorized<vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DdD) {
  test_vectorized<vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ddv) {
  test_vectorized<vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DdV) {
  test_vectorized<vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDd) {
  test_vectorized<vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDD) {
  test_vectorized<vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDv) {
  test_vectorized<vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDV) {
  test_vectorized<vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Dvd) {
  test_vectorized<vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DvD) {
  test_vectorized<vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Dvv) {
  test_vectorized<vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DvV) {
  test_vectorized<vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVd) {
  test_vectorized<vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVD) {
  test_vectorized<vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVv) {
  test_vectorized<vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVV) {
  test_vectorized<vector<double>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vdd) {
  test_vectorized<vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VdD) {
  test_vectorized<vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vdv) {
  test_vectorized<vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VdV) {
  test_vectorized<vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDd) {
  test_vectorized<vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDD) {
  test_vectorized<vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDv) {
  test_vectorized<vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDV) {
  test_vectorized<vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vvd) {
  test_vectorized<vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VvD) {
  test_vectorized<vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vvv) {
  test_vectorized<vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VvV) {
  test_vectorized<vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVd) {
  test_vectorized<vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVD) {
  test_vectorized<vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVv) {
  test_vectorized<vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVV) {
  test_vectorized<vector<double>, vector<var>, vector<var>, TypeParam>();
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
