#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_4_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_4_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>

using stan::agrad::var;
using stan::scalar_type;
using stan::is_vector;
using stan::is_constant;
using stan::is_constant_struct;

template<class T>
T get_params(vector<vector<double> >& parameters, size_t p) {
  return parameters[0][p];
}
template<>
vector<double> get_params<vector<double> >(vector<vector<double> >& parameters, size_t p) {
  vector<double> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    param[n] = parameters[n][p];
  return param;
}
template<>
vector<var> get_params<vector<var> >(vector<vector<double> >& parameters, size_t p) {
  vector<var> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    param[n] = parameters[n][p];
  return param;
}
template<class T>
double get_param(vector<vector<double> >& parameters, size_t n, size_t p) {
  if (is_vector<T>::value)
    return parameters[n][p];
  else
    return parameters[0][p];
}
template<class T0, class T1, class T2, class T3>
void update_expected_gradients(var& logprob,
			       vector<double>& grad0, vector<double>& grad1,
			       vector<double>& grad2, vector<double>& grad3,
			       T0& p0, T1& p1,
			       T2& p2, T3& p3) {
  vector<var> x;
  if (!is_constant<T0>::value)
    x.push_back(p0);
  if (!is_constant<T1>::value)
    x.push_back(p1);
  if (!is_constant<T2>::value)
    x.push_back(p2);
  if (!is_constant<T3>::value)
    x.push_back(p3);
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
  if (!is_constant<T3>::value) {
    grad3.push_back(grad[0]);
    grad.erase(grad.begin());
  }
}
template<class T, 
	 bool is_const>
void add_params(vector<var>& x, T& p) { }
template<>
void add_params<var, false>(vector<var>& x, var& p) {
  x.push_back(p);
}
template<>
void add_params<vector<var>, false>(vector<var>& x, vector<var>& p) {
  x.insert(x.end(), p.begin(), p.end());
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
template<class T0, class T1, class T2, class T3>
void test_gradients(var& logprob,
		    vector<double>& e_grad0, vector<double>& e_grad1,
		    vector<double>& e_grad2, vector<double>& e_grad3,
		    T0& p0, T1& p1,
		    T2& p2, T3& p3) {
  vector<var> x;
  add_params<T0, is_constant_struct<T0>::value>(x, p0);
  add_params<T1, is_constant_struct<T1>::value>(x, p1);
  add_params<T2, is_constant_struct<T2>::value>(x, p2);
  add_params<T3, is_constant_struct<T3>::value>(x, p3);
  vector<double> grad;
  logprob.grad(x, grad);
  
  test_grad<T0, is_constant_struct<T0>::value>(e_grad0, grad, 0);
  test_grad<T1, is_constant_struct<T1>::value>(e_grad1, grad, 1);
  test_grad<T2, is_constant_struct<T2>::value>(e_grad2, grad, 2);
  test_grad<T3, is_constant_struct<T3>::value>(e_grad3, grad, 3);
}
template<class T0, class T1, class T2, class T3, class TypeParam>
void test_vectorized() {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  ASSERT_EQ(parameters[0].size(), 4U);
  
  if (is_constant_struct<T0>::value && is_constant_struct<T1>::value
      && is_constant_struct<T2>::value && is_constant_struct<T3>::value) {
    SUCCEED() << "No need to test all double arguments";
    return;
  }
  if (!is_vector<T0>::value && !is_vector<T1>::value
      && !is_vector<T2>::value && !is_vector<T3>::value) {
    SUCCEED() << "No need to test all non-vector arguments";
    return;
  }

  double e_logprob(0.0);
  vector<double> e_grad_p0, e_grad_p1, e_grad_p2, e_grad_p3;
  for (size_t n = 0; n < parameters.size(); n++) {
    typename scalar_type<T0>::type p0 = get_param<T0>(parameters, n, 0);
    typename scalar_type<T1>::type p1 = get_param<T1>(parameters, n, 1);
    typename scalar_type<T2>::type p2 = get_param<T2>(parameters, n, 2);
    typename scalar_type<T3>::type p3 = get_param<T3>(parameters, n, 3);
    var logprob = _LOG_PROB_<true>(p0, p1, p2, p3);
    e_logprob += logprob.val();
    update_expected_gradients(logprob,
			      e_grad_p0, e_grad_p1, e_grad_p2, e_grad_p3,
			      p0, p1, p2, p3);
  }
  T0 p0 = get_params<T0>(parameters, 0);
  T1 p1 = get_params<T1>(parameters, 1);
  T2 p2 = get_params<T2>(parameters, 2);
  T3 p3 = get_params<T3>(parameters, 3);
  var logprob = _LOG_PROB_<true>(p0, p1, p2, p3);
  EXPECT_FLOAT_EQ(e_logprob, logprob.val())
    << "log probability does not match";

  test_gradients(logprob,
		 e_grad_p0, e_grad_p1, e_grad_p2, e_grad_p3,
		 p0, p1, p2, p3);
  return;
}

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

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
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  params[2],
					  params[3]))
      << "Failed with (d,d,d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1]
      << ", "<< params[2] << ", " << params[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (d,d,d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1]
      << ", "<< params[2] << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  params[2],
					  var(params[3])))
      << "Failed with (d,d,d,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1]
      << ", "<< params[2] << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  var(params[2]),
					  params[3]))
      << "Failed with (d,d,v,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1]
      << ", "<< var(params[2]) << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  var(params[2]),
					  var(params[3])))
      << "Failed with (d,d,v,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1]
      << ", "<< var(params[2]) << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvdd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  params[2],
					  params[3]))
      << "Failed with (d,v,d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << var(params[1])
      << ", "<< params[2] << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  params[2],
					  var(params[3])))
      << "Failed with (d,v,d,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << var(params[1])
      << ", "<< params[2] << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  var(params[2]),
					  params[3]))
      << "Failed with (d,v,v,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << var(params[1])
      << ", "<< var(params[2]) << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  var(params[2]),
					  var(params[3])))
      << "Failed with (d,v,v,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << var(params[1])
      << ", "<< var(params[2]) << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vddd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  params[2],
					  params[3]))
      << "Failed with (v,d,d,d) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << params[1]
      << ", "<< params[2] << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  params[2],
					  var(params[3])))
      << "Failed with (v,d,d,v) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << params[1]
      << ", "<< params[2] << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  var(params[2]),
					  params[3]))
      << "Failed with (v,d,v,d) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << params[1]
      << ", "<< var(params[2]) << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  var(params[2]),
					  var(params[3])))
      << "Failed with (v,d,v,v) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << params[1]
      << ", "<< var(params[2]) << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvdd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  params[2],
					  params[3]))
      << "Failed with (v,v,d,d) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << var(params[1])
      << ", "<< params[2] << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  params[2],
					  var(params[3])))
      << "Failed with (v,v,d,v) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << var(params[1])
      << ", "<< params[2] << ", " << var(params[3]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  var(params[2]),
					  params[3]))
      << "Failed with (v,v,v,d) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << var(params[1])
      << ", "<< var(params[2]) << ", " << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  var(params[2]),
					  var(params[3])))
      << "Failed with (v,v,v,v) at index: " << n << std::endl
      << "(" << var(params[0]) << ", " << var(params[1])
      << ", "<< var(params[2]) << ", " << var(params[3]) << ")" << std::endl;
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dddd) {
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
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1],
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dddv) {
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
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1],
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddvd) {
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
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1],
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddvv) {
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
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1],
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvdd) {
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
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvdv) {
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
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvvd) {
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
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvvv) {
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
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vddd) {
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
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vddv) {
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
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdvd) {
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
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdvv) {
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
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvdd) {
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
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  invalid_params[2],
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvdv) {
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
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  invalid_params[2],
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvvd) {
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
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  var(invalid_params[2]),
				  invalid_params[3]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvvv) {
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
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  var(invalid_params[2]),
				  var(invalid_params[3])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] 
      << "," << invalid_params[2] << "," << invalid_params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dddd) {
  var logprob_true = _LOG_PROB_<true>(this->first_valid_params()[0],
				      this->first_valid_params()[1],
				      this->first_valid_params()[2],
				      this->first_valid_params()[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1],
					 params[2],
					 params[3]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dddv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					params[1],
					params[2],
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      params[1],
				      params[2],
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   params[1],
					   params[2],
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1],
					 params[2],
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_ddvd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					params[1],
					var(params[2]),
					params[3]);
  var logprob_true = _LOG_PROB_<true>(params[0],
				      params[1],
				      var(params[2]),
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   params[1],
					   var(params[2]),
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1],
					 var(params[2]),
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_ddvv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					params[1],
					var(params[2]),
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      params[1],
				      var(params[2]),
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[2] = parameters[n][2];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   params[1],
					   var(params[2]),
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1],
					 var(params[2]),
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvdd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					params[2],
					params[3]);
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      params[2],
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]),
					   params[2],
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]),
					 params[2],
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvdv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					params[2],
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      params[2],
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]),
					   params[2],
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]),
					 params[2],
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvvd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					var(params[2]),
					params[3]);
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      var(params[2]),
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]),
					   var(params[2]),
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]),
					 var(params[2]),
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_dvvv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					var(params[2]),
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      var(params[2]),
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]),
					   var(params[2]),
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]),
					 var(params[2]),
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}




TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vddd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					params[2],
					params[3]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      params[2],
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1],
					   params[2],
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1],
					 params[2],
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vddv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					params[2],
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      params[2],
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1],
					   params[2],
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1],
					 params[2],
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vdvd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					var(params[2]),
					params[3]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      var(params[2]),
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1],
					   var(params[2]),
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1],
					 var(params[2]),
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vdvv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					var(params[2]),
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      var(params[2]),
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[2] = parameters[n][2];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1],
					   var(params[2]),
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1],
					 var(params[2]),
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvdd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					params[2],
					params[3]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      params[2],
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]),
					   params[2],
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]),
					 params[2],
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvdv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					params[2],
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      params[2],
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]),
					   params[2],
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]),
					 params[2],
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvvd) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					var(params[2]),
					params[3]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      var(params[2]),
				      params[3]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]),
					   var(params[2]),
					   params[3]);
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]),
					 var(params[2]),
					 params[3]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, logprob_propto_vvvv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					var(params[2]),
					var(params[3]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      var(params[2]),
				      var(params[3]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    params[3] = parameters[n][3];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]),
					   var(params[2]),
					   var(params[3]));
    
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]),
					 var(params[2]),
					 var(params[3]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] 
      << "," << this->first_valid_params()[2] << "," << this->first_valid_params()[3] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << "," << params[2] << "," << params[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dddd) {
  SUCCEED() << "No op for all double" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;
      
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p[0], p[1], p[2], p3);
    vector<var> v_params(1);
    v_params[0] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g3,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_ddvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
      
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(p[0], p[1], p2, p[3]);
    vector<var> v_params(1);
    v_params[0] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g2,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_ddvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;
      
    var p2(p[2]);
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p[0], p[1], p2, p3);
    vector<var> v_params(2);
    v_params[0] = p2;
    v_params[1] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g2,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
    EXPECT_NEAR(diff_g3,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvdd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
      
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(p[0], p1, p[2], p[3]);
    vector<var> v_params(1);
    v_params[0] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;

    var p1(p[1]);      
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p[0], p1, p[2], p3);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g3,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
     
    var p1(p[1]);
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(p[0], p1, p2, p[3]);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_dvvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;
      
    var p1(p[1]);
    var p2(p[2]);
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p[0], p1, p2, p3);
    vector<var> v_params(3);
    v_params[0] = p1;
    v_params[1] = p2;
    v_params[2] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
    EXPECT_NEAR(diff_g3,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vddd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
      
    var p0(p[0]);
    var lp = _LOG_PROB_<true>(p0, p[1], p[2], p[3]);
    vector<var> v_params(1);
    v_params[0] = p0;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;
      
    var p0(p[0]);
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p0, p[1], p[2], p3);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);


    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g3,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vdvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
      
    var p0(p[0]);
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(p0, p[1], p2, p[3]);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vdvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;

    var p0(p[0]);      
    var p2(p[2]);
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p0, p[1], p2, p3);
    vector<var> v_params(3);
    v_params[0] = p0;
    v_params[1] = p2;
    v_params[2] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);
    
    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
    EXPECT_NEAR(diff_g3,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvdd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
      
    var p0(p[0]);
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(p0, p1, p[2], p[3]);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;

    var p0(p[0]);
    var p1(p[1]);      
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p0, p1, p[2], p3);
    vector<var> v_params(3);
    v_params[0] = p0;
    v_params[1] = p1;
    v_params[2] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g3,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
     
    var p0(p[0]);
    var p1(p[1]);
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(p0, p1, p2, p[3]);
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
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_finite_diff_vvvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1], p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0]-e, p[1], p[2], p[3])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e, p[2], p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1]-e, p[2], p[3])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(p[0], p[1], p[2]+e, p[3]) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2]-e, p[3])) / e_times_2;
    double diff_g3 = (_LOG_PROB_<false>(p[0], p[1], p[2], p[3]+e) 
		      - _LOG_PROB_<false>(p[0], p[1], p[2], p[3]-e)) / e_times_2;
      
    var p0(p[0]);
    var p1(p[1]); 
    var p2(p[2]);
    var p3(p[3]);
    var lp = _LOG_PROB_<true>(p0, p1, p2, p3);
    vector<var> v_params(4);
    v_params[0] = p0;
    v_params[1] = p1;
    v_params[2] = p2;
    v_params[3] = p3;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
    EXPECT_NEAR(diff_g3,
		gradients[3],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dddd) {
  SUCCEED() << "No op for (d,d,d,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p[0], p[1], p[2], p3);
    var expected_lp = TypeParam().log_prob(p[0], p[1], p[2], p3);
    vector<var> v_params(1);
    v_params[0] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl 
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_ddvd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p[1], p2, p[3]);
    var expected_lp = TypeParam().log_prob(p[0], p[1], p2, p[3]);
    vector<var> v_params(1);
    v_params[0] = p2;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_ddvv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p2(p[2]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p[0], p[1], p2, p3);
    var expected_lp = TypeParam().log_prob(p[0], p[1], p2, p3);
    vector<var> v_params(2);
    v_params[0] = p2;
    v_params[1] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;

  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvdd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p[2], p[3]);
    var expected_lp = TypeParam().log_prob(p[0], p1, p[2], p[3]);
    vector<var> v_params(1);
    v_params[0] = p1;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvdv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p[2], p3);
    var expected_lp = TypeParam().log_prob(p[0], p1, p[2], p3);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;

  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvvd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p2, p[3]);
    var expected_lp = TypeParam().log_prob(p[0], p1, p2, p[3]);
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
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_dvvv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p2(p[2]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p2, p3);
    var expected_lp = TypeParam().log_prob(p[0], p1, p2, p3);
    vector<var> v_params(3);
    v_params[0] = p1;
    v_params[1] = p2;
    v_params[2] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vddd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p[2], p[3]);
    var expected_lp = TypeParam().log_prob(p0, p[1], p[2], p[3]);
    vector<var> v_params(1);
    v_params[0] = p0;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl 
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p[2], p3);
    var expected_lp = TypeParam().log_prob(p0, p[1], p[2], p3);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl 
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl 
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vdvd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p2, p[3]);
    var expected_lp = TypeParam().log_prob(p0, p[1], p2, p[3]);
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
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vdvv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p2(p[2]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p2, p3);
    var expected_lp = TypeParam().log_prob(p0, p[1], p2, p3);
    vector<var> v_params(3);
    v_params[0] = p0;
    v_params[1] = p2;
    v_params[2] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvdd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p[2], p[3]);
    var expected_lp = TypeParam().log_prob(p0, p1, p[2], p[3]);
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
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvdv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p[2], p3);
    var expected_lp = TypeParam().log_prob(p0, p1, p[2], p3);
    vector<var> v_params(3);
    v_params[0] = p0;
    v_params[1] = p1;
    v_params[2] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvvd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p2, p[3]);
    var expected_lp = TypeParam().log_prob(p0, p1, p2, p[3]);
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
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, gradient_function_vvvv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    var p2(p[2]);
    var p3(p[3]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p2, p3);
    var expected_lp = TypeParam().log_prob(p0, p1, p2, p3);
    vector<var> v_params(4);
    v_params[0] = p0;
    v_params[1] = p1;
    v_params[2] = p2;
    v_params[3] = p3;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[3],
		    gradients[3])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 3" << std::endl
      << "(" << p[0] << ", " << p[1] << ", " << p[2] << ", " << p[3] << ")" << std::endl;
  }
}

TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddd) {
  test_vectorized<double, double, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddD) {
  test_vectorized<double, double, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddv) {
  test_vectorized<double, double, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dddV) {
  test_vectorized<double, double, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDd) {
 test_vectorized<double, double, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDD) {
  test_vectorized<double, double, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDv) {
  test_vectorized<double, double, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddDV) {
  test_vectorized<double, double, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvd) {
  test_vectorized<double, double, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvD) {
  test_vectorized<double, double, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvv) {
  test_vectorized<double, double, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddvV) {
  test_vectorized<double, double, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVd) {
  test_vectorized<double, double, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVD) {
  test_vectorized<double, double, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVv) {
  test_vectorized<double, double, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddVV) {
  test_vectorized<double, double, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdd) {
  test_vectorized<double, vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdD) {
  test_vectorized<double, vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdv) {
  test_vectorized<double, vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDdV) {
  test_vectorized<double, vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDd) {
  test_vectorized<double, vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDD) {
  test_vectorized<double, vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDv) {
  test_vectorized<double, vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDDV) {
  test_vectorized<double, vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvd) {
  test_vectorized<double, vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvD) {
  test_vectorized<double, vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvv) {
  test_vectorized<double, vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDvV) {
  test_vectorized<double, vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVd) {
  test_vectorized<double, vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVD) {
  test_vectorized<double, vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVv) {
  test_vectorized<double, vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDVV) {
  test_vectorized<double, vector<double>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdd) {
  test_vectorized<double, vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdD) {
  test_vectorized<double, vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdv) {
  test_vectorized<double, vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVdV) {
  test_vectorized<double, vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDd) {
  test_vectorized<double, vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDD) {
  test_vectorized<double, vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDv) {
  test_vectorized<double, vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVDV) {
  test_vectorized<double, vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvd) {
  test_vectorized<double, vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvD) {
  test_vectorized<double, vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvv) {
  test_vectorized<double, vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVvV) {
  test_vectorized<double, vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVd) {
  test_vectorized<double, vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVD) {
  test_vectorized<double, vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVv) {
  test_vectorized<double, vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVVV) {
  test_vectorized<double, vector<double>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Dddd) {
  test_vectorized<vector<double>, double, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DddD) {
  test_vectorized<vector<double>, double, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Dddv) {
  test_vectorized<vector<double>, double, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DddV) {
  test_vectorized<vector<double>, double, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDd) {
  test_vectorized<vector<double>, double, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDD) {
  test_vectorized<vector<double>, double, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDv) {
  test_vectorized<vector<double>, double, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdDV) {
  test_vectorized<vector<double>, double, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Ddvd) {
  test_vectorized<vector<double>, double, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdvD) {
  test_vectorized<vector<double>, double, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_Ddvv) {
  test_vectorized<vector<double>, double, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdvV) {
  test_vectorized<vector<double>, double, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVd) {
  test_vectorized<vector<double>, double, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVD) {
  test_vectorized<vector<double>, double, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVv) {
  test_vectorized<vector<double>, double, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DdVV) {
  test_vectorized<vector<double>, double, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdd) {
  test_vectorized<vector<double>, vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdD) {
  test_vectorized<vector<double>, vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdv) {
  test_vectorized<vector<double>, vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDdV) {
  test_vectorized<vector<double>, vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDd) {
  test_vectorized<vector<double>, vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDD) {
  test_vectorized<vector<double>, vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDv) {
  test_vectorized<vector<double>, vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDDV) {
  test_vectorized<vector<double>, vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvd) {
  test_vectorized<vector<double>, vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvD) {
  test_vectorized<vector<double>, vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvv) {
  test_vectorized<vector<double>, vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDvV) {
  test_vectorized<vector<double>, vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVd) {
  test_vectorized<vector<double>, vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVD) {
  test_vectorized<vector<double>, vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVv) {
  test_vectorized<vector<double>, vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DDVV) {
  test_vectorized<vector<double>, vector<double>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdd) {
  test_vectorized<vector<double>, vector<var>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdD) {
  test_vectorized<vector<double>, vector<var>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdv) {
  test_vectorized<vector<double>, vector<var>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVdV) {
  test_vectorized<vector<double>, vector<var>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDd) {
  test_vectorized<vector<double>, vector<var>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDD) {
  test_vectorized<vector<double>, vector<var>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDv) {
  test_vectorized<vector<double>, vector<var>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVDV) {
  test_vectorized<vector<double>, vector<var>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvd) {
  test_vectorized<vector<double>, vector<var>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvD) {
  test_vectorized<vector<double>, vector<var>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvv) {
  test_vectorized<vector<double>, vector<var>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVvV) {
  test_vectorized<vector<double>, vector<var>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVd) {
  test_vectorized<vector<double>, vector<var>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVD) {
  test_vectorized<vector<double>, vector<var>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVv) {
  test_vectorized<vector<double>, vector<var>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture4, vectorized_DVVV) {
  test_vectorized<vector<double>, vector<var>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddd) {
  test_vectorized<var, double, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddD) {
  test_vectorized<var, double, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddv) {
  test_vectorized<var, double, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vddV) {
  test_vectorized<var, double, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDd) {
  test_vectorized<var, double, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDD) {
  test_vectorized<var, double, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDv) {
  test_vectorized<var, double, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdDV) {
  test_vectorized<var, double, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvd) {
  test_vectorized<var, double, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvD) {
  test_vectorized<var, double, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvv) {
  test_vectorized<var, double, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdvV) {
  test_vectorized<var, double, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVd) {
  test_vectorized<var, double, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVD) {
  test_vectorized<var, double, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVv) {
  test_vectorized<var, double, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vdVV) {
  test_vectorized<var, double, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdd) {
  test_vectorized<var, vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdD) {
  test_vectorized<var, vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdv) {
  test_vectorized<var, vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDdV) {
  test_vectorized<var, vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDd) {
  test_vectorized<var, vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDD) {
  test_vectorized<var, vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDv) {
  test_vectorized<var, vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDDV) {
  test_vectorized<var, vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvd) {
  test_vectorized<var, vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvD) {
  test_vectorized<var, vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvv) {
  test_vectorized<var, vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDvV) {
  test_vectorized<var, vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVd) {
  test_vectorized<var, vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVD) {
  test_vectorized<var, vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVv) {
  test_vectorized<var, vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vDVV) {
  test_vectorized<var, vector<double>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdd) {
  test_vectorized<var, vector<var>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdD) {
  test_vectorized<var, vector<var>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdv) {
  test_vectorized<var, vector<var>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVdV) {
  test_vectorized<var, vector<var>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDd) {
  test_vectorized<var, vector<var>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDD) {
  test_vectorized<var, vector<var>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDv) {
  test_vectorized<var, vector<var>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVDV) {
  test_vectorized<var, vector<var>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvd) {
  test_vectorized<var, vector<var>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvD) {
  test_vectorized<var, vector<var>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvv) {
  test_vectorized<var, vector<var>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVvV) {
  test_vectorized<var, vector<var>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVd) {
  test_vectorized<var, vector<var>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVD) {
  test_vectorized<var, vector<var>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVv) {
  test_vectorized<var, vector<var>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture5, vectorized_vVVV) {
  test_vectorized<var, vector<var>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vddd) {
  test_vectorized<vector<var>, double, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VddD) {
  test_vectorized<vector<var>, double, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vddv) {
  test_vectorized<vector<var>, double, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VddV) {
  test_vectorized<vector<var>, double, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDd) {
  test_vectorized<vector<var>, double, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDD) {
  test_vectorized<vector<var>, double, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDv) {
  test_vectorized<vector<var>, double, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdDV) {
  test_vectorized<vector<var>, double, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vdvd) {
  test_vectorized<vector<var>, double, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdvD) {
  test_vectorized<vector<var>, double, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_Vdvv) {
  test_vectorized<vector<var>, double, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdvV) {
  test_vectorized<vector<var>, double, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVd) {
  test_vectorized<vector<var>, double, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVD) {
  test_vectorized<vector<var>, double, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVv) {
  test_vectorized<vector<var>, double, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VdVV) {
  test_vectorized<vector<var>, double, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdd) {
  test_vectorized<vector<var>, vector<double>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdD) {
  test_vectorized<vector<var>, vector<double>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdv) {
  test_vectorized<vector<var>, vector<double>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDdV) {
  test_vectorized<vector<var>, vector<double>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDd) {
  test_vectorized<vector<var>, vector<double>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDD) {
  test_vectorized<vector<var>, vector<double>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDv) {
  test_vectorized<vector<var>, vector<double>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDDV) {
  test_vectorized<vector<var>, vector<double>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvd) {
  test_vectorized<vector<var>, vector<double>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvD) {
  test_vectorized<vector<var>, vector<double>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvv) {
  test_vectorized<vector<var>, vector<double>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDvV) {
  test_vectorized<vector<var>, vector<double>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVd) {
  test_vectorized<vector<var>, vector<double>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVD) {
  test_vectorized<vector<var>, vector<double>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVv) {
  test_vectorized<vector<var>, vector<double>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VDVV) {
  test_vectorized<vector<var>, vector<double>, vector<var>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdd) {
  test_vectorized<vector<var>, vector<var>, double, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdD) {
  test_vectorized<vector<var>, vector<var>, double, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdv) {
  test_vectorized<vector<var>, vector<var>, double, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVdV) {
  test_vectorized<vector<var>, vector<var>, double, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDd) {
  test_vectorized<vector<var>, vector<var>, vector<double>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDD) {
  test_vectorized<vector<var>, vector<var>, vector<double>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDv) {
  test_vectorized<vector<var>, vector<var>, vector<double>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVDV) {
  test_vectorized<vector<var>, vector<var>, vector<double>, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvd) {
  test_vectorized<vector<var>, vector<var>, var, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvD) {
  test_vectorized<vector<var>, vector<var>, var, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvv) {
  test_vectorized<vector<var>, vector<var>, var, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVvV) {
  test_vectorized<vector<var>, vector<var>, var, vector<var>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVd) {
  test_vectorized<vector<var>, vector<var>, vector<var>, double, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVD) {
  test_vectorized<vector<var>, vector<var>, vector<var>, vector<double>, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVv) {
  test_vectorized<vector<var>, vector<var>, vector<var>, var, TypeParam>();
}
TYPED_TEST_P(AgradDistributionTestFixture6, vectorized_VVVV) {
  test_vectorized<vector<var>, vector<var>, vector<var>, vector<var>, TypeParam>();
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
