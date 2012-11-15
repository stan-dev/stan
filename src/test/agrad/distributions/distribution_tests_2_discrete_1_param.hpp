#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_1_PARAM_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_1_PARAM_HPP___

// i: int
// v: var
// d: double
// I: vector<int>
// V: vector<var>
// D: vector<double>


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
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]),
					  int(params[1]),
					  params[2]))
      << "Failed with (i,i,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << int(params[1]) << ", " << params[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (i,i,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << int(params[1]) << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iiv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]),
					  int(params[1]),
					  var(params[2])))
      << "Failed with (i,i,v) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << int(params[1]) << ", " << var(params[2]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iid) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  int(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << "," << int(invalid_params[1]) << ", " << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 1; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]), 
				  int(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << "," << int(invalid_params[1]) << ", " << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iiv) {
  vector<size_t> index;
  vector<double> invalid_values;
  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  int(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << "," << int(invalid_params[1]) << ", " << var(invalid_params[2]) << ")" << std::endl;
  }
  for (size_t i = 1; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),	
				  int(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i << std::endl
      << "(" << int(invalid_params[0]) << "," << int(invalid_params[1]) << ", " << var(invalid_params[2]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iid) {
  var logprob_true = _LOG_PROB_<true>(int(this->first_valid_params()[0]),
				      int(this->first_valid_params()[1]),
				      this->first_valid_params()[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(int(params[0]),
					 int(params[1]),
					 params[2]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << int(this->first_valid_params()[1]) << ", " << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << int(params[1]) << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iiv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(int(params[0]),
					int(params[1]),
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(int(params[0]),
				      int(params[1]),
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    var logprob2_false = _LOG_PROB_<false>(int(params[0]),
					   int(params[1]),
					   var(params[2]));
    
    var logprob2_true = _LOG_PROB_<true>(int(params[0]),
					 int(params[1]),
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << int(this->first_valid_params()[1]) << ", " << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << int(params[1]) << ", " << params[2] << ")" << std::endl;
  }
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IIV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[n][0];
    int p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<int> p0;
  vector<int> p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
    p1.push_back(int(parameters[n][1]));
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IID) {
  SUCCEED() << "No op for (I,D) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IIv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    int p1 = int(parameters[n][1]);
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  vector<int> p0;
  vector<int> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
    p1.push_back(int(parameters[n][1]));
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IId) {
  SUCCEED() << "No op for (I,I,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IiV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  int p1 = parameters[0][1];
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[n][0];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<int> p0;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IiD) {
  SUCCEED() << "No op for (I,i,D) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iiv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  int p1 = parameters[0][1];
  var p2 = parameters[0][2];
  double expected_grad_p2(0);
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[n][0];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  vector<int> p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<var> x;
  vector<double> grad_p2;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iid) {
  SUCCEED() << "No op for (I,i,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iIV) {
   vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  int p0 = parameters[0][0];
  for (size_t n = 0; n < parameters.size(); n++) {
    int p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<int> p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(int(parameters[n][1]));
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iID) {
  SUCCEED() << "No op for (i,I,D) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iIv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  int p0 = parameters[0][0];
  var p2 = parameters[0][2];
  double expected_grad_p2(0);
  for (size_t n = 0; n < parameters.size(); n++) {
    int p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  vector<int> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<var> x;
  vector<double> grad_p2;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iId) {
  SUCCEED() << "No op for (i,I,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  int p0 = parameters[0][0];
  int p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiD) {
  SUCCEED() << "No op for (i,i,D) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iiv) {
  SUCCEED() << "No op for (i,i,v) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iid) {
  SUCCEED() << "No op for (i,i,d) input" << std::endl;
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
			   vectorized_IIV,
			   vectorized_IID,
			   vectorized_IIv,
			   vectorized_IId,
			   vectorized_IiV,
			   vectorized_IiD,
			   vectorized_Iiv,
			   vectorized_Iid,
			   vectorized_iIV,
			   vectorized_iID,
			   vectorized_iIv,
			   vectorized_iId,
			   vectorized_iiV,
			   vectorized_iiD,
			   vectorized_iiv,
			   vectorized_iid);
#endif
