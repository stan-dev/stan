#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>

using stan::agrad::var;
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
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
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
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
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
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
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
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[2],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl;
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
      << "Index: " << n << " - function value test failed" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[2],
		    gradients[2])
      << "Index: " << n << " - hand-coded test failed for parameter 2" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p1, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1.push_back(grad[1]);
    expected_grad_p2.push_back(grad[2]);
  }

  vector<var> p0, p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  logprob.grad(p0, grad_p0);
  logprob.grad(p1, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1.push_back(grad[1]);
  }

  vector<var> p0, p1;
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  logprob.grad(p0, grad_p0);
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p1;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1.push_back(grad[1]);
    expected_grad_p2 += grad[2];
  }

  vector<var> p0, p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(p0, grad_p0);
  logprob.grad(p1, grad_p1);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VVd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1.push_back(grad[1]);
  }

  vector<var> p0, p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  logprob.grad(p0, grad_p0);
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p2.push_back(grad[1]);
  }

  vector<var> p0, p2;
  vector<double> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  logprob.grad(p0, grad_p0);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
  }

  vector<var> p0;
  vector<double> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  logprob.grad(p0, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p2 += grad[1];
  }

  vector<var> p0;
  vector<double> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(p0, grad_p0);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
  << "Gradient failed for parameter 2";
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VDd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
  }

  vector<var> p0;
  vector<double> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  logprob.grad(p0, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VvV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p2;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1 += grad[1];
    expected_grad_p2.push_back(grad[2]);
  }

  vector<var> p0, p2;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(p0, grad_p0);
  logprob.grad(x, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VvD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1 += grad[1];
  }

  vector<var> p0;
  var p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(p0, grad_p0);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  double expected_grad_p1 = 0.0, expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1 += grad[1];
    expected_grad_p2 += grad[2];
  }

  vector<var> p0;
  var p1 = parameters[0][1];
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  logprob.grad(p0, grad_p0);
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);
  x.clear();
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[0][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1 += grad[1];
  }

  vector<var> p0;
  var p1 = parameters[0][1];
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(p0, grad_p0);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VdV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p2.push_back(grad[1]);
  }

  vector<var> p0, p2;
  double p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  logprob.grad(p0, grad_p0);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_VdD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
  }

  vector<var> p0;
  double p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  logprob.grad(p0, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vdv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[0][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p2 += grad[1];
  }

  vector<var> p0;
  double p1 = parameters[0][1];
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(p0, grad_p0);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
  << "Gradient failed for parameter 2";
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Vdd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[0][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
  }

  vector<var> p0;
  double p1 = parameters[0][1];
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  logprob.grad(p0, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
}

TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2.push_back(grad[1]);
  }

  vector<double> p0;
  vector<var> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  logprob.grad(p1, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<double> p0, p2;
  vector<var> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2 += grad[1];
  }

  vector<double> p0;
  vector<var> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(p1, grad_p1);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DVd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<double> p0;
  vector<var> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    double p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<double> p0, p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> p0, p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    double p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  vector<double> p0, p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2";
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DDd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  
  vector<double> p0, p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DvV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
    expected_grad_p2.push_back(grad[1]);
  }

  vector<double> p0;
  vector<var> p2;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DvD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  vector<double> p0, p2;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Dvv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0, expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
    expected_grad_p2 += grad[1];
  }

  vector<double> p0;
  var p1 = parameters[0][1];
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);
  x.clear();
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Dvd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[0][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  vector<double> p0;
  var p1 = parameters[0][1];
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DdV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    double p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<double> p0;
  double p1 = parameters[0][1];
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
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_DdD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;

  vector<double> p0;
  double p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ddv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    double p1 = parameters[0][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  vector<double> p0;
  double p1 = parameters[0][1];
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
  << "Gradient failed for parameter 2";
}
TYPED_TEST_P(AgradDistributionTestFixture2, vectorized_Ddd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  
  vector<double> p0;
  double p1 = parameters[0][1];
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vVV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p1, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;

    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1.push_back(grad[1]);
    expected_grad_p2.push_back(grad[2]);
  }
  
  var p0 = parameters[0][0];
  vector<var> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p1, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vVD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1.push_back(grad[1]);
  }
  var p0 = parameters[0][0];
  vector<var> p1;
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vVv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p1;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1.push_back(grad[1]);
    expected_grad_p2 += grad[2];
  }

  var p0 = parameters[0][0];
  vector<var> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p1, grad_p1);
  x.clear();
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vVd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1.push_back(grad[1]);
  }

  var p0 = parameters[0][0];
  vector<var> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vDV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p2.push_back(grad[1]);
  }

  var p0 = parameters[0][0];
  vector<double> p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vDD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
  }

  var p0 = parameters[0][0];
  vector<double> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vDv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p2 += grad[1];
  }

  var p0 = parameters[0][0];
  vector<double> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  x.clear();
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;  
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
  << "Gradient failed for parameter 2";
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vDd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
  }

  var p0 = parameters[0][0];
  vector<double> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vvV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p2;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1 += grad[1];
    expected_grad_p2.push_back(grad[2]);
  }

  var p0 = parameters[0][0];
  vector<var> p2;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  x.clear();
  x.push_back(p1);
  logprob.grad(x, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vvD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1 += grad[1];
  }

  var p0 = parameters[0][0];
  var p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  x.clear();
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vdV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p2.push_back(grad[1]);
  }

  var p0 = parameters[0][0];
  vector<var> p2;
  double p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0, grad_p2;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_vdD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
  }

  var p0 = parameters[0][0];
  double p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p0;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2.push_back(grad[1]);
  }

  double p0 = parameters[0][0];
  vector<var> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  logprob.grad(p1, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  double p0 = parameters[0][0];
  vector<var> p1; 
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2 += grad[1];
  }

  double p0 = parameters[0][0];
  vector<var> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(p1, grad_p1);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dVd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  double p0 = parameters[0][0];
  vector<var> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    double p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  double p0 = parameters[0][0];
  vector<double> p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double p0 = parameters[0][0];
  vector<double> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    double p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  double p0 = parameters[0][0];
  vector<double> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2";
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dDd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  
  double p0 = parameters[0][0];
  vector<double> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dvV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
    expected_grad_p2.push_back(grad[1]);
  }

  double p0 = parameters[0][0];
  vector<var> p2;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_dvD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  double p0 = parameters[0][0];
  vector<double> p2;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    double p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  double p0 = parameters[0][0];
  double p1 = parameters[0][1];
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
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture3, vectorized_ddD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;

  double p0 = parameters[0][0];
  double p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
}

//------------------------------------------------------------

// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
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
			   vectorized_VVV,
			   vectorized_VVD,
			   vectorized_VVv,
			   vectorized_VVd,
			   vectorized_VDV,
			   vectorized_VDD,
			   vectorized_VDv,
			   vectorized_VDd,
			   vectorized_VvV,
			   vectorized_VvD,
			   vectorized_Vvv,
			   vectorized_Vvd,
			   vectorized_VdV,
			   vectorized_VdD,
			   vectorized_Vdv,
			   vectorized_Vdd,
			   vectorized_DVV,
			   vectorized_DVD,
			   vectorized_DVv,
			   vectorized_DVd,
			   vectorized_DDV,
			   vectorized_DDD,
			   vectorized_DDv,
			   vectorized_DDd,
			   vectorized_DvV,
			   vectorized_DvD,
			   vectorized_Dvv,
			   vectorized_Dvd,
			   vectorized_DdV,
			   vectorized_DdD,
			   vectorized_Ddv,
			   vectorized_Ddd);
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture3,
			   vectorized_vVV,
			   vectorized_vVD,
			   vectorized_vVv,
			   vectorized_vVd,
			   vectorized_vDV,
			   vectorized_vDD,
			   vectorized_vDv,
			   vectorized_vDd,
			   vectorized_vvV,
			   vectorized_vvD,
			   vectorized_vdV,
			   vectorized_vdD,
			   vectorized_dVV,
			   vectorized_dVD,
			   vectorized_dVv,
			   vectorized_dVd,
			   vectorized_dDV,
			   vectorized_dDD,
			   vectorized_dDv,
			   vectorized_dDd,
			   vectorized_dvV,
			   vectorized_dvD,
			   vectorized_ddV,
			   vectorized_ddD);
#endif
