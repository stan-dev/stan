#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___

using stan::agrad::var;

// v: var
// d: double
// V: vector<var>
// D: vector<double>
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddd) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  params[2]))
      << "Failed with (d,d,d) at index: " << n << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (d,d,d) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ddv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1],
					  var(params[2])))
      << "Failed with (d,d,v) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvd) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  params[2]))
      << "Failed with (d,v,d) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dvv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1]),
					  var(params[2])))
      << "Failed with (d,v,v) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdd) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  params[2]))
      << "Failed with (v,d,d) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vdv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1],
					  var(params[2])))
      << "Failed with (v,d,v) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvd) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  params[2]))
      << "Failed with (d,v,d) at index: " << n << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vvv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1]),
					  var(params[2])))
      << "Failed with (v,v,v) at index: " << n << std::endl;
  }
}


TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddd) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
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
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ddv) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  invalid_params[1],
				  var(invalid_params[2])),
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
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvd) {  
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  var(invalid_params[1]),
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
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dvv) {  
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdd) {  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
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
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vdv) {  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvd) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vvv) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << "," << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ddd) {
  TypeParam t;
  var logprob_true = _LOG_PROB_<true>(this->first_valid_params()[0],
				      this->first_valid_params()[1],
				      this->first_valid_params()[2]);
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					params[1],
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      params[1],
				      var(params[2]));
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					params[2]);
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      params[2]);
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]),
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]),
				      var(params[2]));
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					params[2]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      params[2]);
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1],
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1],
				      var(params[2]));
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					params[2]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      params[2]);
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]),
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]),
				      var(params[2]));
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g2,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dvd) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g1,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dvv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g1,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
    EXPECT_FLOAT_EQ(diff_g2,
		    gradients[1])
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vdd) {  
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g0,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vdv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g0,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
    EXPECT_FLOAT_EQ(diff_g2,
		    gradients[1])
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vvd) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g0,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
    EXPECT_FLOAT_EQ(diff_g1,
		    gradients[1])
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vvv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

    EXPECT_FLOAT_EQ(diff_g0,
		    gradients[0])
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl;
    EXPECT_FLOAT_EQ(diff_g1,
		    gradients[1])
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl;
    EXPECT_FLOAT_EQ(diff_g2,
		    gradients[2])
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ddd) {
  SUCCEED() << "No op for (d,d,d) input" << std::endl;
}

TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ddv) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p[1], p2);
    var expected_lp = t.log_prob(p[0], p[1], p2);
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
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p[2]);
    var expected_lp = t.log_prob(p[0], p1, p[2]);
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
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p[0], p1, p2);
    var expected_lp = t.log_prob(p[0], p1, p2);
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
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p[2]);
    var expected_lp = t.log_prob(p0, p[1], p[2]);
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
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vdv) {  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p[1], p2);
    var expected_lp = t.log_prob(p0, p[1], p2);
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
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vvd) {  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p[2]);
    var expected_lp = t.log_prob(p0, p1, p[2]);
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
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(p0, p1, p2);
    var expected_lp = t.log_prob(p0, p1, p2);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DDD) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DDV) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DVD) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DVV) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VDD) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VDV) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VVD) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VVV) {
  TypeParam t;
  vector<vector<double> > parameters;
  t.valid_values(parameters);
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
			   gradient_function_vvv,
			   vectorized_DDD,
			   vectorized_DDV,
			   vectorized_DVD,
			   vectorized_DVV,
			   vectorized_VDD,
			   vectorized_VDV,
			   vectorized_VVD,
			   vectorized_VVV
			   );
#endif
