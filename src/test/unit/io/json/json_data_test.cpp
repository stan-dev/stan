#include <gtest/gtest.h>

#include <stan/io/json.hpp>


void test_int_var(stan::json::json_data& jdata,
                  const std::string& text,
                  const std::string& name,
                  const std::vector<int>& expected_vals,
                  const std::vector<long unsigned int>& expected_dims) {
  std::cout << "json: " << text << std::endl;
  std::cout << "text int var: " << name << std::endl;
  EXPECT_EQ(true,jdata.contains_i(name));
  std::vector<long unsigned int> dims = jdata.dims_i(name);
  EXPECT_EQ(expected_dims.size(),dims.size());
  for (int i = 0; i<dims.size(); i++) 
    EXPECT_EQ(expected_dims[i],dims[i]);
  std::vector<int> vals = jdata.vals_i(name);
  EXPECT_EQ(expected_vals.size(),vals.size());
  for (int i = 0; i<vals.size(); i++) 
    EXPECT_EQ(expected_vals[i],vals[i]);
}

void test_real_var(stan::json::json_data& jdata,
                  const std::string& text,
                  const std::string& name,
                  const std::vector<double>& expected_vals,
                  const std::vector<long unsigned int>& expected_dims) {
  std::cout << "json: " << text << std::endl;
  std::cout << "text int var: " << name << std::endl;
  EXPECT_EQ(true,jdata.contains_r(name));
  std::vector<long unsigned int> dims = jdata.dims_r(name);
  EXPECT_EQ(expected_dims.size(),dims.size());
  for (int i = 0; i<dims.size(); i++) 
    EXPECT_EQ(expected_dims[i],dims[i]);
  std::vector<double> vals = jdata.vals_r(name);
  EXPECT_EQ(expected_vals.size(),vals.size());
  for (int i = 0; i<vals.size(); i++) 
    EXPECT_EQ(expected_vals[i],vals[i]);
}


TEST(ioJson,jsonData_scalar_int) {
  std::string txt = "{ \"foo\" : 1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<int> expected_vals;
  expected_vals.push_back(1);
  std::vector<long unsigned int> expected_dims;
  test_int_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_scalar_real) {
  std::string txt = "{ \"foo\" : 1.1 }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  std::vector<long unsigned int> expected_dims;
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}


TEST(ioJson,jsonData_real_array_1D) {
  std::string txt = "{ \"foo\" : [ 1.1, 2.2 ] }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(2.2);
  std::vector<long unsigned int> expected_dims;
  expected_dims.push_back(2);
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}

TEST(ioJson,jsonData_real_array_2D) {
  std::string txt = "{ \"foo\" : [ [ 1.1, 1.2 ], [ 2.1, 2.2 ], [ 3.1, 3.2] ]  }";
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  std::vector<double> expected_vals;
  expected_vals.push_back(1.1);
  expected_vals.push_back(1.2);
  expected_vals.push_back(2.1);
  expected_vals.push_back(2.2);
  expected_vals.push_back(3.1);
  expected_vals.push_back(3.2);
  std::vector<long unsigned int> expected_dims;
  expected_dims.push_back(3);
  expected_dims.push_back(2);
  test_real_var(jdata,txt,"foo",expected_vals,expected_dims);
}


  /*
  std::cout << "real-valued vars" << std::endl;
  std::vector<std::string> names_real_vars;
  jdata.names_r(names_real_vars);
  for (std::vector<std::string>::const_iterator it = names_real_vars.begin();
       it != names_real_vars.end(); ++it) 
    std::cout << (*it);

  std::vector<std::string> names_int_vars;
  std::cout << "int-valued vars" << std::endl;
  jdata.names_i(names_int_vars);
  for (std::vector<std::string>::const_iterator it = names_int_vars.begin();
       it != names_int_vars.end(); ++it) 
    std::cout << (*it);
  */
