
#include <limits>
#include <stan/io/array_var_context.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>
#include <test/test-models/good/services/bernoulli.hpp>
#include <Eigen/Dense>

TEST(array_var_context, ctor_int) {
  std::vector<int> v;
  for (size_t i = 0; i < 16; i++) {
    v.push_back(i);
  }
  std::vector<std::vector<size_t>> dims;
  std::vector<size_t> scalar_dim;
  std::vector<size_t> vec_dim;
  std::vector<size_t> zerolen_dim;
  zerolen_dim.push_back(3);
  zerolen_dim.push_back(0);
  vec_dim.push_back(3);
  std::vector<size_t> array_dim;
  array_dim.push_back(3);
  array_dim.push_back(4);
  dims.push_back(scalar_dim);
  dims.push_back(vec_dim);
  dims.push_back(array_dim);
  dims.push_back(zerolen_dim);
  std::vector<std::string> names;
  names.push_back("alpha");
  names.push_back("beta");
  names.push_back("gamma");
  names.push_back("eta");
  stan::io::array_var_context avc(names, v, dims);

  EXPECT_TRUE(avc.contains_i("alpha"));
  EXPECT_TRUE(avc.contains_i("beta"));
  EXPECT_TRUE(avc.contains_i("gamma"));
  EXPECT_TRUE(avc.contains_i("eta"));

  EXPECT_TRUE(avc.contains_r("alpha"));
  EXPECT_TRUE(avc.contains_r("beta"));
  EXPECT_TRUE(avc.contains_r("gamma"));
  EXPECT_TRUE(avc.contains_r("eta"));

  std::vector<int> alpha;
  alpha.push_back(0L);
  EXPECT_EQ(alpha, avc.vals_i("alpha"));
  EXPECT_EQ(scalar_dim, avc.dims_i("alpha"));

  std::vector<int> beta;
  for (size_t i = 1; i <= 3; i++) {
    beta.push_back(i);
  }
  EXPECT_EQ(beta, avc.vals_i("beta"));
  EXPECT_EQ(vec_dim, avc.dims_i("beta"));

  std::vector<int> gamma;
  for (size_t i = 4; i < 16; i++) {
    gamma.push_back(i);
  }
  EXPECT_EQ(gamma, avc.vals_i("gamma"));
  EXPECT_EQ(array_dim, avc.dims_i("gamma"));
}

TEST(array_var_context, ctor_real) {
  std::vector<double> v;
  for (size_t i = 0; i < 16; i++) {
    v.push_back(1.0 * i);
  }
  std::vector<std::vector<size_t>> dims;
  std::vector<size_t> scalar_dim;
  std::vector<size_t> vec_dim;
  std::vector<size_t> zerolen_dim;
  zerolen_dim.push_back(3);
  zerolen_dim.push_back(0);
  vec_dim.push_back(3);
  std::vector<size_t> array_dim;
  array_dim.push_back(3);
  array_dim.push_back(4);
  dims.push_back(scalar_dim);
  dims.push_back(vec_dim);
  dims.push_back(array_dim);
  dims.push_back(zerolen_dim);
  std::vector<std::string> names;
  names.push_back("alpha");
  names.push_back("beta");
  names.push_back("gamma");
  names.push_back("eta");
  stan::io::array_var_context avc(names, v, dims);

  EXPECT_TRUE(avc.contains_r("alpha"));
  EXPECT_TRUE(avc.contains_r("beta"));
  EXPECT_TRUE(avc.contains_r("gamma"));
  EXPECT_TRUE(avc.contains_r("eta"));

  EXPECT_FALSE(avc.contains_i("alpha"));
  EXPECT_FALSE(avc.contains_i("beta"));
  EXPECT_FALSE(avc.contains_i("gamma"));
  EXPECT_FALSE(avc.contains_i("eta"));

  std::vector<double> alpha;
  alpha.push_back(0);
  EXPECT_EQ(alpha, avc.vals_r("alpha"));
  EXPECT_EQ(scalar_dim, avc.dims_r("alpha"));

  std::vector<double> beta;
  for (size_t i = 1; i <= 3; i++) {
    beta.push_back(i);
  }
  EXPECT_EQ(beta, avc.vals_r("beta"));
  EXPECT_EQ(vec_dim, avc.dims_r("beta"));

  std::vector<double> gamma;
  for (size_t i = 4; i < 16; i++) {
    gamma.push_back(i);
  }
  EXPECT_EQ(gamma, avc.vals_r("gamma"));
  EXPECT_EQ(array_dim, avc.dims_r("gamma"));

  std::vector<double> eta;
  EXPECT_EQ(eta, avc.vals_r("eta"));
}

TEST(array_var_context, ctor_RowVectorXd) {
  Eigen::RowVectorXd v;
  v.resize(16);
  for (size_t i = 0; i < 16; i++) {
    v(i) = 1.0 * i;
  }
  std::vector<std::vector<size_t>> dims;
  std::vector<size_t> scalar_dim;
  std::vector<size_t> vec_dim;
  vec_dim.push_back(3);
  std::vector<size_t> array_dim;
  array_dim.push_back(3);
  array_dim.push_back(4);
  dims.push_back(scalar_dim);
  dims.push_back(vec_dim);
  dims.push_back(array_dim);
  std::vector<std::string> names;
  names.push_back("alpha");
  names.push_back("beta");
  names.push_back("gamma");
  stan::io::array_var_context avc(names, v, dims);

  EXPECT_TRUE(avc.contains_r("alpha"));
  EXPECT_TRUE(avc.contains_r("beta"));
  EXPECT_TRUE(avc.contains_r("gamma"));

  EXPECT_FALSE(avc.contains_i("alpha"));
  EXPECT_FALSE(avc.contains_i("beta"));
  EXPECT_FALSE(avc.contains_i("gamma"));

  std::vector<double> alpha;
  alpha.push_back(0);
  EXPECT_EQ(alpha, avc.vals_r("alpha"));
  EXPECT_EQ(scalar_dim, avc.dims_r("alpha"));

  std::vector<double> beta;
  for (size_t i = 1; i <= 3; i++) {
    beta.push_back(i);
  }
  EXPECT_EQ(beta, avc.vals_r("beta"));
  EXPECT_EQ(vec_dim, avc.dims_r("beta"));

  std::vector<double> gamma;
  for (size_t i = 4; i < 16; i++) {
    gamma.push_back(i);
  }
  EXPECT_EQ(gamma, avc.vals_r("gamma"));
  EXPECT_EQ(array_dim, avc.dims_r("gamma"));
}

TEST(array_var_context, invalid_input) {
  try {
    std::vector<double> a;
    a.push_back(3.0);
    std::vector<size_t> array_dim;
    array_dim.push_back(3);
    array_dim.push_back(4);
    std::vector<std::vector<size_t>> dims;
    dims.push_back(array_dim);
    std::vector<std::string> names;
    names.push_back("alpha");
    stan::io::array_var_context avc(names, a, dims);
  } catch (const std::exception& e) {
    return;
  }
  FAIL();
}

TEST(array_var_context, invalid_input2) {
  try {
    std::vector<double> a;
    a.push_back(3.0);
    std::vector<size_t> array_dim;
    array_dim.push_back(3);
    array_dim.push_back(4);
    std::vector<std::vector<size_t>> dims;
    dims.push_back(array_dim);
    std::vector<std::string> names;
    names.push_back("alpha");
    names.push_back("alpha2");
    stan::io::array_var_context avc(names, a, dims);
  } catch (const std::exception& e) {
    return;
  }
  FAIL();
}

TEST(array_var_context, invalid_context_validate) {
  std::vector<int> a;
  std::vector<std::vector<size_t>> dims;
  std::vector<std::string> names;
  stan::io::array_var_context avc(names, a, dims);
  // invalid - empty
  EXPECT_THROW(bernoulli_model_namespace::bernoulli_model(avc, 0, &std::cout),
               std::runtime_error);
  // invalid - missing N and y
  a.push_back(0);
  std::vector<size_t> scalar_dim;
  dims.push_back(scalar_dim);
  names.push_back("K");
  stan::io::array_var_context avc1(names, a, dims);
  EXPECT_THROW(bernoulli_model_namespace::bernoulli_model(avc1, 0, &std::cout),
               std::runtime_error);
  // invalid - missing y
  a.push_back(1);
  dims.push_back(scalar_dim);
  names.push_back("N");
  stan::io::array_var_context avc2(names, a, dims);
  EXPECT_THROW(bernoulli_model_namespace::bernoulli_model(avc2, 0, &std::cout),
               std::runtime_error);
  // OK
  a.push_back(1);
  std::vector<size_t> arr_dim;
  arr_dim.push_back(1);
  dims.push_back(arr_dim);
  names.push_back("y");
  stan::io::array_var_context avc3(names, a, dims);
  EXPECT_NO_THROW(
      bernoulli_model_namespace::bernoulli_model(avc3, 0, &std::cout));
}

TEST(array_var_context, ctor_complex) {
  std::vector<double> v;
  for (size_t i = 0; i < 32; i++) {
    v.push_back(1.0 * i);
  }
  std::vector<std::vector<size_t>> dims;
  std::vector<size_t> alpha_scalar_dim{2};

  std::vector<size_t> beta_vec_dim;
  beta_vec_dim.push_back(3);
  beta_vec_dim.push_back(2);

  std::vector<size_t> zerolen_dim;
  zerolen_dim.push_back(3);
  zerolen_dim.push_back(0);

  std::vector<size_t> array_dim;
  array_dim.push_back(3);
  array_dim.push_back(4);
  array_dim.push_back(2);

  dims.push_back(alpha_scalar_dim);
  dims.push_back(beta_vec_dim);
  dims.push_back(array_dim);
  dims.push_back(zerolen_dim);
  std::vector<std::string> names;
  names.push_back("alpha");
  names.push_back("beta");
  names.push_back("gamma");
  names.push_back("eta");
  stan::io::array_var_context avc(names, v, dims);

  EXPECT_TRUE(avc.contains_r("alpha"));
  EXPECT_TRUE(avc.contains_r("beta"));
  EXPECT_TRUE(avc.contains_r("gamma"));
  EXPECT_TRUE(avc.contains_r("eta"));

  EXPECT_FALSE(avc.contains_i("alpha"));
  EXPECT_FALSE(avc.contains_i("beta"));
  EXPECT_FALSE(avc.contains_i("gamma"));
  EXPECT_FALSE(avc.contains_i("eta"));
  std::vector<std::complex<double>> alpha;
  alpha.push_back(std::complex<double>(0.0, 1.0));
  EXPECT_EQ(alpha, avc.vals_c("alpha"));
  EXPECT_EQ(alpha_scalar_dim, avc.dims_r("alpha"));

  std::vector<std::complex<double>> beta;
  for (size_t i = 2; i <= 6; i += 2) {
    beta.push_back(std::complex<double>{static_cast<double>(i),
                                        static_cast<double>(i + 1)});
  }
  EXPECT_EQ(beta, avc.vals_c("beta"));
  EXPECT_EQ(beta_vec_dim, avc.dims_r("beta"));

  std::vector<std::complex<double>> gamma;
  for (size_t i = 8; i < 32; i += 2) {
    gamma.push_back(std::complex<double>(static_cast<double>(i),
                                         static_cast<double>(i + 1)));
  }
  EXPECT_EQ(gamma, avc.vals_c("gamma"));
  EXPECT_EQ(array_dim, avc.dims_r("gamma"));

  std::vector<std::complex<double>> eta;
  EXPECT_EQ(eta, avc.vals_c("eta"));
}
