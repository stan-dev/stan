#include <stan/io/serializer.hpp>
// expect_near_rel comes from lib/stan_math
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

// vector

TEST(serializer_varvector, read) {
  using stan::math::var_value;
  std::vector<stan::math::var> theta(10, 0.0);
  Eigen::VectorXd x_val(10);
  for (size_t i = 0; i < 10U; ++i) {
    x_val.coeffRef(i) = -static_cast<double>(i);
  }
  var_value<Eigen::VectorXd> x(x_val);
  stan::io::serializer<stan::math::var> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 10U; ++i) {
    EXPECT_FLOAT_EQ(theta[i].val(), x_val[i]);
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// row vector

TEST(serializer_varrowvector, read) {
  using stan::math::var_value;
  std::vector<stan::math::var> theta(10.0, 0.0);
  Eigen::RowVectorXd x_val(10);
  for (size_t i = 0; i < 10U; ++i) {
    x_val.coeffRef(i) = -static_cast<double>(i);
  }
  var_value<Eigen::RowVectorXd> x(x_val);
  stan::io::serializer<stan::math::var> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 10U; ++i) {
    EXPECT_FLOAT_EQ(theta[i].val(), x_val[i]);
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// matrix

TEST(serializer_varmatrix, read) {
  using stan::math::var_value;
  std::vector<stan::math::var> theta(16, 0.0);
  Eigen::MatrixXd x_val(4, 4);
  for (size_t i = 0; i < 16U; ++i) {
    x_val.coeffRef(i) = -static_cast<double>(i);
  }
  var_value<Eigen::MatrixXd> x(x_val);
  stan::io::serializer<stan::math::var> serializer(theta);
  serializer.write(x);
  for (size_t i = 0; i < 16U; ++i) {
    EXPECT_FLOAT_EQ(theta[i].val(), x_val(i));
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// array

TEST(serializer_stdvector_varmat, read) {
  using stan::math::var_value;
  std::vector<stan::math::var> theta(18, 0.0);
  std::vector<var_value<Eigen::MatrixXd>> x;
  for (size_t i = 0; i < 2U; ++i) {
    x.push_back(Eigen::MatrixXd::Random(3, 3));
  }
  stan::io::serializer<stan::math::var> serializer(theta);
  serializer.write(x);
  int sentinal = 0;
  for (size_t i = 0; i < 2U; ++i) {
    for (size_t j = 0; j < 9U; ++j) {
      EXPECT_FLOAT_EQ(theta[sentinal].val(), x[i].val()(j));
      ++sentinal;
    }
  }
  EXPECT_THROW(serializer.write(4), std::runtime_error);
}

// size zero

TEST(serializer, zeroSizeVecs) {
  std::vector<int> theta_i;
  std::vector<stan::math::var> theta;
  theta.push_back(1.0);
  stan::io::serializer<stan::math::var> serializer(theta);

  EXPECT_NO_THROW(serializer.write(1.5));  // finish available

  EXPECT_NO_THROW(serializer.write(
      stan::math::var_value<Eigen::VectorXd>(Eigen::VectorXd(0))));
}

// out of memory

TEST(serializer, eos_exception) {
  std::vector<stan::math::var> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::serializer<stan::math::var> serializer(theta);

  EXPECT_EQ(2U, serializer.available());

  EXPECT_NO_THROW(serializer.write(double{1}));
  EXPECT_NO_THROW(serializer.write(double{1}));
  EXPECT_THROW(serializer.write(double{1}), std::runtime_error);

  // should keep throwing
  EXPECT_THROW(serializer.write(double{1}), std::runtime_error);

  {
    std::vector<stan::math::var> theta(1);
    stan::io::serializer<stan::math::var> serializer(theta);
    EXPECT_THROW(serializer.write(stan::math::var_value<Eigen::VectorXd>(
                     Eigen::VectorXd(2))),
                 std::runtime_error);
  }
}
