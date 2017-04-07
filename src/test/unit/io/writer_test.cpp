#include <limits>
#include <stdexcept>

#include <stan/io/writer.hpp>
#include <stan/io/reader.hpp>
#include <gtest/gtest.h>

#include <stan/math/prim/mat.hpp>

TEST(ioWriter, infBounds) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  // lub

  // lb finite; ub = +inf
  double y = 12;
  writer.scalar_lub_unconstrain(0,std::numeric_limits<double>::infinity(),y);
  writer.scalar_lb_unconstrain(0,y);
  EXPECT_FLOAT_EQ(writer.data_r()[0], writer.data_r()[1]);

  // lb = -inf; ub finite
  double z = -7.7;
  writer.scalar_lub_unconstrain(-std::numeric_limits<double>::infinity(), -1.9, z);
  writer.scalar_ub_unconstrain(-1.9, z);
  EXPECT_FLOAT_EQ(writer.data_r()[2], writer.data_r()[3]);

  // lb = -inf;  ub = +inf
  double w = 197.345;
  writer.scalar_lub_unconstrain(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), w);
  writer.scalar_unconstrain(w);
  EXPECT_FLOAT_EQ(writer.data_r()[4], writer.data_r()[5]);

  // ub = inf
  double u = 9283475;
  writer.scalar_ub_unconstrain(std::numeric_limits<double>::infinity(),u);
  writer.scalar_unconstrain(u);
  EXPECT_FLOAT_EQ(writer.data_r()[6], writer.data_r()[7]);

  // lb = -inf
  double v = -7464.737474;
  writer.scalar_lb_unconstrain(-std::numeric_limits<double>::infinity(),v);
  writer.scalar_unconstrain(v);
  EXPECT_FLOAT_EQ(writer.data_r()[8], writer.data_r()[9]);
}

TEST(io_writer, integer) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);

  int integer = 100;
  writer.integer(integer);

  EXPECT_EQ(integer, writer.data_i()[0]);
}
TEST(io_writer, row_vector_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  int size = 3;
  stan::math::row_vector_d rv(size);
  for (int n = 0; n < size; n++)
    rv(n) = n;

  writer.row_vector_unconstrain(rv);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_EQ(n, writer.data_r()[n]);
}
TEST(io_writer, matrix_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);

  int rows = 2;
  int cols = 3;
  stan::math::matrix_d mat(rows,cols);
  mat << 
    0, 2, 4,
    1, 3, 5;

  writer.matrix_unconstrain(mat);
  ASSERT_EQ((size_t)rows*cols, writer.data_r().size());
  EXPECT_FLOAT_EQ(0, writer.data_r()[0]);
  EXPECT_FLOAT_EQ(1, writer.data_r()[1]);
  EXPECT_FLOAT_EQ(2, writer.data_r()[2]);
  EXPECT_FLOAT_EQ(3, writer.data_r()[3]);
  EXPECT_FLOAT_EQ(4, writer.data_r()[4]);
  EXPECT_FLOAT_EQ(5, writer.data_r()[5]);
}
TEST(io_writer, vector_lb_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  double lb = -2;
  int size = 3;
  stan::math::vector_d vec(size);
  for (int n = 0; n < size; n++)
    vec(n) = n;

  writer.vector_lb_unconstrain(lb,vec);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_FLOAT_EQ(log(n-lb), writer.data_r()[n]);
}
TEST(io_writer, row_vector_lb_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  double lb = -1.5;
  int size = 3;
  stan::math::row_vector_d row_vec(size);
  for (int n = 0; n < size; n++)
    row_vec(n) = n;

  writer.row_vector_lb_unconstrain(lb,row_vec);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_FLOAT_EQ(log(n-lb), writer.data_r()[n]);
}
TEST(io_writer, matrix_lb_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  double lb = -1;
  
  stan::math::matrix_d mat(3,2);
  mat << 
    0, 3,
    1, 4,
    2, 5;
  
  writer.matrix_lb_unconstrain(lb,mat);
  ASSERT_EQ(6U, writer.data_r().size());
  for (int n = 0; n < 6; n++)
    EXPECT_FLOAT_EQ(log(n-lb), writer.data_r()[n]);
}
TEST(io_writer, vector_ub_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  double ub = 10;
  int size = 3;
  stan::math::vector_d vec(size);
  for (int n = 0; n < size; n++)
    vec(n) = n;

  writer.vector_ub_unconstrain(ub,vec);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_FLOAT_EQ(log(ub - n), writer.data_r()[n]);
}
TEST(io_writer, row_vector_ub_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  double ub = 3.1;
  int size = 3;
  stan::math::row_vector_d row_vec(size);
  for (int n = 0; n < size; n++)
    row_vec(n) = n;

  writer.row_vector_ub_unconstrain(ub,row_vec);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_FLOAT_EQ(log(ub - n), writer.data_r()[n]);
}
TEST(io_writer, matrix_ub_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  
  double ub = 6;
  
  stan::math::matrix_d mat(3,2);
  mat << 
    0, 3,
    1, 4,
    2, 5;
  
  writer.matrix_ub_unconstrain(ub,mat);
  ASSERT_EQ(6U, writer.data_r().size());
  for (int n = 0; n < 6; n++)
    EXPECT_FLOAT_EQ(log(ub - n), writer.data_r()[n]);
}
TEST(io_writer, vector_lub_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);

  double lb = -10;
  double ub = 10;
  int size = 3;
  stan::math::vector_d vec(size);
  for (int n = 0; n < size; n++)
    vec(n) = n;

  writer.vector_lub_unconstrain(lb,ub,vec);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_FLOAT_EQ(stan::math::logit((n - lb) / (ub - lb)), writer.data_r()[n]);
}
TEST(io_writer, row_vector_lub_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);

  double lb = -1;
  double ub = 7;
  int size = 3;
  stan::math::row_vector_d row_vec(size);
  for (int n = 0; n < size; n++)
    row_vec(n) = n;

  writer.row_vector_lub_unconstrain(lb,ub,row_vec);
  ASSERT_EQ((size_t)size, writer.data_r().size());
  for (int n = 0; n < size; n++)
    EXPECT_FLOAT_EQ(stan::math::logit((n - lb) / (ub - lb)), writer.data_r()[n]);
}
TEST(io_writer, matrix_lub_unconstrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);

  double lb = -1;
  double ub = 6;
  
  stan::math::matrix_d mat(3,2);
  mat << 
    0, 3,
    1, 4,
    2, 5;
  
  writer.matrix_lub_unconstrain(lb,ub,mat);
  ASSERT_EQ(6U, writer.data_r().size());
  for (int n = 0; n < 6; n++)
    EXPECT_FLOAT_EQ(stan::math::logit((n - lb) / (ub - lb)), writer.data_r()[n]);
}


TEST(io_writer, scalar_pos_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;

  y = 1.0;
  EXPECT_NO_THROW(writer.scalar_pos_unconstrain(y));

  y = -1.0;
  EXPECT_THROW(writer.scalar_pos_unconstrain(y), std::runtime_error);
}
TEST(io_writer, scalar_lb_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  double lb = 0;
  y = 1.0;
  EXPECT_NO_THROW(writer.scalar_lb_unconstrain(lb, y));

  y = -1.0;
  EXPECT_THROW(writer.scalar_lb_unconstrain(lb, y), std::domain_error);
}
TEST(io_writer, scalar_ub_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  double ub = 0;
  y = -1.0;
  EXPECT_NO_THROW(writer.scalar_ub_unconstrain(ub, y));

  y = 1.0;
  EXPECT_THROW(writer.scalar_ub_unconstrain(ub, y), std::domain_error);
}
TEST(io_writer, scalar_lub_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  double lb = 0.0;
  double ub = 1.0;
  y = 0.5;
  EXPECT_NO_THROW(writer.scalar_lub_unconstrain(lb, ub, y));

  y = 2.0;
  EXPECT_THROW(writer.scalar_lub_unconstrain(lb, ub, y), std::domain_error);

  y = -2.0;
  EXPECT_THROW(writer.scalar_lub_unconstrain(lb, ub, y), std::domain_error);
}
TEST(io_writer, corr_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  y = 0.5;
  EXPECT_NO_THROW(writer.corr_unconstrain(y));

  y = 2.0;
  EXPECT_THROW(writer.corr_unconstrain(y), std::domain_error);

  y = -2.0;
  EXPECT_THROW(writer.corr_unconstrain(y), std::domain_error);
}
TEST(io_writer, prob_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  double y;
  y = 0.5;
  EXPECT_NO_THROW(writer.prob_unconstrain(y));

  y = 2.0;
  EXPECT_THROW(writer.prob_unconstrain(y), std::domain_error);

  y = -0.5;
  EXPECT_THROW(writer.prob_unconstrain(y), std::domain_error);
}
TEST(io_writer, ordered_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  
  y.resize(0);
  EXPECT_NO_THROW(writer.ordered_unconstrain(y));
  y.resize(2);
  y << 0.1, 1.0;
  EXPECT_NO_THROW(writer.ordered_unconstrain(y));

  y << -0.5, 1.0;
  EXPECT_NO_THROW(writer.ordered_unconstrain(y));

  y << 1.0, 0.1;
  EXPECT_THROW(writer.ordered_unconstrain(y), std::domain_error);
}
TEST(io_writer, positive_ordered_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  
  y.resize(0);
  EXPECT_NO_THROW(writer.positive_ordered_unconstrain(y));
  y.resize(2);
  y << 0.1, 1.0;
  EXPECT_NO_THROW(writer.positive_ordered_unconstrain(y));

  y << -0.5, 1.0;
  EXPECT_THROW(writer.positive_ordered_unconstrain(y), std::domain_error);

  y << 1.0, 0.1;
  EXPECT_THROW(writer.positive_ordered_unconstrain(y), std::domain_error);
}
TEST(io_writer, unit_vector_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  
  y.resize(2);
  y << sqrt(0.5), sqrt(0.5);
  EXPECT_NO_THROW(writer.unit_vector_unconstrain(y));
  
  y << 1.1, -0.1;
  EXPECT_THROW(writer.unit_vector_unconstrain(y), std::domain_error);

  y << 0.1, 0.1;
  EXPECT_THROW(writer.unit_vector_unconstrain(y), std::domain_error);
  
  y.resize(0);
  EXPECT_THROW(writer.unit_vector_unconstrain(y), std::invalid_argument);
}
TEST(io_writer, simplex_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y;
  
  y.resize(2);
  y << 0.5, 0.5;
  EXPECT_NO_THROW(writer.simplex_unconstrain(y));
  
  y << 1.1, -0.1;
  EXPECT_THROW(writer.simplex_unconstrain(y), std::domain_error);

  y << 0.1, 0.1;
  EXPECT_THROW(writer.simplex_unconstrain(y), std::domain_error);
  
  y.resize(0);
  EXPECT_THROW(writer.simplex_unconstrain(y), std::invalid_argument);
}
TEST(io_writer, corr_matrix_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(2,2);
  y << 1.0, 0.0, 0.0, 1.0;
  EXPECT_NO_THROW(writer.corr_matrix_unconstrain(y));
  
  y << 0.0, 1.0, 0.0, 0.0;
  EXPECT_THROW(writer.corr_matrix_unconstrain(y), std::domain_error);

  y.resize(0,0);
  EXPECT_THROW(writer.corr_matrix_unconstrain(y), std::invalid_argument);

  y.resize(1,2);
  EXPECT_THROW(writer.corr_matrix_unconstrain(y), std::invalid_argument);
}
TEST(io_writer, cov_matrix_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(2,2);
  y << 1.0, 0.0, 0.0, 1.0;
  EXPECT_NO_THROW(writer.cov_matrix_unconstrain(y));
  
  y << 0.0, 1.0, 0.0, 0.0;
  EXPECT_THROW(writer.cov_matrix_unconstrain(y), std::runtime_error);

  y.resize(0,0);
  EXPECT_THROW(writer.cov_matrix_unconstrain(y), std::runtime_error);
  
  y.resize(2,1);
  EXPECT_THROW(writer.cov_matrix_unconstrain(y), std::runtime_error);
}

TEST(io_writer, cholesky_factor_unconstrain_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  
  y.resize(1,1);
  y << 
    1.0;
  EXPECT_NO_THROW(writer.cholesky_factor_unconstrain(y));
  
  y.resize(1,1);
  y <<
    (-1.0);
  EXPECT_THROW(writer.cholesky_factor_unconstrain(y), std::domain_error);
               
  y.resize(0,0);
  EXPECT_THROW(writer.cholesky_factor_unconstrain(y), std::domain_error);
               
  y.resize(1,2);
  EXPECT_THROW(writer.cholesky_factor_unconstrain(y), std::domain_error);

  y.resize(2,1);
  y << 
    1,
    2;
  EXPECT_NO_THROW(writer.cholesky_factor_unconstrain(y));
  
  y.resize(3,3);
  y <<
    1, 0, 0,
    2, 3, 0,
    -4, -5, 6;
  EXPECT_NO_THROW(writer.cholesky_factor_unconstrain(y));
}
TEST(io_reader_writer, cholesky_factor_roundtrip) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  y.resize(3,3);
  y <<
    1, 0, 0,
    2, 3, 0,
    -4, -5, 6;
  writer.cholesky_factor_unconstrain(y);

  std::vector<double> data_r = writer.data_r();
  EXPECT_EQ(6U,data_r.size());

  std::vector<int> data_i(0);
  stan::io::reader<double> reader(data_r,data_i);

  EXPECT_EQ(6U,reader.available());

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L(reader.cholesky_factor_constrain(3,3));
  EXPECT_EQ(3,L.rows());
  EXPECT_EQ(3,L.cols());
  EXPECT_EQ(9,L.size());
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      EXPECT_FLOAT_EQ(y(m,n),L(m,n));
}
TEST(io_reader_writer, cholesky_factor_roundtrip_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  stan::io::writer<double> writer(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y;
  y.resize(4,3);
  y <<
    1, 0, 0,
    2, 3, 0,
    -4, -5, 6,
    -9, 16, -25;
    
  writer.cholesky_factor_unconstrain(y);

  std::vector<double> data_r = writer.data_r();
  EXPECT_EQ(9U,data_r.size());

  std::vector<int> data_i(0);
  stan::io::reader<double> reader(data_r,data_i);

  EXPECT_EQ(9U,reader.available());

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L(reader.cholesky_factor_constrain(4,3));
  EXPECT_EQ(4,L.rows());
  EXPECT_EQ(3,L.cols());
  EXPECT_EQ(12,L.size());
  for (int m = 0; m < 4; ++m)
    for (int n = 0; n < 3; ++n)
      EXPECT_FLOAT_EQ(y(m,n),L(m,n));
}
