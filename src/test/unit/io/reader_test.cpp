#include <stan/io/reader.hpp>
#include <gtest/gtest.h>

TEST(ioReader, zeroSizeVecs) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  stan::io::reader<double> reader(theta,theta_i);
  
  EXPECT_FLOAT_EQ(1.0, reader.scalar());  // finish available
  
  // these all fail in 2.6.3
  EXPECT_EQ(0, reader.std_vector(0).size());
  EXPECT_EQ(0, reader.vector(0).size());
  EXPECT_EQ(0, reader.row_vector(0).size());
  EXPECT_EQ(0, reader.matrix(0,3).size());
  EXPECT_EQ(0, reader.matrix(3,0).size());
}

TEST(io_reader, scalar) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double x = reader.scalar();
  EXPECT_FLOAT_EQ(1.0,x);
  double y = reader.scalar();
  EXPECT_FLOAT_EQ(2.0,y);
  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, scalar_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double x = reader.scalar_constrain();
  EXPECT_FLOAT_EQ(1.0,x);
  double y = reader.scalar_constrain();
  EXPECT_FLOAT_EQ(2.0,y);
  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, scalar_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -1.2;
  double x = reader.scalar_constrain(lp);
  EXPECT_FLOAT_EQ(1.0,x);
  EXPECT_FLOAT_EQ(-1.2,lp);
  double y = reader.scalar_constrain(lp);
  EXPECT_FLOAT_EQ(2.0,y);
  EXPECT_FLOAT_EQ(-1.2,lp);
  EXPECT_EQ(0U,reader.available());
}

TEST(io_reader, scalar_pos) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double x = reader.scalar_pos();
  EXPECT_FLOAT_EQ(1.0,x);
  double y = reader.scalar_pos();
  EXPECT_FLOAT_EQ(2.0,y);
  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, scalar_pos_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(-1.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_NO_THROW (reader.scalar_pos());
  EXPECT_THROW (reader.scalar_pos(), std::domain_error);
}
TEST(io_reader, scalar_pos_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(exp(-2.0), reader.scalar_pos_constrain());
  EXPECT_FLOAT_EQ(exp(3.0), reader.scalar_pos_constrain());
  EXPECT_FLOAT_EQ(exp(-1.0), reader.scalar_pos_constrain());
  EXPECT_FLOAT_EQ(exp(0.0), reader.scalar_pos_constrain());
}
TEST(io_reader, scalar_pos_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -1.2;
  EXPECT_FLOAT_EQ(exp(-2.0), reader.scalar_pos_constrain(lp));
  EXPECT_FLOAT_EQ(exp(3.0), reader.scalar_pos_constrain(lp));
  EXPECT_FLOAT_EQ(exp(-1.0), reader.scalar_pos_constrain(lp));
  EXPECT_FLOAT_EQ(exp(0.0), reader.scalar_pos_constrain(lp));
  EXPECT_FLOAT_EQ(-1.2 + -2.0 + 3.0 - 1.0, lp);
}



TEST(io_reader, scalar_lb) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double x = reader.scalar_lb(-2.0);
  EXPECT_FLOAT_EQ(-1.0,x);
  double y = reader.scalar_lb(1.0);
  EXPECT_FLOAT_EQ(2.0,y);
  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, scalar_lb_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_NO_THROW (reader.scalar_lb(-1.0));
  EXPECT_THROW (reader.scalar_lb(3.0), std::domain_error);
}
TEST(io_reader, scalar_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0), reader.scalar_lb_constrain(1.0));
  EXPECT_FLOAT_EQ(5.0 + exp(3.0), reader.scalar_lb_constrain(5.0));
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0), reader.scalar_lb_constrain(-2.0));
  EXPECT_FLOAT_EQ(15.0 + exp(0.0), reader.scalar_lb_constrain(15.0));
}
TEST(io_reader, scalar_lb_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -1.5;
  EXPECT_FLOAT_EQ(1.0 + exp(-2.0), reader.scalar_lb_constrain(1.0,lp));
  EXPECT_FLOAT_EQ(5.0 + exp(3.0), reader.scalar_lb_constrain(5.0,lp));
  EXPECT_FLOAT_EQ(-2.0 + exp(-1.0), reader.scalar_lb_constrain(-2.0,lp));
  EXPECT_FLOAT_EQ(15.0 + exp(0.0), reader.scalar_lb_constrain(15.0,lp));
  EXPECT_FLOAT_EQ(-1.5 - 2.0 + 3.0 - 1.0, lp);
}

TEST(io_reader, scalar_ub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double x = reader.scalar_ub(-0.5);
  EXPECT_FLOAT_EQ(-1.0,x);
  double y = reader.scalar_ub(5.0);
  EXPECT_FLOAT_EQ(2.0,y);

  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, scalar_ub_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_NO_THROW (reader.scalar_ub(-1.0));
  EXPECT_THROW (reader.scalar_ub(1.0), std::domain_error);
}
TEST(io_reader, scalar_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0), reader.scalar_ub_constrain(1.0));
  EXPECT_FLOAT_EQ(5.0 - exp(3.0), reader.scalar_ub_constrain(5.0));
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0), reader.scalar_ub_constrain(-2.0));
  EXPECT_FLOAT_EQ(15.0 - exp(0.0), reader.scalar_ub_constrain(15.0));
}
TEST(io_reader, scalar_ub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -12.9;
  EXPECT_FLOAT_EQ(1.0 - exp(-2.0), reader.scalar_ub_constrain(1.0,lp));
  EXPECT_FLOAT_EQ(5.0 - exp(3.0), reader.scalar_ub_constrain(5.0,lp));
  EXPECT_FLOAT_EQ(-2.0 - exp(-1.0), reader.scalar_ub_constrain(-2.0,lp));
  EXPECT_FLOAT_EQ(15.0 - exp(0.0), reader.scalar_ub_constrain(15.0,lp));
  EXPECT_FLOAT_EQ(-12.9 -2.0 + 3.0 - 1.0, lp);
}

TEST(io_reader, scalar_lub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  double x = reader.scalar_lub(-3.0,3.0);
  EXPECT_FLOAT_EQ(-1.0,x);
  double y = reader.scalar_lub(-3.0,3.0);
  EXPECT_FLOAT_EQ(2.0,y);

  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, scalar_lub_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-1.0);
  theta.push_back(2.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_NO_THROW (reader.scalar_lub(-2.0, 2.0));
  EXPECT_THROW (reader.scalar_lub(-1.0, 1.0), std::domain_error);
}
const double inv_logit_m2 = 0.1192029; //stan::math::inv_logit(-2.0)
const double inv_logit_m1 = 0.2689414; //stan::math::inv_logit(-1.0)
const double inv_logit_0 = 0.5;        //stan::math::inv_logit(0)
const double inv_logit_3 = 0.9525741;  //stan::math::inv_logit(3.0)

TEST(io_reader, scalar_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(inv_logit_m2, reader.scalar_lub_constrain(0.0,1.0));
  EXPECT_FLOAT_EQ(3.0 + 2.0 * inv_logit_3, reader.scalar_lub_constrain(3.0,5.0));
  EXPECT_FLOAT_EQ(-3.0 + 5.0 * inv_logit_m1, reader.scalar_lub_constrain(-3.0,2.0));
  EXPECT_FLOAT_EQ(-15.0 + 30.0 * inv_logit_0, reader.scalar_lub_constrain(-15.0,15.0));
}
TEST(io_reader, scalar_lub_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -7.2;
  EXPECT_FLOAT_EQ(0.0 + 1.0 * inv_logit_m2, reader.scalar_lub_constrain(0.0,1.0,lp));
  EXPECT_FLOAT_EQ(3.0 + 2.0 * inv_logit_3, reader.scalar_lub_constrain(3.0,5.0,lp));
  EXPECT_FLOAT_EQ(-3.0 + 5.0 * inv_logit_m1, reader.scalar_lub_constrain(-3.0,2.0,lp));
  EXPECT_FLOAT_EQ(-15.0 + 30.0 * inv_logit_0, reader.scalar_lub_constrain(-15.0,15.0,lp));
  double expected_lp = -7.2 
    + log((1.0 - 0.0) * inv_logit_m2 * (1 - inv_logit_m2)) 
    + log((5.0 - 3.0) * inv_logit_3 * (1 - inv_logit_3))
    + log((2.0 - -3.0) * inv_logit_m1 * (1 - inv_logit_m1))
    + log((15.0 - -15.0) * inv_logit_0 * (1 - inv_logit_0));
  EXPECT_FLOAT_EQ(expected_lp,lp);
}

TEST(io_reader, prob) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(0.9);
  theta.push_back(0.1);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double p1 = reader.prob();
  EXPECT_FLOAT_EQ(0.9,p1);
  double p2 = reader.prob();
  EXPECT_FLOAT_EQ(0.1,p2);
  double p3 = reader.prob();
  EXPECT_FLOAT_EQ(0.0,p3);

  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, prob_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(inv_logit_m2, reader.prob_constrain());
  EXPECT_FLOAT_EQ(inv_logit_3, reader.prob_constrain());
  EXPECT_FLOAT_EQ(inv_logit_m1, reader.prob_constrain());
  EXPECT_FLOAT_EQ(inv_logit_0, reader.prob_constrain());
}
TEST(io_reader, prob_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -0.88;
  EXPECT_FLOAT_EQ(inv_logit_m2, reader.prob_constrain(lp));
  EXPECT_FLOAT_EQ(inv_logit_3, reader.prob_constrain(lp));
  EXPECT_FLOAT_EQ(inv_logit_m1, reader.prob_constrain(lp));
  EXPECT_FLOAT_EQ(inv_logit_0, reader.prob_constrain(lp));
  double expected_lp = -0.88
    + log(inv_logit_m2 * (1.0 - inv_logit_m2))
    + log(inv_logit_3 * (1.0 - inv_logit_3))
    + log(inv_logit_m1 * (1.0 - inv_logit_m1))
    + log(inv_logit_0 * (1.0 - inv_logit_0));
  EXPECT_FLOAT_EQ(expected_lp,lp);
}


TEST(io_reader, corr) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-0.9);
  theta.push_back(0.1);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double rho1 = reader.corr();
  EXPECT_FLOAT_EQ(-0.9,rho1);
  double rho2 = reader.corr();
  EXPECT_FLOAT_EQ(0.1,rho2);
  double rho3 = reader.corr();
  EXPECT_FLOAT_EQ(0.0,rho3);

  EXPECT_EQ(0U,reader.available());
}
TEST(io_reader, corr_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-0.9);
  theta.push_back(-1.1);
  theta.push_back(1.1);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_NO_THROW (reader.corr());
  EXPECT_THROW (reader.corr(), std::domain_error);
  EXPECT_THROW (reader.corr(), std::domain_error);
}
TEST(io_reader, corr_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(tanh(-2.0), reader.corr_constrain());
  EXPECT_FLOAT_EQ(tanh(3.0), reader.corr_constrain());
  EXPECT_FLOAT_EQ(tanh(-1.0), reader.corr_constrain());
  EXPECT_FLOAT_EQ(tanh(0.0), reader.corr_constrain());
}
TEST(io_reader, corr_constrain_jacobian) {
  using std::tanh;
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -10.0;
  EXPECT_FLOAT_EQ(tanh(-2.0), reader.corr_constrain(lp));
  EXPECT_FLOAT_EQ(tanh(3.0), reader.corr_constrain(lp));
  EXPECT_FLOAT_EQ(tanh(-1.0), reader.corr_constrain(lp));
  EXPECT_FLOAT_EQ(tanh(0.0), reader.corr_constrain(lp));
  double expected_lp = -10.0
    + log(1.0 - tanh(-2.0) * tanh(-2.0))
    + log(1.0 - tanh(3.0) * tanh(3.0))
    + log(1.0 - tanh(-1.0) * tanh(-1.0))
    + log(1.0 - tanh(0.0) * tanh(0.0));
  EXPECT_FLOAT_EQ(expected_lp,lp);
}


TEST(io_reader, std_vector) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (size_t i = 0; i < 10U; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  std::vector<double> y = reader.std_vector(3);
  EXPECT_EQ(3U,y.size());
  EXPECT_FLOAT_EQ(10.0,y[0]);
  EXPECT_FLOAT_EQ(11.0,y[1]);
  EXPECT_FLOAT_EQ(12.0,y[2]);

  std::vector<double> z = reader.std_vector(17);
  EXPECT_EQ(17U,z.size());
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(30.0,a);
}

TEST(io_reader, vector) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  Eigen::Matrix<double,Eigen::Dynamic,1> y = reader.vector(4);
  EXPECT_EQ(4,y.rows());
  EXPECT_EQ(1,y.cols());
  EXPECT_EQ(4,y.size());
  EXPECT_FLOAT_EQ(7.0,y[0]);
  EXPECT_FLOAT_EQ(8.0,y[1]);
  EXPECT_FLOAT_EQ(9.0,y[2]);
  EXPECT_FLOAT_EQ(10.0,y[3]);
  
  double z = reader.scalar();
  EXPECT_FLOAT_EQ(11.0,z);
}

TEST(io_reader, row_vector) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100U; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (size_t i = 0; i < 7U; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  Eigen::Matrix<double,1,Eigen::Dynamic> y = reader.row_vector(4);
  EXPECT_EQ(4,y.cols());
  EXPECT_EQ(1,y.rows());
  EXPECT_EQ(4,y.size());
  EXPECT_FLOAT_EQ(7.0,y[0]);
  EXPECT_FLOAT_EQ(8.0,y[1]);
  EXPECT_FLOAT_EQ(9.0,y[2]);
  EXPECT_FLOAT_EQ(10.0,y[3]);
  
  double z = reader.scalar();
  EXPECT_FLOAT_EQ(11.0,z);
}


TEST(io_reader, matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix(3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(7.0,y(0,0));
  EXPECT_FLOAT_EQ(8.0,y(1,0));
  EXPECT_FLOAT_EQ(9.0,y(2,0));
  EXPECT_FLOAT_EQ(10.0,y(0,1));
  EXPECT_FLOAT_EQ(11.0,y(1,1));
  EXPECT_FLOAT_EQ(12.0,y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_constrain(3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(7.0,y(0,0));
  EXPECT_FLOAT_EQ(8.0,y(1,0));
  EXPECT_FLOAT_EQ(9.0,y(2,0));
  EXPECT_FLOAT_EQ(10.0,y(0,1));
  EXPECT_FLOAT_EQ(11.0,y(1,1));
  EXPECT_FLOAT_EQ(12.0,y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}


TEST(io_reader, unit_vector) {
  std::vector<int> theta_i(0);
  std::vector<double> theta(4,sqrt(0.25));
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y = reader.unit_vector(4);
  EXPECT_EQ(4,y.size());
  EXPECT_FLOAT_EQ(sqrt(0.25),y[0]);
  EXPECT_FLOAT_EQ(sqrt(0.25),y[1]);
  EXPECT_FLOAT_EQ(sqrt(0.25),y[2]);
  EXPECT_FLOAT_EQ(sqrt(0.25),y[3]);
}
TEST(io_reader, unit_vector_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  theta[0] = 0.00;
  theta[1] = -sqrt(0.29);
  theta[2] = sqrt(0.70);
  theta[3] = -sqrt(0.01);
  theta[4] = sqrt(1.0);
  theta[5] = sqrt(1.0);
  EXPECT_NO_THROW (reader.unit_vector(4));
  EXPECT_THROW (reader.unit_vector(2), std::domain_error);
  EXPECT_THROW (reader.unit_vector(0), std::invalid_argument);
}

TEST(io_reader, simplex) {
  std::vector<int> theta_i(0);
  std::vector<double> theta(4,0.25);
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> y = reader.simplex(4);
  EXPECT_EQ(4,y.size());
  EXPECT_FLOAT_EQ(0.25,y[0]);
  EXPECT_FLOAT_EQ(0.25,y[1]);
  EXPECT_FLOAT_EQ(0.25,y[2]);
  EXPECT_FLOAT_EQ(0.25,y[3]);
}
TEST(io_reader, simplex_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  theta[0] = 0.00;
  theta[1] = 0.29;
  theta[2] = 0.70;
  theta[3] = 0.01;
  theta[4] = 1.0;
  theta[5] = 1.0;
  EXPECT_NO_THROW (reader.simplex(4));
  EXPECT_THROW (reader.simplex(2), std::domain_error);
  EXPECT_THROW (reader.simplex(0), std::invalid_argument);
}



TEST(io_reader, ordered) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(0.0,reader.scalar()); // throw away theta[0]
  Eigen::Matrix<double,Eigen::Dynamic,1> y = reader.ordered(5);
  EXPECT_EQ(5,y.size());
  EXPECT_FLOAT_EQ(1.0,y[0]);
  EXPECT_FLOAT_EQ(2.0,y[1]);
  EXPECT_FLOAT_EQ(5.0,y[4]);
  EXPECT_FLOAT_EQ(6.0,reader.scalar());
}
TEST(io_reader, ordered_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(0.0,reader.scalar()); // throw away theta[0]
  Eigen::Matrix<double,Eigen::Dynamic,1> y = reader.ordered(5);
  EXPECT_EQ(5,y.size());
  EXPECT_FLOAT_EQ(1.0,y[0]);
  EXPECT_FLOAT_EQ(2.0,y[1]);
  EXPECT_FLOAT_EQ(5.0,y[4]);
  EXPECT_FLOAT_EQ(6.0,reader.scalar());
}
TEST(io_reader, ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> phi(reader.ordered_constrain(4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
}
TEST(io_reader, ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = 3.0;
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -101.1;
  double expected_lp = lp
    - 1.0 - 2.0 + 0.0;
  Eigen::Matrix<double,Eigen::Dynamic,1> phi(reader.ordered_constrain(4,lp));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
  EXPECT_FLOAT_EQ(expected_lp,lp);
}

TEST(io_reader, positive_ordered) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_FLOAT_EQ(0.0,reader.scalar()); // throw away theta[0]
  Eigen::Matrix<double,Eigen::Dynamic,1> y = reader.positive_ordered(5);
  EXPECT_EQ(5,y.size());
  EXPECT_FLOAT_EQ(1.0,y[0]);
  EXPECT_FLOAT_EQ(2.0,y[1]);
  EXPECT_FLOAT_EQ(5.0,y[4]);
  EXPECT_FLOAT_EQ(6.0,reader.scalar());
}

TEST(io_reader, positive_ordered_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,1> phi(reader.positive_ordered_constrain(4));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
}
TEST(io_reader, positive_ordered_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(3.0);
  theta.push_back(-1.0);
  theta.push_back(-2.0);
  theta.push_back(0.0);
  double v0 = exp(3.0);
  double v1 = v0 + exp(-1.0);
  double v2 = v1 + exp(-2.0);
  double v3 = v2 + exp(0.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -101.1;
  double expected_lp = lp
    + 3.0 - 1.0 - 2.0 + 0.0;
  Eigen::Matrix<double,Eigen::Dynamic,1> phi(reader.positive_ordered_constrain(4,lp));
  EXPECT_FLOAT_EQ(v0, phi[0]);
  EXPECT_FLOAT_EQ(v1, phi[1]);
  EXPECT_FLOAT_EQ(v2, phi[2]);
  EXPECT_FLOAT_EQ(v3, phi[3]);
  EXPECT_FLOAT_EQ(expected_lp,lp);
}

TEST(io_reader, corr_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 1.0000000000000000;
  theta[1] = 0.1817621852191252;
  theta[2] = 0.8620926037637362;
  theta[3] = 0.1817621852191252;
  theta[4] = 1.0000000000000000;
  theta[5] = 0.2248293054822660;
  theta[6] = 0.8620926037637362;
  theta[7] = 0.2248293054822660;
  theta[8] = 1.0000000000000001; // allow some tolerance
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S
    = reader.corr_matrix(3);
  EXPECT_FLOAT_EQ(theta[0], S(0,0));
}
TEST(io_reader, corr_matrix_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 1.5;
  theta[1] = 1.0;
  theta[2] = 2.0;
  theta[3] = 0.0;
  theta[4] = 1.0;
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_THROW (reader.corr_matrix(1), std::domain_error);
  EXPECT_THROW (reader.corr_matrix(2), std::domain_error);
}
TEST(io_reader,corr_matrix_constrain) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<double> reader(theta,theta_i);
  Matrix<double,Dynamic,Dynamic> R(reader.corr_matrix_constrain(3U));
  EXPECT_EQ(3,R.rows());
  EXPECT_EQ(3,R.cols());
  EXPECT_EQ(9,R.size());
  EXPECT_EQ(4U,reader.available());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_FLOAT_EQ(1.0,R(i,i));
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(R(i,j),R(j,i));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
    solver(R,Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10);
}
TEST(io_reader,corr_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -9.2;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> R(reader.corr_matrix_constrain(3U,lp));
  EXPECT_EQ(3,R.rows());
  EXPECT_EQ(3,R.cols());
  EXPECT_EQ(9,R.size());
  EXPECT_EQ(4U,reader.available());
  for (size_t i = 0; i < 3U; ++i) {
    EXPECT_FLOAT_EQ(1.0,R(i,i));
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(R(i,j),R(j,i));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
    solver(R,Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10);
  // FIXME: test jacobian
}




TEST(io_reader, cov_matrix) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 6.3295234356180128;
  theta[1] = 0.6351775806192667;
  theta[2] = 3.8081029582054304;
  theta[3] = 0.6351775806192667;
  theta[4] = 1.9293554162496527;
  theta[5] = 0.5483126868366485;
  theta[6] = 3.8081029582054304;
  theta[7] = 0.5483126868366485;
  theta[8] = 3.0827514661973088;
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S
    = reader.cov_matrix(3);
  EXPECT_FLOAT_EQ(theta[0], S(0,0));
  EXPECT_FLOAT_EQ(theta[1], S(1,0));
  EXPECT_FLOAT_EQ(theta[7], S(2,1));
  EXPECT_FLOAT_EQ(theta[8], S(2,2));
}
TEST(io_reader, cov_matrix_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = 6.3;
  theta[1] = 0.7;
  theta[2] = 0.6;
  theta[3] = 1.9;
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_THROW (reader.cov_matrix(2), std::domain_error);
  EXPECT_THROW (reader.cov_matrix(0), std::invalid_argument);
}
TEST(io_reader,cov_matrix_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S(reader.cov_matrix_constrain(3U));
  EXPECT_EQ(3,S.rows());
  EXPECT_EQ(3,S.cols());
  EXPECT_EQ(9,S.size());
  EXPECT_EQ(1U,reader.available());
  for (size_t i = 0; i < 3U; ++i)
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(S(i,j),S(j,i));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
    solver(S,Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10); // check positive definite with smallest eigenvalue > 0
}
TEST(io_reader,cov_matrix_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  theta.push_back(-2.0);
  theta.push_back(-1.0);
  theta.push_back(0.0);
  theta.push_back(0.5);
  theta.push_back(1.0);
  theta.push_back(2.0);
  theta.push_back(1.0);
  stan::io::reader<double> reader(theta,theta_i);
  double lp = -3.1;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S(reader.cov_matrix_constrain(3U,lp));

  EXPECT_EQ(3,S.rows());
  EXPECT_EQ(3,S.cols());
  EXPECT_EQ(9,S.size());
  EXPECT_EQ(1U,reader.available());
  for (size_t i = 0; i < 3U; ++i)
    for (size_t j = i + 1; j < 3U; ++j)
      EXPECT_FLOAT_EQ(S(i,j),S(j,i));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
    solver(S,Eigen::EigenvaluesOnly);
  assert(solver.eigenvalues()[0] > 1E-10); // check positive definite with smallest eigenvalue > 0
  // FIXME: test Jacobian
}



TEST(io_reader, cholesky_factor) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  // column major
  theta[0] = 1;
  theta[1] = 2;
  theta[2] = 3;

  theta[3] = 0;
  theta[4] = 4;
  theta[5] = 5;

  theta[6] = 0;
  theta[7] = 0;
  theta[8] = 6;
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S
    = reader.cholesky_factor(3,3);
  EXPECT_FLOAT_EQ(theta[0], S(0,0));
  EXPECT_FLOAT_EQ(theta[1], S(1,0));
  EXPECT_FLOAT_EQ(theta[7], S(1,2));
  EXPECT_FLOAT_EQ(theta[8], S(2,2));
}
TEST(io_reader, cholesky_factor_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  // column major
  theta[0] = 1;
  theta[1] = 2;
  theta[2] = 3;

  theta[3] = 0;
  theta[4] = 4;
  theta[5] = 5;

  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S
    = reader.cholesky_factor(3,2);
  EXPECT_FLOAT_EQ(theta[0], S(0,0));
  EXPECT_FLOAT_EQ(theta[1], S(1,0));
  EXPECT_FLOAT_EQ(theta[2], S(2,0));

  EXPECT_FLOAT_EQ(theta[3], S(0,1));
  EXPECT_FLOAT_EQ(theta[4], S(1,1));
  EXPECT_FLOAT_EQ(theta[5], S(2,1));
}


TEST(io_reader, cholesky_factor_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  theta[0] = -6.3;
  stan::io::reader<double> reader(theta,theta_i);
  EXPECT_THROW(reader.cholesky_factor(2,2), std::domain_error);
  EXPECT_THROW(reader.cholesky_factor(0,0), std::domain_error);
  
  theta[0] = 1;
  EXPECT_THROW(reader.cholesky_factor(2,3), std::domain_error);
}
TEST(io_reader,cholesky_factor_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L(reader.cholesky_factor_constrain(3U,3U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor("test_cholesky_factor_constrain","L",L));
  EXPECT_EQ(3,L.rows());
  EXPECT_EQ(3,L.cols());
  EXPECT_EQ(9,L.size());
  EXPECT_EQ(2U,reader.available());
}
TEST(io_reader,cholesky_factor_constrain_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L(reader.cholesky_factor_constrain(3U,2U));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor("test_cholesky_factor_constrain","L",L));
  EXPECT_EQ(3,L.rows());
  EXPECT_EQ(2,L.cols());
  EXPECT_EQ(6,L.size());
  EXPECT_EQ(3U,reader.available());
}
TEST(io_reader,cholesky_factor_constrain_jacobian) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 8; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  double lp = 1.9;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L(reader.cholesky_factor_constrain(3U,3U,lp));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor("test_cholesky_factor_constrain","L",L));
  EXPECT_EQ(3,L.rows());
  EXPECT_EQ(3,L.cols());
  EXPECT_EQ(9,L.size());
  EXPECT_EQ(2U,reader.available());
  EXPECT_EQ(1.9 + log(L(0,0)) + log(L(1,1)) + log(L(2,2)), 
            lp);
}
TEST(io_reader,cholesky_factor_constrain_jacobian_asymmetric) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (int i = 0; i < 12; ++i)
    theta.push_back(-static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  double lp = 1.9;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> L(reader.cholesky_factor_constrain(4U,3U,lp));
  EXPECT_NO_THROW(stan::math::check_cholesky_factor("test_cholesky_factor_constrain","L",L));
  EXPECT_EQ(4,L.rows());
  EXPECT_EQ(3,L.cols());
  EXPECT_EQ(12,L.size());
  EXPECT_EQ(3U,reader.available());
  EXPECT_EQ(1.9 + log(L(0,0)) + log(L(1,1)) + log(L(2,2)), 
            lp);
}

TEST(io_reader, cholesky_corr) {
  std::vector<int> theta_i;
  std::vector<double> theta(9);
  // column major
  theta[0] = 1;
  theta[1] = 0;
  theta[2] = 0;

  theta[3] = 0;
  theta[4] = 1;
  theta[5] = 0;

  theta[6] = 0;
  theta[7] = 0;
  theta[8] = 1;
  stan::io::reader<double> reader(theta, theta_i);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S
    = reader.cholesky_corr(3);
  EXPECT_FLOAT_EQ(theta[0], S(0,0));
  EXPECT_FLOAT_EQ(theta[1], S(1,0));
  EXPECT_FLOAT_EQ(theta[4], S(1,1));
  EXPECT_FLOAT_EQ(theta[7], S(1,2));
  EXPECT_FLOAT_EQ(theta[8], S(2,2));
}

TEST(io_reader, cholesky_corr_exception) {
  std::vector<int> theta_i;
  std::vector<double> theta(9);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> S;

  // non lower-triangular matrix (column major)
  // the rest of these tests are with check_cholesky_factor_corr
  theta[0] = 1;
  theta[1] = 0;
  theta[2] = 0;

  theta[3] = 0.5;
  theta[4] = 1;
  theta[5] = 0;

  theta[6] = 0;
  theta[7] = 0;
  theta[8] = 1;
  stan::io::reader<double> reader(theta, theta_i);
  EXPECT_THROW(reader.cholesky_corr(3),
               std::domain_error);
}


TEST(io_reader,eos_exception) {
  std::vector<double> theta;
  theta.push_back(1.0);
  theta.push_back(2.0);
  std::vector<int> theta_i;
  theta_i.push_back(1);
  stan::io::reader<double> reader(theta,theta_i);

  EXPECT_EQ(2U,reader.available());
  EXPECT_EQ(1U,reader.available_i());

  EXPECT_NO_THROW(reader.scalar());
  EXPECT_NO_THROW(reader.scalar());
  EXPECT_THROW(reader.scalar(),std::runtime_error);

  // should go back to working
  EXPECT_NO_THROW(reader.integer());
  EXPECT_THROW(reader.integer(),std::runtime_error);

  // should keep throwing
  EXPECT_THROW(reader.scalar(),std::runtime_error);
  EXPECT_THROW(reader.integer(),std::runtime_error);
}


TEST(io_reader,lub_exception) {
  std::vector<double> theta;
  for (int i = 0; i < 10; ++i)
    theta.push_back(static_cast<double>(i));
  std::vector<int> theta_i;
  for (int i = 0; i < 10; ++i)
    theta_i.push_back(i);
  stan::io::reader<double> reader(theta,theta_i);

  EXPECT_THROW(reader.scalar_lb(10.0),std::domain_error);
  EXPECT_THROW(reader.scalar_ub(-2.0),std::domain_error);
  EXPECT_THROW(reader.scalar_lub(-20.0,-18.0),std::domain_error);
  EXPECT_THROW(reader.scalar_lub(-18.0,-20.0),std::domain_error);
  EXPECT_FLOAT_EQ(4.0, reader.scalar());

  EXPECT_THROW(reader.integer_lb(10),std::runtime_error);
  EXPECT_THROW(reader.integer_ub(-2),std::runtime_error);
  EXPECT_THROW(reader.integer_lub(-20,-18),std::runtime_error);
  EXPECT_THROW(reader.integer_lub(-18,-20),std::runtime_error);
  EXPECT_EQ(4, reader.integer());
}


TEST(io_reader, matrix_lb) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double lb = -1.5;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_lb(lb,3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(7.0,y(0,0));
  EXPECT_FLOAT_EQ(8.0,y(1,0));
  EXPECT_FLOAT_EQ(9.0,y(2,0));
  EXPECT_FLOAT_EQ(10.0,y(0,1));
  EXPECT_FLOAT_EQ(11.0,y(1,1));
  EXPECT_FLOAT_EQ(12.0,y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_lb_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double lb = -1.5;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_lb_constrain(lb,3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0,lb),y(0,0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0,lb),y(1,0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0,lb),y(2,0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0,lb),y(0,1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0,lb),y(1,1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0,lb),y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_lb_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double lb = -1.5;
  double lp = -5.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_lb_constrain(lb,3,2,lp);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(7.0,lb,lp),y(0,0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(8.0,lb,lp),y(1,0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(9.0,lb,lp),y(2,0));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(10.0,lb,lp),y(0,1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(11.0,lb,lp),y(1,1));
  EXPECT_FLOAT_EQ(stan::math::lb_constrain(12.0,lb,lp),y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}


TEST(io_reader, matrix_ub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double ub = 12.5;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_ub(ub,3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(7.0,y(0,0));
  EXPECT_FLOAT_EQ(8.0,y(1,0));
  EXPECT_FLOAT_EQ(9.0,y(2,0));
  EXPECT_FLOAT_EQ(10.0,y(0,1));
  EXPECT_FLOAT_EQ(11.0,y(1,1));
  EXPECT_FLOAT_EQ(12.0,y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_ub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double ub = 14.1;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_ub_constrain(ub,3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0,ub),y(0,0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0,ub),y(1,0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0,ub),y(2,0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0,ub),y(0,1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0,ub),y(1,1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0,ub),y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_ub_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double ub = 12.1;
  double lp = -5.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_ub_constrain(ub,3,2,lp);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(7.0,ub,lp),y(0,0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(8.0,ub,lp),y(1,0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(9.0,ub,lp),y(2,0));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(10.0,ub,lp),y(0,1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(11.0,ub,lp),y(1,1));
  EXPECT_FLOAT_EQ(stan::math::ub_constrain(12.0,ub,lp),y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_lub) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double lb = 6.9;
  double ub = 12.5;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_lub(lb,ub,3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(7.0,y(0,0));
  EXPECT_FLOAT_EQ(8.0,y(1,0));
  EXPECT_FLOAT_EQ(9.0,y(2,0));
  EXPECT_FLOAT_EQ(10.0,y(0,1));
  EXPECT_FLOAT_EQ(11.0,y(1,1));
  EXPECT_FLOAT_EQ(12.0,y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_lub_constrain) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double lb = 3.5;
  double ub = 14.1;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_lub_constrain(lb,ub,3,2);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0,lb,ub),y(0,0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0,lb,ub),y(1,0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0,lb,ub),y(2,0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0,lb,ub),y(0,1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0,lb,ub),y(1,1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0,lb,ub),y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}

TEST(io_reader, matrix_lub_constrain_lp) {
  std::vector<int> theta_i;
  std::vector<double> theta;
  for (size_t i = 0; i < 100.0; ++i)
    theta.push_back(static_cast<double>(i));
  stan::io::reader<double> reader(theta,theta_i);
  for (int i = 0; i < 7; ++i) {
    double x = reader.scalar();
    EXPECT_FLOAT_EQ(static_cast<double>(i),x);
  }
  double lb = 4.1;
  double ub = 12.1;
  double lp = -5.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y = reader.matrix_lub_constrain(lb,ub,3,2,lp);
  EXPECT_EQ(3,y.rows());
  EXPECT_EQ(2,y.cols());
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(7.0,lb,ub,lp),y(0,0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(8.0,lb,ub,lp),y(1,0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(9.0,lb,ub,lp),y(2,0));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(10.0,lb,ub,lp),y(0,1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(11.0,lb,ub,lp),y(1,1));
  EXPECT_FLOAT_EQ(stan::math::lub_constrain(12.0,lb,ub,lp),y(2,1));
  
  double a = reader.scalar();
  EXPECT_FLOAT_EQ(13.0,a);
}
