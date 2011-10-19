#include <cmath>
#include <vector>
#include <stdexcept>
#include <gtest/gtest.h>
#include "stan/prob/online_avg.hpp"

TEST(prob_online_avg,num_dimensions) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());
}

TEST(prob_online_avg,everything) {
  std::vector<double> x(3);
  x[0] = 1.0;  x[1] = -2.0;  x[2] = 0.0;

  std::vector<double> y(3);
  y[0] = 3.0; y[1] = 2.0; y[2] = 1.0;
  
  std::vector<double> z(3);
  z[0] = 2.0; z[1] = 0.0; z[2] = 0.5;

  std::vector<double> a(3);

  stan::prob::online_avg ola(3);

  EXPECT_FLOAT_EQ(0,ola.avg(0));
  EXPECT_FLOAT_EQ(0,ola.avg(1));
  EXPECT_FLOAT_EQ(0,ola.avg(2));
  
  EXPECT_FLOAT_EQ(0.0,ola.sample_variance(0));
  EXPECT_FLOAT_EQ(0.0,ola.sample_variance(1));
  EXPECT_FLOAT_EQ(0.0,ola.sample_variance(2));

  EXPECT_FLOAT_EQ(0.0,ola.sample_deviation(0));
  EXPECT_FLOAT_EQ(0.0,ola.sample_deviation(1));
  EXPECT_FLOAT_EQ(0.0,ola.sample_deviation(2));

  EXPECT_EQ(0U,ola.num_samples());

  ola.add(x);
  EXPECT_FLOAT_EQ(1.0,ola.avg(0));
  EXPECT_FLOAT_EQ(-2.0,ola.avg(1));
  EXPECT_FLOAT_EQ(0.0,ola.avg(2));

  EXPECT_FLOAT_EQ(0.0,ola.sample_variance(0));
  EXPECT_FLOAT_EQ(0.0,ola.sample_variance(1));
  EXPECT_FLOAT_EQ(0.0,ola.sample_variance(2));

  EXPECT_FLOAT_EQ(0.0,ola.sample_deviation(0));
  EXPECT_FLOAT_EQ(0.0,ola.sample_deviation(1));
  EXPECT_FLOAT_EQ(0.0,ola.sample_deviation(2));

  EXPECT_EQ(1U,ola.num_samples());

  ola.add(y);

  EXPECT_FLOAT_EQ(2.0,ola.avg(0));
  EXPECT_FLOAT_EQ(0.0,ola.avg(1));
  EXPECT_FLOAT_EQ(0.5,ola.avg(2));
  ola.avgs(a);
  EXPECT_FLOAT_EQ(2.0,a[0]);
  EXPECT_FLOAT_EQ(0.0,a[1]);
  EXPECT_FLOAT_EQ(0.5,a[2]);

  EXPECT_FLOAT_EQ(2.0,ola.sample_variance(0));
  EXPECT_FLOAT_EQ(8.0,ola.sample_variance(1));
  EXPECT_FLOAT_EQ(0.5,ola.sample_variance(2));
  ola.sample_variances(a);
  EXPECT_FLOAT_EQ(2.0,a[0]);
  EXPECT_FLOAT_EQ(8.0,a[1]);
  EXPECT_FLOAT_EQ(0.5,a[2]);

  EXPECT_FLOAT_EQ(sqrt(2.0),ola.sample_deviation(0));
  EXPECT_FLOAT_EQ(sqrt(8.0),ola.sample_deviation(1));
  EXPECT_FLOAT_EQ(sqrt(0.5),ola.sample_deviation(2));
  ola.sample_deviations(a);
  EXPECT_FLOAT_EQ(sqrt(2.0),a[0]);
  EXPECT_FLOAT_EQ(sqrt(8.0),a[1]);
  EXPECT_FLOAT_EQ(sqrt(0.5),a[2]);
  

  EXPECT_EQ(2U,ola.num_samples());

  ola.add(z);
  EXPECT_FLOAT_EQ(2.0,ola.avg(0));
  EXPECT_FLOAT_EQ(0.0,ola.avg(1));
  EXPECT_FLOAT_EQ(0.5,ola.avg(2));

  EXPECT_FLOAT_EQ(1.0,ola.sample_variance(0));
  EXPECT_FLOAT_EQ(4.0,ola.sample_variance(1));
  EXPECT_FLOAT_EQ(0.25,ola.sample_variance(2));

  EXPECT_FLOAT_EQ(1.0,ola.sample_deviation(0));
  EXPECT_FLOAT_EQ(2.0,ola.sample_deviation(1));
  EXPECT_FLOAT_EQ(0.5,ola.sample_deviation(2));

  EXPECT_EQ(3U,ola.num_samples());

  ola.remove(y);
  EXPECT_FLOAT_EQ(1.5,ola.avg(0));
  EXPECT_FLOAT_EQ(-1.0,ola.avg(1));
  EXPECT_FLOAT_EQ(0.25,ola.avg(2));
  
  EXPECT_FLOAT_EQ(0.5,ola.sample_variance(0));
  EXPECT_FLOAT_EQ(2.0,ola.sample_variance(1));
  EXPECT_FLOAT_EQ(0.125,ola.sample_variance(2));

  EXPECT_FLOAT_EQ(sqrt(0.5),ola.sample_deviation(0));
  EXPECT_FLOAT_EQ(sqrt(2.0),ola.sample_deviation(1));
  EXPECT_FLOAT_EQ(sqrt(0.125),ola.sample_deviation(2));

  EXPECT_EQ(2U,ola.num_samples());
}
TEST(prob_online_avg,add_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());

  std::vector<double> x(2);
  EXPECT_THROW (ola.add(x), std::runtime_error);
}
TEST(prob_online_avg,remove_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(0U,ola.num_samples());

  std::vector<double> x;
  EXPECT_THROW (ola.remove(x), std::runtime_error);
}
TEST(prob_online_avg,avg_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());

  ola.avg(0);
  ola.avg(1);
  ola.avg(2);
  EXPECT_THROW (ola.avg(3), std::runtime_error);
}
TEST(prob_online_avg,sample_variance_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());

  ola.sample_variance(0);
  ola.sample_variance(1);
  ola.sample_variance(2);
  EXPECT_THROW (ola.sample_variance(3), std::runtime_error);
}
TEST(prob_online_avg,avgs_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());

  std::vector<double> x(3);
  ola.avgs(x);
  x.resize(2);
  EXPECT_THROW (ola.avgs(x), std::runtime_error);
}
TEST(prob_online_avg,sample_variances_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());

  std::vector<double> x(3);
  ola.sample_variances(x);
  x.resize(2);
  EXPECT_THROW (ola.sample_variances(x), std::runtime_error);
}
TEST(prob_online_avg,sample_deviations_exception) {
  stan::prob::online_avg ola(3);
  EXPECT_EQ(3U,ola.num_dimensions());

  std::vector<double> x(3);
  ola.sample_deviations(x);
  x.resize(2);
  EXPECT_THROW (ola.sample_deviations(x), std::runtime_error);
}
