#include <gtest/gtest.h>
#include <stan/analyze/misc/gpd_fit.cpp>

  

TEST(StanServices, khat_lx) {  
  std::vector<double> a(4);
  a[0] = .0001;
  a[1] = .123;
  a[2] = 1.44;
  a[3] = 3.141;

  Eigen::Matrix<double, -1, 1> x;
  x.resize(3, 1);
  x << .001, .002, .003;
  EXPECT_NO_THROW(stan::math::services::experimental::advi::lx(a, x));
  EXPECT_FLOAT_EQ(a[0], 5.214608);
  EXPECT_FLOAT_EQ(a[1], 5.214711);
  EXPECT_FLOAT_EQ(a[2], 5.215810);
  EXPECT_FLOAT_EQ(a[3], 5.217236);
}


TEST(StanServices, khat_adjust_wip) {
  int n = 100;
  double k_est = 1.212321;
  
  EXPECT_NO_THROW(stan::math::services::experimental::advi::adjust_k_wip(k_est, n));
  EXPECT_FLOAT_EQ(k_est, 1.147565); 
}


TEST(StanServices, khat_compute_khat) {
  double k = 1010101;
  std::vector<double> x(5);
  x[0] = 5.0; x[1] = 3.0; x[2] = 2.0;
  x[3] = 4.0; x[4] = 1.0;

  EXPECT_NO_THROW(stan::math::services::experimental::advi::compute_khat(x, 30, k));
  EXPECT_FLOAT_EQ(k, 0.1265339);
}
