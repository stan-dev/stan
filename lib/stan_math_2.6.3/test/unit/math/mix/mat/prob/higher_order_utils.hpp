#include <cmath>
#include <vector>
#include <iomanip>

void test_hess_eq(Eigen::Matrix<double, -1, -1> hess_1,
                    Eigen::Matrix<double, -1, -1> hess_2) {
  for (int i = 0; i < hess_1.size(); ++i){
    double tolerance = 1e-6 * fmax(fabs(hess_1(i)), fabs(hess_2(i))) + 1e-10;
    EXPECT_NEAR(hess_1(i),hess_2(i), tolerance);
  }
}

void test_grad_hess_eq(std::vector<Eigen::Matrix<double, -1, -1> > g_hess_1,
                         std::vector<Eigen::Matrix<double, -1, -1> > g_hess_2) {
  for (size_t m = 0; m < g_hess_1.size(); ++m)
    for (int i = 0; i < g_hess_1[m].size(); ++i) {
      double tolerance = 1e-6 * fmax(fabs(g_hess_1[m](i)), fabs(g_hess_2[m](i))) + 1e-11;
      EXPECT_NEAR(g_hess_1[m](i),g_hess_2[m](i), tolerance);
    }
}
