#include <gtest/gtest.h>
#include <stan/optimization/bfgs_update.hpp>

TEST(OptimizationBfgsUpdate, bfgs_update_secant) {
  const int nDim = 10;

  typedef stan::optimization::BFGSUpdate_HInv<double, nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  // Construct a set of BFGS update vectors and check that
  // the secant equation H*yk = sk is always satisfied.
  for (int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk, sk, i == 0);

    // Because the constructed update vectors are all orthogonal the secant
    // equation should be exactly satisfied for all nDim updates.
    for (int j = 0; j <= i; j++) {
      sk.setZero(nDim);
      yk.setZero(nDim);
      sk[i - j] = 1;
      yk[i - j] = 1;

      bfgsUp.search_direction(sdir, yk);

      EXPECT_NEAR((sdir + sk).norm(), 0.0, 1e-10);
    }
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_enlarges_small_estimate) {
  // The reset on a pair with curvature 1e4 along e1 sets H = 1e-4 I. The next
  // pair has curvature 1 along e2, so H is first multiplied by 1e4, also
  // along e3, which no pair has explored (H e3 stayed 1e-4 e3 before). A
  // pair with curvature 100 along e3, where H is then 1, does not shrink H:
  // H e1 stays e1, although the first pair measured curvature 1e4 along e1.
  stan::optimization::BFGSUpdate_HInv<> bfgsUp;
  Eigen::VectorXd s = Eigen::VectorXd::Unit(3, 0), p;
  bfgsUp.update(1e4 * s, s, true);
  s = Eigen::VectorXd::Unit(3, 1);
  bfgsUp.update(s, s);
  bfgsUp.search_direction(p, Eigen::VectorXd::Unit(3, 2));
  EXPECT_FLOAT_EQ(-1, p[2]);
  s = Eigen::VectorXd::Unit(3, 2);
  bfgsUp.update(100 * s, s);
  bfgsUp.search_direction(p, Eigen::VectorXd::Unit(3, 0));
  EXPECT_FLOAT_EQ(-1, p[0]);
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_update) {
  const int nDim = 10;

  typedef stan::optimization::BFGSUpdate_HInv<double, nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  for (int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk, sk, i == 0);
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_search_direction) {
  const int nDim = 10;
  typedef stan::optimization::BFGSUpdate_HInv<double, nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);
  sk.setZero(nDim);
  yk.setZero(nDim);
  sk[0] = 1;
  yk[0] = 1;
  bfgsUp.update(yk, sk, true);

  for (int i = 0; i < nDim; i++) {
    for (int j = 0; j <= i; j++) {
      yk.setZero(nDim);
      yk[i - j] = 1;

      bfgsUp.search_direction(sdir, yk);
    }
  }
}
