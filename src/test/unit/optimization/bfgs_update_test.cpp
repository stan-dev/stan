#include <gtest/gtest.h>
#include <stan/optimization/bfgs_update.hpp>

TEST(OptimizationBfgsUpdate, bfgs_update_secant) {
  const int nDim = 10;

  typedef stan::optimization::BFGSUpdate_HInv<double,nDim> QNUpdateT;
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

    bfgsUp.update(yk,sk,i==0);

    // Because the constructed update vectors are all orthogonal the secant
    // equation should be exactlty satisfied for all nDim updates.
    for (int j = 0; j <= i; j++) {
      sk.setZero(nDim);
      yk.setZero(nDim);
      sk[i - j] = 1;
      yk[i - j] = 1;

      bfgsUp.search_direction(sdir,yk);
      
      EXPECT_NEAR((sdir + sk).norm(),0.0,1e-10);
    }
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_update) {
  const int nDim = 10;

  typedef stan::optimization::BFGSUpdate_HInv<double,nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  for (int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk,sk,i==0);
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_search_direction) {
  const int nDim = 10;
  typedef stan::optimization::BFGSUpdate_HInv<double,nDim> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);
  sk.setZero(nDim);
  yk.setZero(nDim);
  sk[0] = 1;
  yk[0] = 1;
  bfgsUp.update(yk,sk,true);

  for (int i = 0; i < nDim; i++) {

    for (int j = 0; j <= i; j++) {
      yk.setZero(nDim);
      yk[i - j] = 1;
      
      bfgsUp.search_direction(sdir,yk);
    }
  }
}
