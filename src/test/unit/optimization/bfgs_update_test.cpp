#include <gtest/gtest.h>
#include <stan/optimization/bfgs_update.hpp>

TEST(OptimizationBfgsUpdate, bfgs_update_secant) {
  typedef stan::optimization::BFGSUpdate_HInv<> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  const unsigned int nDim = 10;
  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  // Construct a set of BFGS update vectors and check that
  // the secant equation H*yk = sk is always satisfied.
  for (unsigned int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk,sk,i==0);

    // Because the constructed update vectors are all orthogonal the secant
    // equation should be exactlty satisfied for all nDim updates.
    for (unsigned int j = 0; j <= i; j++) {
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
  typedef stan::optimization::BFGSUpdate_HInv<> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  const unsigned int nDim = 10;
  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  for (unsigned int i = 0; i < nDim; i++) {
    sk.setZero(nDim);
    yk.setZero(nDim);
    sk[i] = 1;
    yk[i] = 1;

    bfgsUp.update(yk,sk,i==0);
  }
}

TEST(OptimizationBfgsUpdate, BFGSUpdate_HInv_search_direction) {
  typedef stan::optimization::BFGSUpdate_HInv<> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  const unsigned int nDim = 10;
  QNUpdateT bfgsUp;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  for (unsigned int i = 0; i < nDim; i++) {

    for (unsigned int j = 0; j <= i; j++) {
      sk.setZero(nDim);
      yk.setZero(nDim);
      sk[i - j] = 1;
      yk[i - j] = 1;

      bfgsUp.search_direction(sdir,yk);
      
    }
  }
}
