#include <gtest/gtest.h>
#include <stan/optimization/lbfgs_update.hpp>

TEST(OptimizationLbfgsUpdate, lbfgs_update_secant) {
  typedef stan::optimization::LBFGSUpdate<> QNUpdateT;
  typedef QNUpdateT::VectorT VectorT;

  const unsigned int nDim = 10;
  const unsigned int maxRank = 3;
  VectorT yk(nDim), sk(nDim), sdir(nDim);

  // Construct a set of BFGS update vectors and check that
  // the secant equation H*yk = sk is always satisfied.
  for (unsigned int rank = 1; rank <= maxRank; rank++) {
    QNUpdateT bfgsUp(rank);
    for (unsigned int i = 0; i < nDim; i++) {
      sk.setZero(nDim);
      yk.setZero(nDim);
      sk[i] = 1;
      yk[i] = 1;

      bfgsUp.update(yk,sk,i==0);

      // Because the constructed update vectors are all orthogonal the secant
      // equation should be exactlty satisfied for all nDim updates.
      for (unsigned int j = 0; j <= std::min(rank,i); j++) {
        sk.setZero(nDim);
        yk.setZero(nDim);
        sk[i - j] = 1;
        yk[i - j] = 1;

        bfgsUp.search_direction(sdir,yk);
      
        EXPECT_NEAR((sdir + sk).norm(),0.0,1e-10);
      }
    }
  }
}
