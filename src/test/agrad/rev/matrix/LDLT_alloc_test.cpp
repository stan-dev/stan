#include <stan/agrad/rev/matrix/LDLT_alloc.hpp>
#include <gtest/gtest.h>

TEST(AgradRevMatrix,LDLT_alloc) {
  using stan::agrad::LDLT_alloc;
  using stan::agrad::var;

  LDLT_alloc<-1,-1> *alloc = new LDLT_alloc<-1,-1>(); // DO NOT DELETE, allocated on the vari stack

  Eigen::Matrix<var,-1,-1> M(2,2);

  M << 2,1,1,2;

  EXPECT_NO_THROW(alloc->compute(M));

  EXPECT_EQ(alloc->N_,2);

  EXPECT_FLOAT_EQ(alloc->log_abs_det(),1.0986122886681096);

}

