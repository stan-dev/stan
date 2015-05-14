#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/arr/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>

TEST(AgradRevErrorHandlingMatrix, checkPosDefiniteMatrix_nan) {
  using stan::math::var;
  using Eigen::Dynamic;
  using Eigen::Matrix;

  Matrix<var, Dynamic, Dynamic> y;
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::math::check_pos_definite;

  y.resize(1,1);
  y << nan;
  EXPECT_THROW(check_pos_definite("checkPosDefiniteMatrix", "y", y), 
               std::domain_error);
  
  y.resize(3,3);
  y << 2, -1, 0,
    -1, 2, -1,
    0, -1, 2;
  EXPECT_TRUE(check_pos_definite("checkPosDefiniteMatrix", "y", y));
                                 
  for (int i = 0; i < y.rows(); i++)
    for (int j = 0; j < y.cols(); j++) {
      y << 2, -1, 0, -1, 2, -1, 0, -1, 2;
      y(i,j) = nan;
      if (i >= j)
        EXPECT_THROW(check_pos_definite("checkPosDefiniteMatrix", "y", y), 
                     std::domain_error);
    }

  y << 0, 0, 0, 0, 0, 0, 0, 0, 0;
  EXPECT_THROW(check_pos_definite("checkPosDefiniteMatrix", "y", y), 
               std::domain_error);
}

