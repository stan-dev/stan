#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/multi_student_t_log.hpp>
#include <stan/math/prim/mat/prob/multi_student_t_rng.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/fwd/scal/fun/is_nan.hpp>
#include <stan/math/rev/scal/fun/is_nan.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using std::vector;
using stan::math::multi_student_t_log;

TEST(ProbDistributionsMultiStudentT,fvar_var) {
  using stan::math::var;
  using stan::math::fvar;
  Matrix<fvar<var>,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<var>,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<var>,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  for (int i = 0; i < 3; i++) {
    y(i).d_ = 1.0;
    mu(i).d_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_ = 1.0;
  }

  fvar<var> lp = multi_student_t_log(y,nu,mu,Sigma);
  EXPECT_NEAR(-10.1246,lp.val_.val(),0.0001);
  EXPECT_NEAR(-0.0411685,lp.d_.val(),0.0001);
}

TEST(ProbDistributionsMultiStudentT,fvar_fvar_var) {
  using stan::math::var;
  using stan::math::fvar;
  Matrix<fvar<fvar<var> >,Dynamic,1> y(3,1);
  y << 2.0, -2.0, 11.0;
  Matrix<fvar<fvar<var> >,Dynamic,1> mu(3,1);
  mu << 1.0, -1.0, 3.0;
  Matrix<fvar<fvar<var> >,Dynamic,Dynamic> Sigma(3,3);
  Sigma << 9.0, -3.0, 0.0,
    -3.0,  4.0, 0.0,
    0.0, 0.0, 5.0;
  double nu = 4.0;

  for (int i = 0; i < 3; i++) {
    y(i).d_.val_ = 1.0;
    mu(i).d_.val_ = 1.0;
    for (int j = 0; j < 3; j++)
      Sigma(i,j).d_.val_ = 1.0;
  }

  fvar<fvar<var> > lp = multi_student_t_log(y,nu,mu,Sigma);
  EXPECT_NEAR(-10.1246,lp.val_.val_.val(),0.0001);
  EXPECT_NEAR(-0.0411685,lp.d_.val_.val(),0.0001);
}
