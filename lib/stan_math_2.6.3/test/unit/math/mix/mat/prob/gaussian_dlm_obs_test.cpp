#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <gtest/gtest.h>
#include <stan/math/prim/mat/prob/gaussian_dlm_obs_log.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/fwd/mat/fun/trace_quad_form.hpp>
#include <stan/math/fwd/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;
using stan::math::gaussian_dlm_obs_log;

TEST(ProbDistributionsGaussianDLM,LoglikeUU_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;

  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> FF(1, 1);
  FF << fvar<var>(0.585528817843856, 1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> GG(1, 1);
  GG << fvar<var>(-0.109303314681054,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> V(1, 1);
  V << fvar<var>(2.25500747900521,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> W(1, 1);
  W << fvar<var>(0.461487989960454,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> C0(1, 1);
  C0 << fvar<var>(65.2373490156606,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> m0(1);
  m0 << fvar<var>(11.5829455171551,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> y(1, 10);
  y << fvar<var>(-0.286804393606091,1.0), fvar<var>(1.30654039013044,1.0),
    fvar<var>(0.184631538931975,1.0), fvar<var>(1.76116251447979,1.0), 
    fvar<var>(1.64691178557684,1.0), fvar<var>(0.0599998209370169,1.0), 
    fvar<var>(-0.498099220647035,1.0), fvar<var>(1.77794756092381,1.0), 
    fvar<var>(-0.435458550812876,1.0), fvar<var>(1.17332931763075,1.0);
  double ll_expected = -16.2484978375184;

  fvar<var> lp_ref = gaussian_dlm_obs_log(y, FF, GG, V, W, m0, C0);
  EXPECT_FLOAT_EQ(ll_expected,lp_ref.val_.val());
  EXPECT_FLOAT_EQ(-3.8427677,lp_ref.d_.val());
}

TEST(ProbDistributionsGaussianDLM,LoglikeMM_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;

  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> FF(2, 3);
  FF << fvar<var>(0.585528817843856,1.0),
    fvar<var>(0.709466017509524,1.0),
    fvar<var>(-0.109303314681054,1.0),
    fvar<var>(-0.453497173462763,1.0),
    fvar<var>(0.605887455840394,1.0),
    fvar<var>(-1.81795596770373,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> GG(2, 2);
  GG << fvar<var>(0.520216457554957,1.0),
    fvar<var>(0.816899839520583,1.0),
    fvar<var>(-0.750531994502331,1.0),
    fvar<var>(-0.886357521243213,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> V(3, 3);
  V << fvar<var>(7.19105866377728,1.0),
    fvar<var>(-0.311731853764732,1.0),
    fvar<var>(4.87333111936296,1.0),
    fvar<var>(-0.311731853764732,1.0),
    fvar<var>(3.27048576782842,1.0),
    fvar<var>(0.457616661474554,1.0),
    fvar<var>(4.87333111936296,1.0),
    fvar<var>(0.457616661474554,1.0),
    fvar<var>(5.86564522448303,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> W(2, 2);
  W << fvar<var>(2.24277594357501,1.0),
    fvar<var>(-1.65863136283477,1.0),
    fvar<var>(-1.65863136283477,1.0),
    fvar<var>(6.69010664813895,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> C0(2, 2);
  C0 << fvar<var>(82.1224673418328,1.0),
    fvar<var>(0, 1.0),
    fvar<var>(0,1.0),
    fvar<var>(56.0195157304406,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, 1> m0(2);
  m0 << fvar<var>(-0.892071328367409,1.0),
    fvar<var>(3.74785137677115,1.0);
  Eigen::Matrix<fvar<var>, Eigen::Dynamic, Eigen::Dynamic> y(3, 10);
  y << fvar<var>(4.05787944965558,1.0), 
    fvar<var>(2.129936403626,1.0), 
    fvar<var>(4.7831157467878,1.0),
    fvar<var>(-3.24787355040931,1.0), 
    fvar<var>(3.29106435886992,1.0), 
    fvar<var>(-5.3704927108258,1.0),
    fvar<var>( -0.816249625704044,1.0),
    fvar<var>(1.48037050701867,1.0), 
    fvar<var>(-2.68345235365616,1.0),
    fvar<var>(2.44624163805141,1.0),
    fvar<var>(0.409922815875619,1.0),
    fvar<var>(4.24853291677921,1.0),
    fvar<var>(3.29113479311716,1.0),
    fvar<var>(-0.49506486892086,1.0),
    fvar<var>(-2.23350858809309,1.0),
    fvar<var>(-1.47295668380559,1.0),
    fvar<var>(2.32945737887854,1.0),
    fvar<var>(4.81422683437484,1.0), 
    fvar<var>(-3.30712917135304,1.0),
    fvar<var>(-4.86150232097887,1.0), 
    fvar<var>(-1.27602161517314,1.0), 
    fvar<var>(-1.15325860784026,1.0),
    fvar<var>(-1.20424472088483,1.0),
    fvar<var>(-2.53407127990878,1.0),
    fvar<var>(-1.0641380744013,1.0),
    fvar<var>(-2.38506878287814,1.0),
    fvar<var>(0.690976145192563,1.0),
    fvar<var>(-3.25066033978687,1.0),
    fvar<var>(1.32299515908216,1.0),
    fvar<var>(0.746844140961399,1.0);
  double ll_expected = -85.2615847497409;

  fvar<var> lp_ref = gaussian_dlm_obs_log(y, FF, GG, V, W, m0, C0);
  // the error adds up in the multivariate version due to the inversion.
  EXPECT_NEAR(ll_expected,lp_ref.val_.val(), 1e-4);
  EXPECT_NEAR(18.89044287309947,lp_ref.d_.val(), 1e-4);
}

TEST(ProbDistributionsGaussianDLM,LoglikeUU_fvar_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;

  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> FF(1, 1);
  FF << fvar<fvar<var> >(0.585528817843856, 1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> GG(1, 1);
  GG << fvar<fvar<var> >(-0.109303314681054,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> V(1, 1);
  V << fvar<fvar<var> >(2.25500747900521,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> W(1, 1);
  W << fvar<fvar<var> >(0.461487989960454,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> C0(1, 1);
  C0 << fvar<fvar<var> >(65.2373490156606,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1> m0(1);
  m0 << fvar<fvar<var> >(11.5829455171551,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> y(1, 10);
  y << fvar<fvar<var> >(-0.286804393606091,1.0), fvar<fvar<var> >(1.30654039013044,1.0),
    fvar<fvar<var> >(0.184631538931975,1.0), fvar<fvar<var> >(1.76116251447979,1.0), 
    fvar<fvar<var> >(1.64691178557684,1.0), fvar<fvar<var> >(0.0599998209370169,1.0), 
    fvar<fvar<var> >(-0.498099220647035,1.0), fvar<fvar<var> >(1.77794756092381,1.0), 
    fvar<fvar<var> >(-0.435458550812876,1.0), fvar<fvar<var> >(1.17332931763075,1.0);
  double ll_expected = -16.2484978375184;

  fvar<fvar<var> > lp_ref = gaussian_dlm_obs_log(y, FF, GG, V, W, m0, C0);
  EXPECT_FLOAT_EQ(ll_expected,lp_ref.val_.val_.val());
  EXPECT_FLOAT_EQ(-3.8427677,lp_ref.d_.val_.val());
}

TEST(ProbDistributionsGaussianDLM,LoglikeMM_fvar_fvar_var) {
  using stan::math::fvar;
  using stan::math::var;

  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> FF(2, 3);
  FF << fvar<fvar<var> >(0.585528817843856,1.0),
    fvar<fvar<var> >(0.709466017509524,1.0),
    fvar<fvar<var> >(-0.109303314681054,1.0),
    fvar<fvar<var> >(-0.453497173462763,1.0),
    fvar<fvar<var> >(0.605887455840394,1.0),
    fvar<fvar<var> >(-1.81795596770373,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> GG(2, 2);
  GG << fvar<fvar<var> >(0.520216457554957,1.0),
    fvar<fvar<var> >(0.816899839520583,1.0),
    fvar<fvar<var> >(-0.750531994502331,1.0),
    fvar<fvar<var> >(-0.886357521243213,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> V(3, 3);
  V << fvar<fvar<var> >(7.19105866377728,1.0),
    fvar<fvar<var> >(-0.311731853764732,1.0),
    fvar<fvar<var> >(4.87333111936296,1.0),
    fvar<fvar<var> >(-0.311731853764732,1.0),
    fvar<fvar<var> >(3.27048576782842,1.0),
    fvar<fvar<var> >(0.457616661474554,1.0),
    fvar<fvar<var> >(4.87333111936296,1.0),
    fvar<fvar<var> >(0.457616661474554,1.0),
    fvar<fvar<var> >(5.86564522448303,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> W(2, 2);
  W << fvar<fvar<var> >(2.24277594357501,1.0),
    fvar<fvar<var> >(-1.65863136283477,1.0),
    fvar<fvar<var> >(-1.65863136283477,1.0),
    fvar<fvar<var> >(6.69010664813895,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> C0(2, 2);
  C0 << fvar<fvar<var> >(82.1224673418328,1.0),
    fvar<fvar<var> >(0, 1.0),
    fvar<fvar<var> >(0,1.0),
    fvar<fvar<var> >(56.0195157304406,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, 1> m0(2);
  m0 << fvar<fvar<var> >(-0.892071328367409,1.0),
    fvar<fvar<var> >(3.74785137677115,1.0);
  Eigen::Matrix<fvar<fvar<var> >, Eigen::Dynamic, Eigen::Dynamic> y(3, 10);
  y << fvar<fvar<var> >(4.05787944965558,1.0), 
    fvar<fvar<var> >(2.129936403626,1.0), 
    fvar<fvar<var> >(4.7831157467878,1.0),
    fvar<fvar<var> >(-3.24787355040931,1.0), 
    fvar<fvar<var> >(3.29106435886992,1.0), 
    fvar<fvar<var> >(-5.3704927108258,1.0),
    fvar<fvar<var> >( -0.816249625704044,1.0),
    fvar<fvar<var> >(1.48037050701867,1.0), 
    fvar<fvar<var> >(-2.68345235365616,1.0),
    fvar<fvar<var> >(2.44624163805141,1.0),
    fvar<fvar<var> >(0.409922815875619,1.0),
    fvar<fvar<var> >(4.24853291677921,1.0),
    fvar<fvar<var> >(3.29113479311716,1.0),
    fvar<fvar<var> >(-0.49506486892086,1.0),
    fvar<fvar<var> >(-2.23350858809309,1.0),
    fvar<fvar<var> >(-1.47295668380559,1.0),
    fvar<fvar<var> >(2.32945737887854,1.0),
    fvar<fvar<var> >(4.81422683437484,1.0), 
    fvar<fvar<var> >(-3.30712917135304,1.0),
    fvar<fvar<var> >(-4.86150232097887,1.0), 
    fvar<fvar<var> >(-1.27602161517314,1.0), 
    fvar<fvar<var> >(-1.15325860784026,1.0),
    fvar<fvar<var> >(-1.20424472088483,1.0),
    fvar<fvar<var> >(-2.53407127990878,1.0),
    fvar<fvar<var> >(-1.0641380744013,1.0),
    fvar<fvar<var> >(-2.38506878287814,1.0),
    fvar<fvar<var> >(0.690976145192563,1.0),
    fvar<fvar<var> >(-3.25066033978687,1.0),
    fvar<fvar<var> >(1.32299515908216,1.0),
    fvar<fvar<var> >(0.746844140961399,1.0);
  double ll_expected = -85.2615847497409;

  fvar<fvar<var> > lp_ref = gaussian_dlm_obs_log(y, FF, GG, V, W, m0, C0);
  // the error adds up in the multivariate version due to the inversion.
  EXPECT_NEAR(ll_expected,lp_ref.val_.val_.val(), 1e-4);
  EXPECT_NEAR(18.89044287309947,lp_ref.d_.val_.val(), 1e-4);
}
