#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/gaussian_dlm.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::MatrixXd;

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;
using stan::prob::gaussian_dlm_log;

typedef policy<
  domain_error<errno_on_error>,
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error>
  > errno_policy;

/*
   The log-likelihoods are compared with results from R package dlm
*/

TEST(ProbDistributionsGaussianDLM,LoglikeUU) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(1, 1);
  FF << 0.585528817843856;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(1, 1);
  GG << -0.109303314681054;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V(1, 1);
  V << 2.25500747900521;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(1, 1);
  W << 0.461487989960454;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C0(1, 1);
  C0 << 65.2373490156606;
  Eigen::Matrix<double, Eigen::Dynamic, 1> m0(1);
  m0 << 11.5829455171551;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(1, 10);
  y << -0.286804393606091, 1.30654039013044, 0.184631538931975, 1.76116251447979, 1.64691178557684, 0.0599998209370169, -0.498099220647035, 1.77794756092381, -0.435458550812876, 1.17332931763075;
  double ll_expected = -16.2484978375184;

  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W, m0, C0);
  EXPECT_FLOAT_EQ(lp_ref, ll_expected);
}

TEST(ProbDistributionsGaussianDLM,LoglikeMM) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(2, 3);
  FF << 0.585528817843856,0.709466017509524,-0.109303314681054,-0.453497173462763,0.605887455840394,-1.81795596770373;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(2, 2);
  GG << 0.520216457554957,0.816899839520583,-0.750531994502331,-0.886357521243213;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V(3, 3);
  V << 7.19105866377728,-0.311731853764732,4.87333111936296,-0.311731853764732,3.27048576782842,0.457616661474554,4.87333111936296,0.457616661474554,5.86564522448303;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(2, 2);
  W << 2.24277594357501,-1.65863136283477,-1.65863136283477,6.69010664813895;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C0(2, 2);
  C0 << 82.1224673418328,0,0,56.0195157304406;
  Eigen::Matrix<double, Eigen::Dynamic, 1> m0(2);
  m0 << -0.892071328367409,3.74785137677115;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(3, 10);
  y << 4.05787944965558, 2.129936403626, 4.7831157467878, -3.24787355040931, 3.29106435886992, -5.3704927108258, -0.816249625704044, 1.48037050701867, -2.68345235365616, 2.44624163805141, 0.409922815875619, 4.24853291677921, 3.29113479311716, -0.49506486892086, -2.23350858809309, -1.47295668380559, 2.32945737887854, 4.81422683437484, -3.30712917135304, -4.86150232097887, -1.27602161517314, -1.15325860784026, -1.20424472088483, -2.53407127990878, -1.0641380744013, -2.38506878287814, 0.690976145192563, -3.25066033978687, 1.32299515908216, 0.746844140961399;
  double ll_expected = -85.2615847497409;

  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W, m0, C0);
  // the error adds up in the multivariate version due to the inversion.
  EXPECT_NEAR(lp_ref, ll_expected, 1e-4);
}

TEST(ProbDistributionsGaussianDLM,LoglikeUUSeq) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(1, 1);
  FF << 0.585528817843856;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(1, 1);
  GG << -0.109303314681054;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(1, 1);
  W << 0.461487989960454;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C0(1, 1);
  C0 << 65.2373490156606;
  Eigen::Matrix<double, Eigen::Dynamic, 1> m0(1);
  m0 << 11.5829455171551;
  Eigen::Matrix<double, Eigen::Dynamic, 1> V(1);
  V << 2.25500747900521;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(1, 10);
  y << -0.286804393606091, 1.30654039013044, 0.184631538931975, 1.76116251447979, 1.64691178557684, 0.0599998209370169, -0.498099220647035, 1.77794756092381, -0.435458550812876, 1.17332931763075;
  double ll_expected = -16.2484978375184;

  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W, m0, C0);
  EXPECT_FLOAT_EQ(lp_ref, ll_expected);
}

TEST(ProbDistributionsGaussianDLM,LoglikeMMSeq) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(2, 3);
  FF << 0.585528817843856,0.709466017509524,-0.109303314681054,-0.453497173462763,0.605887455840394,-1.81795596770373;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(2, 2);
  GG << 0.520216457554957,0.816899839520583,-0.750531994502331,-0.886357521243213;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(2, 2);
  W << 2.24277594357501,-1.65863136283477,-1.65863136283477,6.69010664813895;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> C0(2, 2);
  C0 << 82.1224673418328,0,0,56.0195157304406;
  Eigen::Matrix<double, Eigen::Dynamic, 1> m0(2);
  m0 << -0.892071328367409,3.74785137677115;
  Eigen::Matrix<double, Eigen::Dynamic, 1> V(3);
  V << 7.19105866377728,3.27048576782842,5.86564522448303;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(3, 10);
  y << 4.6192929816929, 2.26894555443421, 3.61335021783362, -4.51389305654121, 3.08033023711521, -4.82109003178482, -2.54481105697422, 1.18754549447415, -1.42836336886182, 3.63685652388162, 0.595814660705009, 3.54442019268414, 3.1049183858329, -0.333667025669854, -4.51083833189994, -2.16199020343709, 2.0276722565752, 7.50025078627574, -4.62619641974711, -5.06870294715032, -0.305820649788242, -0.395878816467899, -1.10528492007673, -2.51313807448059, -1.44699002950331, -2.43925609241825, 0.902652349582918, -5.82732638176514, 0.861614157026216, 2.56883513585703;
  double ll_expected = -89.1533619880878;

  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W, m0, C0);
  EXPECT_FLOAT_EQ(lp_ref, ll_expected);
}

// test inputs when V is a matrix
TEST(ProbDistributionsGaussianDLM,InputMatrixV) {
  // consistent matrices
  Matrix<double, Dynamic, Dynamic> FF = MatrixXd::Random(2, 3);
  Matrix<double, Dynamic, Dynamic> GG = MatrixXd::Random(2, 2);
  Matrix<double, Dynamic, Dynamic> V = MatrixXd::Identity(3, 3);
  Matrix<double, Dynamic, Dynamic> W = MatrixXd::Identity(2, 2);
  Matrix<double, Dynamic, Dynamic> y = MatrixXd::Random(3, 5);
  Matrix<double, Dynamic, 1> m0 = Matrix<double, Dynamic, 1>::Random(2);
  Matrix<double, Dynamic, Dynamic> C0 = MatrixXd::Identity(2,2);

  // Check F
  Matrix<double, Dynamic, Dynamic> bad_FF_1 = MatrixXd::Random(4, 3);
  EXPECT_THROW(gaussian_dlm_log(y, bad_FF_1, GG, V, W, m0, C0), std::domain_error);
  Matrix<double, Dynamic, Dynamic> bad_FF_2 = MatrixXd::Random(2, 4);
  EXPECT_THROW(gaussian_dlm_log(y, bad_FF_2, GG, V, W, m0, C0), std::domain_error);

  // Check G
  Matrix<double, Dynamic, Dynamic> bad_GG_1 = MatrixXd::Random(3, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, bad_GG_1, V, W, m0, C0), std::domain_error);
  Matrix<double, Dynamic, Dynamic> bad_GG_2 = MatrixXd::Random(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, bad_GG_2, V, W, m0, C0), std::domain_error);

  // Check V
  //Not symmetric
  V(0, 2) = 1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // negative
  V(0, 2) = -1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // wrong size
  Matrix<double, Dynamic, Dynamic> V1 = MatrixXd::Identity(2, 2);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V1, W, m0, C0), std::domain_error);
  // not square
  Matrix<double, Dynamic, Dynamic> V3 = MatrixXd::Identity(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V3, W, m0, C0), std::domain_error);

  //Not symmetric
  W(0, 1) = 1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // negative
  W(0, 1) = -1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // wrong size
  Matrix<double, Dynamic, Dynamic> W1 = MatrixXd::Identity(3, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W1, m0, C0), std::domain_error);
  // not square
  Matrix<double, Dynamic, Dynamic> W2 = MatrixXd::Identity(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W2, m0, C0), std::domain_error);

  // double
  Matrix<double, Dynamic, 1> m0_bad = Matrix<double, Dynamic, 1>::Zero(4, 1);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0_bad, C0), std::domain_error);


  //Not symmetric
  C0(0, 1) = 1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // negative
  C0(0, 1) = -1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // wrong size
  Matrix<double, Dynamic, Dynamic> C0_1 = MatrixXd::Identity(3, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W1, m0, C0_1), std::domain_error);
  // not square
  Matrix<double, Dynamic, Dynamic> C0_2 = MatrixXd::Identity(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W1, m0, C0_2), std::domain_error);

}

// test inputs when V is a matrix
TEST(ProbDistributionsGaussianDLM,InputVectorV) {
  // consistent matrices
  Matrix<double, Dynamic, Dynamic> FF = MatrixXd::Random(2, 3);
  Matrix<double, Dynamic, Dynamic> GG = MatrixXd::Random(2, 2);
  Matrix<double, Dynamic, 1> V = Matrix<double, Dynamic, 1>::Constant(3, 1.0);
  Matrix<double, Dynamic, Dynamic> W = MatrixXd::Identity(2, 2);
  Matrix<double, Dynamic, Dynamic> y = MatrixXd::Random(3, 5);
  Matrix<double, Dynamic, 1> m0 = Matrix<double, Dynamic, 1>::Random(2);
  Matrix<double, Dynamic, Dynamic> C0 = MatrixXd::Identity(2,2);

  // Check F
  Matrix<double, Dynamic, Dynamic> bad_FF_1 = MatrixXd::Random(4, 3);
  EXPECT_THROW(gaussian_dlm_log(y, bad_FF_1, GG, V, W, m0, C0), std::domain_error);
  Matrix<double, Dynamic, Dynamic> bad_FF_2 = MatrixXd::Random(2, 4);
  EXPECT_THROW(gaussian_dlm_log(y, bad_FF_2, GG, V, W, m0, C0), std::domain_error);

  // Check G
  Matrix<double, Dynamic, Dynamic> bad_GG_1 = MatrixXd::Random(3, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, bad_GG_1, V, W, m0, C0), std::domain_error);
  Matrix<double, Dynamic, Dynamic> bad_GG_2 = MatrixXd::Random(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, bad_GG_2, V, W, m0, C0), std::domain_error);

  // Check V
  // negative
  V(0) = -1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // wrong size
  Matrix<double, Dynamic, 1> V1 = Matrix<double, Dynamic, 1>::Constant(2, 1.0);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V1, W, m0, C0), std::domain_error);

  //Not symmetric
  W(0, 1) = 1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // negative
  W(0, 1) = -1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // wrong size
  Matrix<double, Dynamic, Dynamic> W1 = MatrixXd::Identity(3, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W1, m0, C0), std::domain_error);
  // not square
  Matrix<double, Dynamic, Dynamic> W2 = MatrixXd::Identity(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W2, m0, C0), std::domain_error);

  // double
  Matrix<double, Dynamic, 1> m0_bad = Matrix<double, Dynamic, 1>::Zero(4, 1);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0_bad, C0), std::domain_error);


  //Not symmetric
  C0(0, 1) = 1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // negative
  C0(0, 1) = -1;
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W, m0, C0), std::domain_error);
  // wrong size
  Matrix<double, Dynamic, Dynamic> C0_1 = MatrixXd::Identity(3, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W1, m0, C0_1), std::domain_error);
  // not square
  Matrix<double, Dynamic, Dynamic> C0_2 = MatrixXd::Identity(2, 3);
  EXPECT_THROW(gaussian_dlm_log(y, FF, GG, V, W1, m0, C0_2), std::domain_error);

}
