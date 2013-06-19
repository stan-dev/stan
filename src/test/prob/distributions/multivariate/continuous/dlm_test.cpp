#include <gtest/gtest.h>
#include "stan/prob/distributions/multivariate/continuous/gaussian_dlm.hpp"

using Eigen::Dynamic;
using Eigen::Matrix;

using boost::math::policies::policy;
using boost::math::policies::evaluation_error;
using boost::math::policies::domain_error;
using boost::math::policies::overflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::errno_on_error;
using stan::prob::gaussian_dlm_log;

// NOTE: what does this do?
typedef policy<
  domain_error<errno_on_error>, 
  pole_error<errno_on_error>,
  overflow_error<errno_on_error>,
  evaluation_error<errno_on_error> 
  > errno_policy;

TEST(ProbDistributionsGaussianDLM,LoglikeUU) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(1, 1);
  FF << 0.585528817843856;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(1, 1);
  GG << -0.109303314681054;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> V(1, 1);
  V << 2.25500747900521;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(1, 1);
  W << 0.461487989960454;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(1, 10);
  y << 0.435794268894196, 1.22755796113506, 0.193264580222731, 1.76021889445093, 1.6470149263738, 0.059988547306031, -0.4980979884018, 1.77794742623532, -0.435458536090977, 1.1733293160216;
  double ll_expected = -21.0504107180498;

  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W);
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
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(3, 10);
  y << 0.663454889128383, 1.84506567787219, -0.887054506009854, -2.41521382778332, 3.70875481434455, -4.55234499588754, -0.605128307725781, 1.37074056153151, -2.12107763131894, 2.24236743746261, 0.386569006173759, 3.91825641599579, 3.33368860016417, -0.469849046245645, -2.19806833413509, -1.45754611697235, 2.32377488296413, 4.8514565931381, -3.3192402890674, -4.86325482585216, -1.29504029175967, -1.15116681671907, -1.20273925104474, -2.53276712020605, -1.06306293348278, -2.38535359596865, 0.693389700056057, -3.25137196974247, 1.32287059182635, 0.745762191888096;
  double ll_expected = -90.0322237154493;
  
  // the error adds up in the multivariate version due to the inversion
  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W);
  EXPECT_NEAR(lp_ref, ll_expected, 0.01);
}

TEST(ProbDistributionsGaussianDLM,LoglikeUUSeq) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(1, 1);
  FF << 0.585528817843856;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(1, 1);
  GG << -0.109303314681054;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(1, 1);
  W << 0.461487989960454;
  Eigen::Matrix<double, Eigen::Dynamic, 1> V(1);
  V << 2.25500747900521;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(1, 10);
  y << -30.2383237005506, 4.58034073011305, -0.173205689832086, 1.80027530969998, 1.64263662741284, 0.0604671098951288, -0.49815029687907, 1.77795314372528, -0.435459161031581, 1.17332938432968;
  double ll_expected = -21.0616654161768;
  double lp_ref = gaussian_dlm_log(y, FF, GG, V, W);
  EXPECT_FLOAT_EQ(lp_ref, ll_expected);
}

TEST(ProbDistributionsGaussianDLM,LoglikeMMSeq) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> FF(2, 3);
 FF << 0.585528817843856,0.709466017509524,-0.109303314681054,-0.453497173462763,0.605887455840394,-1.81795596770373;
 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> GG(2, 2);
 GG << 0.520216457554957,0.816899839520583,-0.750531994502331,-0.886357521243213;
 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W(2, 2);
 W << 2.24277594357501,-1.65863136283477,-1.65863136283477,6.69010664813895;
 Eigen::Matrix<double, Eigen::Dynamic, 1> V(3);
 V << 7.19105866377728,3.27048576782842,5.86564522448303;
 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> y(3, 10);
 y << -645.337793038888, -85.2449543825825, -1027.80947332792, 115.550478561317, 83.7383685001328, 79.50821971367, 52.2957712041988, -15.0415380740324, 124.483546197942, -34.6937213615964, -5.72301661410856, -55.3761417070498, 8.80290702350217, 4.44692709691049, -2.07768404916278, 1.57843833028236, 1.23783669851953, 15.5659876348306, -6.86188181853719, -5.50621793641665, -3.62888602818304, -0.145891471793544, -0.825028473757867, -2.52251372025581, -1.19867033251869, -2.47536209021952, 1.41122996270707, -5.95624742000524, 0.832231787736836, 2.38404919543544;
 double ll_expected = -82647.3683004951;
 double lp_ref = gaussian_dlm_log(y, FF, GG, V, W);
 EXPECT_FLOAT_EQ(lp_ref, ll_expected);
}


