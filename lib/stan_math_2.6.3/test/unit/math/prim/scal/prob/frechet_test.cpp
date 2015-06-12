#include <stan/math/prim/scal/prob/frechet_rng.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>

TEST(ProbDistributionsFrechet, error_check) {
  boost::random::mt19937 rng;
  EXPECT_NO_THROW(stan::math::frechet_rng(2.0,3.0,rng));

  EXPECT_THROW(stan::math::frechet_rng(-2.0,3.0,rng),std::domain_error);
  EXPECT_THROW(stan::math::frechet_rng(2.0,-3.0,rng),std::domain_error);
  EXPECT_THROW(stan::math::frechet_rng(stan::math::positive_infinity(),3.0,rng),
               std::domain_error);
}

TEST(ProbDistributionsFrechet, chiSquareGoodnessFitTest) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  // boost does not provide Frechet, so we use Weibull
  // and check that Stan-generated 1/Frechet(shape,scale) fits boost Weibull(shape,1/scale)
  boost::math::weibull_distribution<>dist (2.0,5.0);
  boost::math::chi_squared mydist(K-1);

  double loc[K - 1];
  for(int i = 1; i < K; i++)
    loc[i - 1] = quantile(dist, i * std::pow(K, -1.0));

  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = 1.0 / stan::math::frechet_rng(2.0,1.0/5.0,rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}

TEST(ProbDistributionsFrechet, chiSquareGoodnessFitTest_2) {
  boost::random::mt19937 rng;
  int N = 10000;
  int K = boost::math::round(2 * std::pow(N, 0.4));
  boost::math::chi_squared mydist(K-1);

  // compare rng to values generated in R
  stan::math::vector_d loc(K - 1);
  loc << 0.0955415954270533857029, 0.1041316533397634719327, 0.1103740935559852087700,
    0.1155522740053754326972, 0.1201122408786449852203, 0.1242675441002982356098,
    0.1281388937447018039339, 0.1318020457964521607863, 0.1353081473284461211382,
    0.1386936692085097289073, 0.1419857581568354232271, 0.1452053426999067309300,
    0.1483690435024691001153, 0.1514904048409720094259, 0.1545807200109403090060,
    0.1576496031786457641122, 0.1607053971093497513056, 0.1637554713779679482766,
    0.1668064455827929504217, 0.1698643600576038303895, 0.1729348091403954956746,
    0.1760230473253571181758, 0.1791340755387774430485, 0.1822727127227262711173,
    0.1854436565133893710655, 0.1886515358368690276070, 0.1919009575713397086627,
    0.1951965489490360139424, 0.1985429970347641626116, 0.2019450863797647710562,
    0.2054077357842012396816, 0.2089360349903569558094, 0.2125352820597541936287,
    0.2162110221529071896196, 0.2199690884253143186022, 0.2238156457748539107655,
    0.2277572382230414749227, 0.2318008407862954534107, 0.2359539167955838900870,
    0.2402244817572899704405, 0.2446211750202796908482, 0.2491533407314772130547,
    0.2538311198348010000458, 0.2586655552093837240335, 0.2636687124701073692279,
    0.2688538194897237554315, 0.2742354283777497792052, 0.2798296045075444049566,
    0.2856541482738663706442, 0.2917288566621027978698, 0.2980758335169671480180,
    0.3047198597496216798675, 0.3116888378074926801986, 0.3190143288159938972370,
    0.3267322062618903677489, 0.3348834574628689608744, 0.3435151741391668234193,
    0.3526817873222083399298, 0.3624466213226111843682, 0.3728838691486778267326,
    0.3840811316557718457787, 0.3961427211814618765118, 0.4091940177230903863403,
    0.4233872987100369411628, 0.4389096706055364283117, 0.4559940614596232899558,
    0.4749347769469325353242, 0.4961100434630218436460, 0.5200155800775090320087,
    0.5473162043798307507814, 0.5789281712257871026495, 0.6161565249522205078847,
    0.6609369681056525003271, 0.7162914824634227795030, 0.7872641783601075360366,
    0.8830792885403591085947, 1.0230050174489953018764, 1.2569469392970760157624,
    1.7832436936202971100585;


  int count = 0;
  int bin [K];
  double expect [K];
  for(int i = 0 ; i < K; i++) {
    bin[i] = 0;
    expect[i] = N / K;
  }

  while (count < N) {
    double a = stan::math::frechet_rng(2.0,0.2,rng);
    int i = 0;
    while (i < K-1 && a > loc[i]) 
      ++i;
    ++bin[i];
    count++;
   }

  double chi = 0;

  for(int j = 0; j < K; j++)
    chi += ((bin[j] - expect[j]) * (bin[j] - expect[j]) / expect[j]);

  EXPECT_TRUE(chi < quantile(complement(mydist, 1e-6)));
}
