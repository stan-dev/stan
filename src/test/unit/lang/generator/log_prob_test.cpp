#include <stan/lang/ast_def.cpp>
#include <stan/lang/generator.hpp>
#include <stan/io/dump.hpp>
#include <test/test-models/good/lang/test_lp.hpp>
#include <test/unit/lang/utility.hpp>
#include <gtest/gtest.h>
#include <boost/random/additive_combine.hpp>
#include <iostream>
#include <sstream>

TEST(lang, logProbPolymorphismDouble) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  test_lp_model_namespace::test_lp_model model(dump);

  std::vector<double> params_r(2);
  params_r[0] = 1.0;
  params_r[1] = -3.2;

  std::vector<int> params_i;

  Matrix<double, Dynamic, 1> params_r_vec(2);
  for (int i = 0; i < 2; ++i)
    params_r_vec(i) = params_r[i];

  double lp1 = model.log_prob<true,true>(params_r, params_i, 0);
  double lp2 = model.log_prob<true,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<true,false>(params_r, params_i, 0);
  lp2 = model.log_prob<true,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  lp1 = model.log_prob<false,true>(params_r, params_i, 0);
  lp2 = model.log_prob<false,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);
  lp1 = model.log_prob<false,false>(params_r, params_i, 0);
  lp2 = model.log_prob<false,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1, lp2);

  // only test generate_inits for doubles -- no var allowed
  std::string init_txt = "y <- c(-2.9,1.2)";
  std::stringstream init_in(init_txt);
  stan::io::dump init_dump(init_in);
  std::vector<int> params_i_init;
  std::vector<double> params_r_init;
  std::stringstream pstream;
  model.transform_inits(init_dump, params_i_init, params_r_init, &pstream);
  EXPECT_EQ(0U, params_i_init.size());
  EXPECT_EQ(2U, params_r_init.size());

  Matrix<double,Dynamic,1> params_r_vec_init;
  model.transform_inits(init_dump, params_r_vec_init, &pstream);
  EXPECT_EQ(int(params_r.size()), params_r_vec_init.size());
  for (int i = 0; i < params_r_vec_init.size(); ++i)
    EXPECT_FLOAT_EQ(params_r_init[i], params_r_vec_init(i));

  // only test write_array for doubles --- no var allowed
  std::vector<double> params_r_write(2);
  params_r_write[0] = -3.2;
  params_r_write[1] = 1.79;
  std::vector<int> params_i_write;

  Matrix<double,Dynamic,1> params_r_vec_write(2);
  params_r_vec_write << -3.2, 1.79;

  boost::ecuyer1988 rng(123);
  for (int incl_tp = 0; incl_tp < 2; ++incl_tp) {
    for (int incl_gq = 0; incl_gq < 2; ++incl_gq) {
      std::vector<double> vars_write;
      Matrix<double,Dynamic,1> vars_vec_write(17);
      model.write_array(rng, params_r_write, params_i_write, vars_write, incl_tp, incl_gq, 0);
      model.write_array(rng, params_r_vec_write, vars_vec_write, incl_tp, incl_gq, 0);
      EXPECT_EQ(int(vars_write.size()), vars_vec_write.size());
      for (int i = 0; i < vars_vec_write.size(); ++i)
        EXPECT_FLOAT_EQ(vars_write[i], vars_vec_write(i));
    }
  }

}
TEST(lang, logProbPolymorphismVar) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;

  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  test_lp_model_namespace::test_lp_model model(dump);

  std::vector<var> params_r(2);
  params_r[0] = 1.0;
  params_r[1] = -3.2;

  std::vector<int> params_i;

  Matrix<var, Dynamic, 1> params_r_vec(2);
  for (int i = 0; i < 2; ++i)
    params_r_vec(i) = params_r[i];

  var lp1 = model.log_prob<true,true>(params_r, params_i, 0);
  var lp2 = model.log_prob<true,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<true,false>(params_r, params_i, 0);
  lp2 = model.log_prob<true,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<false,true>(params_r, params_i, 0);
  lp2 = model.log_prob<false,true>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());

  lp1 = model.log_prob<false,false>(params_r, params_i, 0);
  lp2 = model.log_prob<false,false>(params_r_vec, 0);
  EXPECT_FLOAT_EQ(lp1.val(), lp2.val());
}
