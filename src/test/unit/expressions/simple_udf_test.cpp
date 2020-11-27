#include <test/expressions/expression_test_helpers.hpp>
#include <test/test-models/expressions/simple_udf.hpp>

TEST(ExpressionTestPrim, add_udf0) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> arg_mat0
      = stan::test::make_arg<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> arg_mat1
      = stan::test::make_arg<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>();

  auto res_mat
      = simple_udf_model_namespace::add_udf(arg_mat0, arg_mat1, nullptr);

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> arg_expr0
      = stan::test::make_arg<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>();
  int counter0 = 0;
  stan::test::counterOp<double> counter_op0(&counter0);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> arg_expr1
      = stan::test::make_arg<
          Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>();
  int counter1 = 0;
  stan::test::counterOp<double> counter_op1(&counter1);

  auto res_expr = simple_udf_model_namespace::add_udf(
      arg_expr0.unaryExpr(counter_op0), arg_expr1.unaryExpr(counter_op1),
      nullptr);

  EXPECT_STAN_EQ(res_expr, res_mat);

  EXPECT_LE(counter0, 1);
  EXPECT_LE(counter1, 1);
}

TEST(ExpressionTestRev, add_udf0) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> arg_mat0
      = stan::test::make_arg<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>>();
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> arg_mat1
      = stan::test::make_arg<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>>();

  auto res_mat
      = simple_udf_model_namespace::add_udf(arg_mat0, arg_mat1, nullptr);

  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> arg_expr0
      = stan::test::make_arg<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>>();
  int counter0 = 0;
  stan::test::counterOp<stan::math::var> counter_op0(&counter0);
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> arg_expr1
      = stan::test::make_arg<
          Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic>>();
  int counter1 = 0;
  stan::test::counterOp<stan::math::var> counter_op1(&counter1);

  auto res_expr = simple_udf_model_namespace::add_udf(
      arg_expr0.unaryExpr(counter_op0), arg_expr1.unaryExpr(counter_op1),
      nullptr);

  EXPECT_STAN_EQ(res_expr, res_mat);

  EXPECT_LE(counter0, 1);
  EXPECT_LE(counter1, 1);
  (stan::test::recursive_sum(res_mat) + stan::test::recursive_sum(res_expr))
      .grad();
  EXPECT_STAN_ADJ_EQ(arg_expr0, arg_mat0);
  EXPECT_STAN_ADJ_EQ(arg_expr1, arg_mat1);
}

TEST(ExpressionTestFwd, add_udf0) {
  Eigen::Matrix<stan::math::fvar<double>, Eigen::Dynamic, Eigen::Dynamic>
      arg_mat0
      = stan::test::make_arg<Eigen::Matrix<stan::math::fvar<double>,
                                           Eigen::Dynamic, Eigen::Dynamic>>();
  Eigen::Matrix<stan::math::fvar<double>, Eigen::Dynamic, Eigen::Dynamic>
      arg_mat1
      = stan::test::make_arg<Eigen::Matrix<stan::math::fvar<double>,
                                           Eigen::Dynamic, Eigen::Dynamic>>();

  auto res_mat
      = simple_udf_model_namespace::add_udf(arg_mat0, arg_mat1, nullptr);

  Eigen::Matrix<stan::math::fvar<double>, Eigen::Dynamic, Eigen::Dynamic>
      arg_expr0
      = stan::test::make_arg<Eigen::Matrix<stan::math::fvar<double>,
                                           Eigen::Dynamic, Eigen::Dynamic>>();
  int counter0 = 0;
  stan::test::counterOp<stan::math::fvar<double>> counter_op0(&counter0);
  Eigen::Matrix<stan::math::fvar<double>, Eigen::Dynamic, Eigen::Dynamic>
      arg_expr1
      = stan::test::make_arg<Eigen::Matrix<stan::math::fvar<double>,
                                           Eigen::Dynamic, Eigen::Dynamic>>();
  int counter1 = 0;
  stan::test::counterOp<stan::math::fvar<double>> counter_op1(&counter1);

  auto res_expr = simple_udf_model_namespace::add_udf(
      arg_expr0.unaryExpr(counter_op0), arg_expr1.unaryExpr(counter_op1),
      nullptr);

  EXPECT_STAN_EQ(res_expr, res_mat);

  EXPECT_LE(counter0, 1);
  EXPECT_LE(counter1, 1);
}
