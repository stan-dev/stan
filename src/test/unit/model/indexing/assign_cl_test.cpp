#ifdef STAN_OPENCL
#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/assign.hpp>
#include <stan/model/indexing/assign_cl.hpp>
#include <stan/model/indexing/rvalue.hpp>
#include <stan/model/indexing/rvalue_cl.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/model/indexing/util_cl.hpp>
#include <tuple>

using stan::model::assign;
using stan::model::rvalue;

using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;

TEST(ModelIndexing, assign_opencl_vector_1d) {
  Eigen::VectorXd m1(4);
  m1 << 1, 2, 3, 4;
  Eigen::VectorXd m2(4);
  m2 << 4, 5, 6, 7;
  stan::math::matrix_cl<double> m1_cl(m1);
  stan::math::matrix_cl<double> m2_cl(m2);
  Eigen::VectorXi m1_i(4);
  m1_i << 1, 3, 5, 7;
  Eigen::VectorXi m2_i(4);
  m2_i << 2, 4, 6, 8;
  stan::math::matrix_cl<int> m1_i_cl(m1_i);
  stan::math::matrix_cl<int> m2_i_cl(m2_i);
  stan::math::matrix_cl<double> m_err(5, 5);
  stan::math::matrix_cl<int> m_i_err(5, 5);
  stan::math::matrix_cl<double> m_empty_cl(0, 1);
  stan::math::matrix_cl<int> m_i_empty_cl(0, 1);
  auto indices = std::make_tuple(
      index_omni(), index_multi(std::vector<int>{1, 3, 2}), index_min(2),
      index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  constexpr int N_ind = std::tuple_size<decltype(indices)>::value;
  stan::math::for_each(
      [&](auto index) {
        // prim
        Eigen::VectorXd m_test = m1;
        stan::math::matrix_cl<double> m_test_cl = m1_cl;
        Eigen::VectorXi m_i_test = m1_i;
        stan::math::matrix_cl<int> m_i_test_cl = m1_i_cl;

        auto index_cl = opencl_index(index);

        assign(m_test, rvalue(m2, "rvalue double", index), "assign double",
               index);
        assign(m_test_cl, rvalue(m2_cl, "rvalue double cl", index_cl),
               "assign double cl", index_cl);
        Eigen::VectorXd test1 = stan::math::from_matrix_cl(m_test_cl);
        assign(m_i_test, rvalue(m2_i, "rvalue int", index), "assign int",
               index);
        assign(m_i_test_cl, rvalue(m2_i_cl, "rvalue int cl", index_cl),
               "assign int cl", index_cl);
        EXPECT_MATRIX_EQ(m_test, test1);
        Eigen::VectorXi test2 = stan::math::from_matrix_cl(m_i_test_cl);
        EXPECT_MATRIX_EQ(m_i_test, test2);

        EXPECT_THROW(assign(m_test_cl, m_err, "double cl assign err", index_cl),
                     std::invalid_argument);
        EXPECT_THROW(
            assign(m_i_test_cl, m_i_err, "int cl assign err", index_cl),
            std::invalid_argument);
        // for index_omni this throws invalid_argument and for others
        // out_of_range
        EXPECT_ANY_THROW(assign(m_empty_cl,
                                rvalue(m2_cl, "rvalue double cl", index_cl),
                                "double cl index err", index_cl));
        EXPECT_ANY_THROW(assign(m_i_empty_cl,
                                rvalue(m2_i_cl, "rvalue int cl", index_cl),
                                "int cl index err", index_cl));

        // rev = prim

        stan::math::vector_v m1_v1 = m1;
        stan::math::vector_v m1_v2 = m1;
        stan::math::var_value<stan::math::matrix_cl<double>> m1_v_cl
            = stan::math::to_matrix_cl(m1_v2);
        stan::math::var_value<stan::math::matrix_cl<double>> m_empty_v_cl(
            stan::math::matrix_cl<double>(0, 1));
        stan::math::vector_v m1_v11
            = m1_v1;  // workaround index_omni changing m1_v11
        assign(m1_v11, rvalue(m2, "rvalue double", index), "assign var1",
               index);
        assign(m1_v_cl, rvalue(m2_cl, "rvalue cl double", index_cl),
               "assign cl var1", index_cl);
        EXPECT_MATRIX_EQ(m1_v11.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));

        set_adjoints1(m1_v11);
        set_adjoints1(m1_v_cl);

        stan::math::grad();

        EXPECT_MATRIX_EQ(m1_v1.adj(),
                         stan::math::from_matrix_cl(m1_v_cl.adj()));
        EXPECT_MATRIX_EQ(m1_v1.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));

        EXPECT_THROW(assign(m1_v_cl, m_err, "double err1", index_cl),
                     std::invalid_argument);
        EXPECT_ANY_THROW(assign(m_empty_v_cl,
                                rvalue(m2_cl, "rvalue double cl", index_cl),
                                "double cl index err", index_cl));

        stan::math::recover_memory();

        // rev = rev

        m1_v1 = m1;
        m1_v2 = m1;
        stan::math::vector_v m2_v1 = m2;
        stan::math::vector_v m2_v2 = m2;
        m1_v_cl = stan::math::to_matrix_cl(m1_v2);
        stan::math::var_value<stan::math::matrix_cl<double>> m2_v_cl
            = stan::math::to_matrix_cl(m2_v2);
        m1_v11 = m1_v1;  // workaround index_omni changing m1_v11
        assign(m1_v11, rvalue(m2_v1, "rvalue var", index), "assign var2",
               index);
        assign(m1_v_cl, rvalue(m2_v_cl, "rvalue cl var", index_cl),
               "assign cl var2", index_cl);
        EXPECT_MATRIX_EQ(m1_v11.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));

        set_adjoints1(m1_v11);
        set_adjoints2(m2_v1);
        set_adjoints1(m1_v_cl);
        set_adjoints2(m2_v_cl);

        stan::math::grad();

        EXPECT_MATRIX_EQ(m1_v1.adj(),
                         stan::math::from_matrix_cl(m1_v_cl.adj()));
        EXPECT_MATRIX_EQ(m2_v1.adj(),
                         stan::math::from_matrix_cl(m2_v_cl.adj()));
        EXPECT_MATRIX_EQ(m1_v1.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));
        EXPECT_MATRIX_EQ(m2_v1.val(),
                         stan::math::from_matrix_cl(m2_v_cl.val()));

        stan::math::var_value<stan::math::matrix_cl<double>> m_v_err = m_err;
        EXPECT_THROW(assign(m1_v_cl, m_v_err, "double err2", index_cl),
                     std::invalid_argument);
        EXPECT_ANY_THROW(assign(m_empty_v_cl,
                                rvalue(m2_v_cl, "rvalue double cl", index_cl),
                                "double cl index err", index_cl));

        stan::math::recover_memory();
      },
      indices);
}

TEST(ModelIndexing, assign_opencl_matrix_1d) {
  Eigen::MatrixXd m1(4, 4);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7;
  Eigen::MatrixXd m2(4, 4);
  m2 << 9, 8, 7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3;
  stan::math::matrix_cl<double> m1_cl(m1);
  stan::math::matrix_cl<double> m2_cl(m2);
  Eigen::MatrixXi m1_i(4, 4);
  m1_i << 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7;
  Eigen::MatrixXi m2_i(4, 4);
  m2_i << 9, 8, 7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3;
  stan::math::matrix_cl<int> m1_i_cl(m1_i);
  stan::math::matrix_cl<int> m2_i_cl(m2_i);
  stan::math::matrix_cl<double> m_err(5, 5);
  stan::math::matrix_cl<int> m_i_err(5, 5);
  stan::math::matrix_cl<double> m_empty_cl(0, 0);
  stan::math::matrix_cl<int> m_i_empty_cl(0, 0);
  auto indices = std::make_tuple(
      index_uni(1), index_omni(), index_multi(std::vector<int>{1, 3, 2}),
      index_min(2), index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  constexpr int N_ind = std::tuple_size<decltype(indices)>::value;

  stan::math::for_each(
      [&](auto index) {
        // prim

        Eigen::MatrixXd m_test = m1;
        stan::math::matrix_cl<double> m_test_cl = m1_cl;
        Eigen::MatrixXi m_i_test = m1_i;
        stan::math::matrix_cl<int> m_i_test_cl = m1_i_cl;

        auto index_cl = opencl_index(index);

        assign(m_test, rvalue(m2, "rvalue double", index), "assign double",
               index);
        assign(m_test_cl, rvalue(m2_cl, "rvalue double cl", index_cl),
               "assign double", index_cl);
        assign(m_i_test, rvalue(m2_i, "rvalue int", index), "assign int",
               index);
        assign(m_i_test_cl, rvalue(m2_i_cl, "rvalue int cl", index_cl),
               "assign int cl", index_cl);

        EXPECT_MATRIX_EQ(m_test, stan::math::from_matrix_cl(m_test_cl));
        EXPECT_MATRIX_EQ(m_i_test, stan::math::from_matrix_cl(m_i_test_cl));

        EXPECT_THROW(assign(m_test_cl, m_err, "double err", index_cl),
                     std::invalid_argument);
        EXPECT_THROW(assign(m_i_test_cl, m_i_err, "int err", index_cl),
                     std::invalid_argument);
        // for index_omni this throws invalid_argument and for others
        // out_of_range
        EXPECT_ANY_THROW(assign(m_empty_cl,
                                rvalue(m2_cl, "rvalue double cl", index_cl),
                                "double cl index err", index_cl));
        EXPECT_ANY_THROW(assign(m_i_empty_cl,
                                rvalue(m2_i_cl, "rvalue int cl", index_cl),
                                "int cl index err", index_cl));

        // rev = prim

        stan::math::matrix_v m1_v1 = m1;
        stan::math::matrix_v m1_v2 = m1;
        stan::math::var_value<stan::math::matrix_cl<double>> m1_v_cl
            = stan::math::to_matrix_cl(m1_v2);
        stan::math::var_value<stan::math::matrix_cl<double>> m_empty_v_cl(
            stan::math::matrix_cl<double>(0, 0));
        stan::math::matrix_v m1_v11
            = m1_v1;  // workaround index_omni changing m1_v11
        assign(m1_v11, rvalue(m2, "rvalue double", index), "assign var1",
               index);
        assign(m1_v_cl, rvalue(m2_cl, "rvalue cl double", index_cl),
               "assign cl var1", index_cl);
        EXPECT_MATRIX_EQ(m1_v11.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));

        set_adjoints1(m1_v11);
        set_adjoints1(m1_v_cl);

        stan::math::grad();

        EXPECT_MATRIX_EQ(m1_v1.adj(),
                         stan::math::from_matrix_cl(m1_v_cl.adj()));
        EXPECT_MATRIX_EQ(m1_v1.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));

        EXPECT_THROW(assign(m1_v_cl, m_err, "var err1", index_cl),
                     std::invalid_argument);
        EXPECT_ANY_THROW(assign(m_empty_v_cl,
                                rvalue(m2_cl, "rvalue double cl", index_cl),
                                "double cl index err", index_cl));

        stan::math::recover_memory();

        // rev = rev

        m1_v1 = m1;
        m1_v2 = m1;
        stan::math::matrix_v m2_v1 = m2;
        stan::math::matrix_v m2_v2 = m2;
        m1_v_cl = stan::math::to_matrix_cl(m1_v2);
        stan::math::var_value<stan::math::matrix_cl<double>> m2_v_cl
            = stan::math::to_matrix_cl(m2_v2);
        m1_v11 = m1_v1;  // workaround index_omni changing m1_v11
        assign(m1_v11, rvalue(m2_v1, "rvalue var", index), "assign var", index);
        assign(m1_v_cl, rvalue(m2_v_cl, "rvalue var cl", index_cl),
               "assign var cl", index_cl);
        EXPECT_MATRIX_EQ(m1_v11.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));

        set_adjoints1(m1_v11);
        set_adjoints2(m2_v1);
        set_adjoints1(m1_v_cl);
        set_adjoints2(m2_v_cl);

        stan::math::grad();

        EXPECT_MATRIX_EQ(m1_v1.adj(),
                         stan::math::from_matrix_cl(m1_v_cl.adj()));
        EXPECT_MATRIX_EQ(m2_v1.adj(),
                         stan::math::from_matrix_cl(m2_v_cl.adj()));
        EXPECT_MATRIX_EQ(m1_v1.val(),
                         stan::math::from_matrix_cl(m1_v_cl.val()));
        EXPECT_MATRIX_EQ(m2_v1.val(),
                         stan::math::from_matrix_cl(m2_v_cl.val()));

        stan::math::var_value<stan::math::matrix_cl<double>> m_v_err = m_err;
        EXPECT_THROW(assign(m1_v_cl, m_v_err, "var err2", index_cl),
                     std::invalid_argument);
        EXPECT_ANY_THROW(assign(m_empty_v_cl,
                                rvalue(m2_v_cl, "rvalue double cl", index_cl),
                                "double cl index err", index_cl));

        stan::math::recover_memory();
      },
      indices);
}

TEST(ModelIndexing, assign_opencl_matrix_2d) {
  Eigen::MatrixXd m1(4, 4);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7;
  Eigen::MatrixXd m2(4, 4);
  m2 << 9, 8, 7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3;
  stan::math::matrix_cl<double> m1_cl(m1);
  stan::math::matrix_cl<double> m2_cl(m2);
  Eigen::MatrixXi m1_i(4, 4);
  m1_i << 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7;
  Eigen::MatrixXi m2_i(4, 4);
  m2_i << 9, 8, 7, 6, 5, 4, 3, 2, 1, 9, 8, 7, 6, 5, 4, 3;
  stan::math::matrix_cl<int> m1_i_cl(m1_i);
  stan::math::matrix_cl<int> m2_i_cl(m2_i);
  auto indices = std::make_tuple(
      index_uni(1), index_omni(), index_multi(std::vector<int>{1, 3, 2}),
      index_min(2), index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  constexpr int N_ind = std::tuple_size<decltype(indices)>::value;

  stan::math::for_each(
      [&](auto index1) {
        stan::math::for_each(
            [&](auto index2) {
              // prim

              Eigen::MatrixXd m_test = m1;
              stan::math::matrix_cl<double> m_test_cl = m1_cl;
              Eigen::MatrixXi m_i_test = m1_i;
              stan::math::matrix_cl<int> m_i_test_cl = m1_i_cl;

              auto index1_cl = opencl_index(index1);
              auto index2_cl = opencl_index(index2);

              assign(m_test, rvalue(m2, "rvalue double", index1, index2),
                     "assign double", index1, index2);
              assign(m_test_cl,
                     rvalue(m2_cl, "rvalue double cl", index1_cl, index2_cl),
                     "assign double cl", index1_cl, index2_cl);
              assign(m_i_test, rvalue(m2_i, "rvalue int", index1, index2),
                     "assign int", index1, index2);
              assign(m_i_test_cl,
                     rvalue(m2_i_cl, "rvalue int cl", index1_cl, index2_cl),
                     "assign int cl", index1_cl, index2_cl);

              EXPECT_MATRIX_EQ(m_test, stan::math::from_matrix_cl(m_test_cl));
              EXPECT_MATRIX_EQ(m_i_test,
                               stan::math::from_matrix_cl(m_i_test_cl));

              // rev = prim

              stan::math::matrix_v m1_v1 = m1;
              stan::math::matrix_v m1_v2 = m1;
              stan::math::var_value<stan::math::matrix_cl<double>> m1_v_cl
                  = stan::math::to_matrix_cl(m1_v2);
              stan::math::matrix_v m1_v11
                  = m1_v1;  // workaround index_omni changing m1_v11
              assign(m1_v11, rvalue(m2, "rvalue double", index1, index2),
                     "assign var1", index1, index2);
              assign(m1_v_cl,
                     rvalue(m2_cl, "rvalue cl double", index1_cl, index2_cl),
                     "assign cl var1", index1_cl, index2_cl);
              EXPECT_MATRIX_EQ(m1_v11.val(),
                               stan::math::from_matrix_cl(m1_v_cl.val()));

              set_adjoints1(m1_v11);
              set_adjoints1(m1_v_cl);

              stan::math::grad();

              EXPECT_MATRIX_EQ(m1_v1.adj(),
                               stan::math::from_matrix_cl(m1_v_cl.adj()));
              EXPECT_MATRIX_EQ(m1_v1.val(),
                               stan::math::from_matrix_cl(m1_v_cl.val()));

              stan::math::recover_memory();

              // rev = rev

              m1_v1 = m1;
              m1_v2 = m1;
              stan::math::matrix_v m2_v1 = m2;
              stan::math::matrix_v m2_v2 = m2;
              m1_v_cl = stan::math::to_matrix_cl(m1_v2);
              stan::math::var_value<stan::math::matrix_cl<double>> m2_v_cl
                  = stan::math::to_matrix_cl(m2_v2);
              m1_v11 = m1_v1;  // workaround index_omni changing m1_v11
              assign(m1_v11, rvalue(m2_v1, "rvalue var", index1, index2),
                     "assign var2", index1, index2);
              assign(m1_v_cl,
                     rvalue(m2_v_cl, "rvalue cl var", index1_cl, index2_cl),
                     "assign cl var2", index1_cl, index2_cl);
              EXPECT_MATRIX_EQ(m1_v11.val(),
                               stan::math::from_matrix_cl(m1_v_cl.val()));

              set_adjoints1(m1_v11);
              set_adjoints2(m2_v1);
              set_adjoints1(m1_v_cl);
              set_adjoints2(m2_v_cl);

              stan::math::grad();

              EXPECT_MATRIX_EQ(m1_v1.adj(),
                               stan::math::from_matrix_cl(m1_v_cl.adj()));
              EXPECT_MATRIX_EQ(m2_v1.adj(),
                               stan::math::from_matrix_cl(m2_v_cl.adj()));
              EXPECT_MATRIX_EQ(m1_v1.val(),
                               stan::math::from_matrix_cl(m1_v_cl.val()));
              EXPECT_MATRIX_EQ(m2_v1.val(),
                               stan::math::from_matrix_cl(m2_v_cl.val()));

              stan::math::recover_memory();
            },
            indices);
      },
      indices);
}

#endif
