#ifdef STAN_OPENCL
#include <iostream>
#include <stdexcept>
#include <vector>
#include <stan/model/indexing/rvalue.hpp>
#include <stan/model/indexing/rvalue_cl.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/model/indexing/util_cl.hpp>
#include <tuple>

using stan::model::rvalue;

using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;

template <typename T_eig, typename T_cl>
void expect_eq(const T_eig& a, const T_cl& b) {
  EXPECT_MATRIX_EQ(a, b);
}
void expect_eq(const double a, const double b) { EXPECT_EQ(a, b); }
void expect_eq(const int a, const int b) { EXPECT_EQ(a, b); }

const char* print_index_type(stan::model::index_uni) { return "index_uni"; }

const char* print_index_type(stan::model::index_multi) { return "index_multi"; }

const char* print_index_type(stan::model::index_min) { return "index_min"; }

const char* print_index_type(stan::model::index_max) { return "index_max"; }

const char* print_index_type(stan::model::index_min_max) {
  return "index_min_max";
}

const char* print_index_type(stan::model::index_omni) { return "index_omni"; }

TEST(ModelIndexing, rvalue_opencl_vector_1d) {
  Eigen::VectorXd m(4);
  m << 1, 2, 3, 4;
  stan::math::matrix_cl<double> m_cl(m);
  Eigen::VectorXi m_i(4);
  m_i << 1, 2, 3, 4;
  stan::math::matrix_cl<int> m_i_cl(m_i);
  auto indices = std::make_tuple(
      index_omni(), index_multi(std::vector<int>{1, 2, 1, 3, 1}), index_min(2),
      index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  stan::math::for_each(
      [&indices, &m, &m_cl, &m_i, &m_i_cl](const auto& ind1) {
        expect_eq(rvalue(m, "", ind1), from_matrix_cl_nonscalar(rvalue(
                                           m_cl, "", opencl_index(ind1))));
        expect_eq(rvalue(m_i, "", ind1), from_matrix_cl_nonscalar(rvalue(
                                             m_i_cl, "", opencl_index(ind1))));

        stan::math::vector_v m_v1 = m;
        stan::math::vector_v m_v2 = m;
        stan::math::var_value<stan::math::matrix_cl<double>> m_v_cl
            = stan::math::to_matrix_cl(m_v2);
        auto correct = stan::math::eval(rvalue(m_v1, "", ind1));
        auto res
            = from_matrix_cl_nonscalar(rvalue(m_v_cl, "", opencl_index(ind1)));
        expect_eq(correct.val(), res.val());
        set_adjoints1(correct);
        set_adjoints1(res);
        stan::math::grad();
        expect_eq(m_v1.adj(), m_v2.adj());
        stan::math::recover_memory();
      },
      indices);
}

TEST(ModelIndexing, rvalue_opencl_matrix_1d) {
  Eigen::MatrixXd m(4, 4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  stan::math::matrix_cl<double> m_cl(m);
  Eigen::MatrixXi m_i(4, 4);
  m_i << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  stan::math::matrix_cl<int> m_i_cl(m_i);
  auto indices = std::make_tuple(
      index_uni(3), index_omni(), index_multi(std::vector<int>{1, 2, 1, 3, 1}),
      index_min(2), index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  stan::math::for_each(
      [&indices, &m, &m_cl, &m_i, &m_i_cl](const auto& ind1) {
        expect_eq(rvalue(m, "", ind1), from_matrix_cl_nonscalar(rvalue(
                                           m_cl, "", opencl_index(ind1))));
        expect_eq(rvalue(m_i, "", ind1), from_matrix_cl_nonscalar(rvalue(
                                             m_i_cl, "", opencl_index(ind1))));

        stan::math::matrix_v m_v1 = m;
        stan::math::matrix_v m_v2 = m;
        stan::math::var_value<stan::math::matrix_cl<double>> m_v_cl
            = stan::math::to_matrix_cl(m_v2);
        auto correct = rvalue(m_v1, "", ind1);
        auto res
            = from_matrix_cl_nonscalar(rvalue(m_v_cl, "", opencl_index(ind1)));
        expect_eq(correct.val(), res.val());
        set_adjoints1(correct);
        set_adjoints1(res);
        stan::math::grad();
        expect_eq(m_v1.adj(), m_v2.adj());
        stan::math::recover_memory();
      },
      indices);
}

TEST(ModelIndexing, rvalue_opencl_matrix_2d) {
  Eigen::MatrixXd m(4, 4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  stan::math::matrix_cl<double> m_cl(m);
  Eigen::MatrixXi m_i(4, 4);
  m_i << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  stan::math::matrix_cl<int> m_i_cl(m_i);
  auto indices = std::make_tuple(
      index_uni(3), index_omni(), index_multi(std::vector<int>{1, 2, 1, 3, 1}),
      index_min(2), index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  stan::math::for_each(
      [&](const auto& ind1) {
        stan::math::for_each(
            [&](const auto& ind2) {
              expect_eq(rvalue(m, "", ind1, ind2),
                        from_matrix_cl_nonscalar(rvalue(
                            m_cl, "", opencl_index(ind1), opencl_index(ind2))));
              expect_eq(
                  rvalue(m_i, "", ind1, ind2),
                  from_matrix_cl_nonscalar(rvalue(
                      m_i_cl, "", opencl_index(ind1), opencl_index(ind2))));

              stan::math::matrix_v m_v1 = m;
              stan::math::matrix_v m_v2 = m;
              stan::math::var_value<stan::math::matrix_cl<double>> m_v_cl
                  = stan::math::to_matrix_cl(m_v2);
              auto correct = rvalue(m_v1, "", ind1, ind2);
              auto res = from_matrix_cl_nonscalar(
                  rvalue(m_v_cl, "", opencl_index(ind1), opencl_index(ind2)));
              expect_eq(correct.val(), res.val());
              set_adjoints1(correct);
              set_adjoints1(res);
              stan::math::grad();
              expect_eq(m_v1.adj(), m_v2.adj());
              stan::math::recover_memory();
            },
            indices);
      },
      indices);
}

TEST(ModelIndexing, rvalue_opencl_matrix_2d_errors) {
  Eigen::MatrixXd m(4, 4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  stan::math::matrix_cl<double> m_cl(m);
  stan::math::matrix_v m_v = m;
  stan::math::var_value<stan::math::matrix_cl<double>> m_v_cl
      = stan::math::to_matrix_cl(m_v);
  auto indices_err = std::make_tuple(
      index_uni(5), index_uni(-1), index_multi(std::vector<int>{-1, 2}),
      index_multi(std::vector<int>{5, 2}), index_min(5), index_max(0),
      index_min_max(-1, 3), index_min_max(2, 5), index_min_max(5, 1),
      index_min_max(3, -1));
  auto indices = std::make_tuple(
      index_uni(3), index_omni(), index_multi(std::vector<int>{1, 2, 1, 3, 1}),
      index_min(2), index_max(3), index_min_max(2, 3), index_min_max(3, 1));
  stan::math::for_each(
      [&indices_err, &indices, &m, &m_cl, &m_v_cl](const auto& ind) {
        stan::math::for_each(
            [&m, &m_cl, &m_v_cl, &ind](const auto& ind_err) {
              EXPECT_THROW(
                  rvalue(m_cl, "", opencl_index(ind), opencl_index(ind_err)),
                  std::out_of_range);
              EXPECT_THROW(
                  rvalue(m_cl, "", opencl_index(ind_err), opencl_index(ind)),
                  std::out_of_range);
              EXPECT_THROW(
                  rvalue(m_v_cl, "", opencl_index(ind), opencl_index(ind_err)),
                  std::out_of_range);
              EXPECT_THROW(
                  rvalue(m_v_cl, "", opencl_index(ind_err), opencl_index(ind)),
                  std::out_of_range);
            },
            indices_err);
      },
      indices);
}

TEST(ModelIndexing, rvalue_opencl_matrix_1d_errors) {
  Eigen::MatrixXd m(4, 4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
  stan::math::matrix_cl<double> m_cl(m);
  stan::math::matrix_v m_v = m;
  stan::math::var_value<stan::math::matrix_cl<double>> m_v_cl
      = stan::math::to_matrix_cl(m_v);
  auto indices = std::make_tuple(
      index_uni(5), index_uni(-1), index_multi(std::vector<int>{-1, 2}),
      index_multi(std::vector<int>{5, 2}), index_min(5), index_max(0),
      index_min_max(-1, 3), index_min_max(2, 5), index_min_max(5, 1),
      index_min_max(3, -1));
  stan::math::for_each(
      [&indices, &m, &m_cl, &m_v_cl](const auto& ind1) {
        EXPECT_THROW(rvalue(m_cl, "", opencl_index(ind1)), std::out_of_range);
        EXPECT_THROW(rvalue(m_v_cl, "", opencl_index(ind1)), std::out_of_range);
      },
      indices);
}

TEST(ModelIndexing, rvalue_opencl_vector_1d_errors) {
  Eigen::VectorXd m(4);
  m << 1, 2, 3, 4;
  stan::math::matrix_cl<double> m_cl(m);
  stan::math::vector_v m_v = m;
  stan::math::var_value<stan::math::matrix_cl<double>> m_v_cl
      = stan::math::to_matrix_cl(m_v);
  auto indices = std::make_tuple(
      index_uni(5), index_uni(-1), index_multi(std::vector<int>{-1, 2}),
      index_multi(std::vector<int>{5, 2}), index_min(5), index_max(0),
      index_min_max(-1, 3), index_min_max(2, 5), index_min_max(5, 1),
      index_min_max(3, -1));
  stan::math::for_each(
      [&indices, &m, &m_cl, &m_v_cl](const auto& ind1) {
        EXPECT_THROW(rvalue(m_cl, "", opencl_index(ind1)), std::out_of_range);
        EXPECT_THROW(rvalue(m_v_cl, "", opencl_index(ind1)), std::out_of_range);
      },
      indices);
}

#endif
