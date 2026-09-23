#include <stan/model/indexing.hpp>
#include <stan/math/rev/fun/sum.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using stan::model::assign;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_uni;
using stan::model::rvalue;

namespace {

// An empty read contributes no adjoints; an empty assignment leaves the
// original values and their derivatives unchanged.
template <typename T, typename... Idxs>
void check_empty_slice(T& x, int rows, int cols, const Idxs&... idxs) {
  auto selected = rvalue(x, "x", idxs...);
  EXPECT_EQ(rows, selected.rows());
  EXPECT_EQ(cols, selected.cols());
  if constexpr (stan::is_var<stan::scalar_type_t<T>>::value) {
    stan::math::set_zero_all_adjoints();
    stan::math::sum(selected).grad();
    for (int i = 0; i < x.size(); ++i) {
      if constexpr (stan::is_var_matrix<T>::value) {
        EXPECT_EQ(0, x.adj().coeff(i));
      } else {
        EXPECT_EQ(0, x.coeff(i).adj());
      }
    }
  }
  using plain_t = stan::plain_type_t<decltype(selected)>;
  plain_t empty(selected);
  EXPECT_NO_THROW(assign(x, empty, "x", idxs...));
  EXPECT_TRUE(stan::math::value_of(x).isOnes());
  if constexpr (stan::is_var<stan::scalar_type_t<T>>::value) {
    stan::math::set_zero_all_adjoints();
    stan::math::sum(x).grad();
    for (int i = 0; i < x.size(); ++i) {
      if constexpr (stan::is_var_matrix<T>::value) {
        EXPECT_EQ(1, x.adj().coeff(i));
      } else {
        EXPECT_EQ(1, x.coeff(i).adj());
      }
    }
  }
}

template <typename T>
void check_vector_empty_ranges() {
  using values_t
      = Eigen::Matrix<double, T::RowsAtCompileTime, T::ColsAtCompileTime>;
  for (int size : {0, 3}) {
    T x(values_t::Ones(size));
    const int rows = T::RowsAtCompileTime == 1 ? 1 : 0;
    const int cols = T::RowsAtCompileTime == 1 ? 0 : 1;
    for (int min : {size + 1, size + 4, std::numeric_limits<int>::max()}) {
      check_empty_slice(x, rows, cols, index_min(min));
      EXPECT_THROW(assign(x, values_t::Ones(1), "x", index_min(min)),
                   std::invalid_argument);
    }
    for (int max : {0, -3, std::numeric_limits<int>::min()}) {
      check_empty_slice(x, rows, cols, index_max(max));
      EXPECT_THROW(assign(x, values_t::Ones(1), "x", index_max(max)),
                   std::invalid_argument);
    }
    EXPECT_THROW(rvalue(x, "x", index_min(0)), std::out_of_range);
    EXPECT_THROW(rvalue(x, "x", index_max(size + 1)), std::out_of_range);
  }
}

template <typename T>
void check_matrix_empty_ranges() {
  for (int rows : {0, 3}) {
    for (int cols : {0, 4}) {
      T x(Eigen::MatrixXd::Ones(rows, cols));
      for (int min : {rows + 1, rows + 4, std::numeric_limits<int>::max()}) {
        check_empty_slice(x, 0, cols, index_min(min));
        check_empty_slice(x, 0, cols, index_min(min), index_omni());
        EXPECT_THROW(assign(x, Eigen::MatrixXd(1, cols), "x", index_min(min)),
                     std::invalid_argument);
        EXPECT_THROW(
            assign(x, Eigen::MatrixXd(0, cols + 1), "x", index_min(min)),
            std::invalid_argument);
        if (cols > 0) {
          check_empty_slice(x, 0, 1, index_min(min), index_uni(1));
          check_empty_slice(x, 0, 2, index_min(min),
                            index_multi(std::vector<int>{1, 2}));
        }
      }
      for (int min : {cols + 1, cols + 4, std::numeric_limits<int>::max()}) {
        check_empty_slice(x, rows, 0, index_omni(), index_min(min));
        check_empty_slice(x, 0, 0, index_min(rows + 4), index_min(min));
        check_empty_slice(x, 0, 0, index_max(0), index_min(min));
        EXPECT_THROW(assign(x, Eigen::MatrixXd(rows, 1), "x", index_omni(),
                            index_min(min)),
                     std::invalid_argument);
        if (rows > 0) {
          check_empty_slice(x, 1, 0, index_uni(1), index_min(min));
          check_empty_slice(x, 2, 0, index_multi(std::vector<int>{1, 2}),
                            index_min(min));
        }
      }
      for (int max : {0, -3, std::numeric_limits<int>::min()}) {
        check_empty_slice(x, 0, cols, index_max(max));
        check_empty_slice(x, rows, 0, index_omni(), index_max(max));
        check_empty_slice(x, 0, 0, index_min(rows + 4), index_max(max));
        EXPECT_THROW(assign(x, Eigen::MatrixXd(1, cols), "x", index_max(max)),
                     std::invalid_argument);
        EXPECT_THROW(assign(x, Eigen::MatrixXd(rows, 1), "x", index_omni(),
                            index_max(max)),
                     std::invalid_argument);
      }
      EXPECT_THROW(rvalue(x, "x", index_min(0)), std::out_of_range);
      EXPECT_THROW(rvalue(x, "x", index_omni(), index_min(0)),
                   std::out_of_range);
      EXPECT_THROW(rvalue(x, "x", index_max(rows + 1)), std::out_of_range);
      EXPECT_THROW(rvalue(x, "x", index_omni(), index_max(cols + 1)),
                   std::out_of_range);
    }
  }
}

}  // namespace

TEST(ModelIndexingEmptyRange, eigen) {
  check_vector_empty_ranges<Eigen::VectorXd>();
  check_vector_empty_ranges<Eigen::RowVectorXd>();
  check_matrix_empty_ranges<Eigen::MatrixXd>();
}

TEST(ModelIndexingEmptyRange, eigenVar) {
  stan::math::nested_rev_autodiff nested;
  check_vector_empty_ranges<Eigen::Matrix<stan::math::var, -1, 1>>();
  check_vector_empty_ranges<Eigen::Matrix<stan::math::var, 1, -1>>();
  check_matrix_empty_ranges<Eigen::Matrix<stan::math::var, -1, -1>>();
}

TEST(ModelIndexingEmptyRange, varmat) {
  stan::math::nested_rev_autodiff nested;
  check_vector_empty_ranges<stan::math::var_value<Eigen::VectorXd>>();
  check_vector_empty_ranges<stan::math::var_value<Eigen::RowVectorXd>>();
  check_matrix_empty_ranges<stan::math::var_value<Eigen::MatrixXd>>();
}

TEST(ModelIndexingEmptyRange, arrays) {
  for (int size : {0, 3}) {
    std::vector<double> x(size, 1);
    std::vector<std::vector<double>> xx(size, x);
    const auto check = [&](auto idx) {
      EXPECT_TRUE(rvalue(x, "x", idx).empty());
      EXPECT_TRUE(rvalue(xx, "xx", idx, index_omni()).empty());
      EXPECT_NO_THROW(assign(x, std::vector<double>{}, "x", idx));
      EXPECT_NO_THROW(assign(xx, std::vector<std::vector<double>>{}, "xx", idx,
                             index_omni()));
      EXPECT_THROW(assign(x, std::vector<double>{2}, "x", idx),
                   std::invalid_argument);
      if (size > 0) {
        auto inner_empty = rvalue(xx, "xx", index_omni(), idx);
        EXPECT_EQ(size, inner_empty.size());
        for (const auto& inner : inner_empty) {
          EXPECT_TRUE(inner.empty());
        }
        EXPECT_NO_THROW(assign(xx, inner_empty, "xx", index_omni(), idx));
      }
      EXPECT_EQ(std::vector<double>(size, 1), x);
      EXPECT_EQ(std::vector<std::vector<double>>(size, x), xx);
    };
    for (int min : {size + 1, size + 4, std::numeric_limits<int>::max()}) {
      check(index_min(min));
    }
    for (int max : {0, -3, std::numeric_limits<int>::min()}) {
      check(index_max(max));
    }
  }
}
