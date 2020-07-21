#include <gtest/gtest.h>
#include <stan/math/prim.hpp>
#include <stan/math/rev.hpp>
#include <stan/math/fwd.hpp>
#include <vector>
#include <random>

namespace stan {
namespace test {

template <typename Scal>
struct counterOp {
  int* counter_;
  counterOp(int* counter) { counter_ = counter; }
  const Scal& operator()(const Scal& a) const {
    (*counter_)++;
    return a;
  }
};

template <typename T>
auto recursive_sum(const T& a) {
  return math::sum(a);
}

template <typename T>
auto recursive_sum(const std::vector<T>& a) {
  scalar_type_t<T> res = recursive_sum(a[0]);
  for (int i = 0; i < a.size(); i++) {
    res += recursive_sum(a[i]);
  }
  return res;
}

template <typename T, require_integral_t<T>* = nullptr>
T make_arg() {
  return 1;
}
template <typename T, require_floating_point_t<T>* = nullptr>
T make_arg() {
  return 0.4;
}
template <typename T, require_var_t<T>* = nullptr>
T make_arg() {
  return 0.4;
}
template <typename T, require_fvar_t<T>* = nullptr>
T make_arg() {
  return {0.4, 0.5};
}
template <typename T, require_eigen_vector_t<T>* = nullptr>
T make_arg() {
  T res(1);
  res << make_arg<value_type_t<T>>();
  return res;
}
template <typename T, require_eigen_matrix_t<T>* = nullptr>
T make_arg() {
  T res(1, 1);
  res << make_arg<value_type_t<T>>();
  return res;
}
template <typename T, require_std_vector_t<T>* = nullptr>
T make_arg() {
  using V = value_type_t<T>;
  V tmp = make_arg<V>();
  T res;
  res.push_back(tmp);
  return res;
}
template <typename T, require_same_t<T, std::minstd_rand>* = nullptr>
T make_arg() {
  return std::minstd_rand(0);
}

template <typename T, require_arithmetic_t<T>* = nullptr>
void expect_eq(T a, T b, const char* msg) {
  EXPECT_EQ(a, b) << msg;
}

void expect_eq(math::var a, math::var b, const char* msg) {
  EXPECT_EQ(a.val(), b.val()) << msg;
}

template <typename T, require_arithmetic_t<T>* = nullptr>
void expect_eq(math::fvar<T> a, math::fvar<T> b, const char* msg) {
  expect_eq(a.val(), b.val(), msg);
  expect_eq(a.tangent(), b.tangent(), msg);
}

template <typename T1, typename T2, require_all_eigen_t<T1, T2>* = nullptr,
          require_vt_same<T1, T2>* = nullptr>
void expect_eq(const T1& a, const T2& b, const char* msg) {
  EXPECT_EQ(a.rows(), b.rows()) << msg;
  EXPECT_EQ(a.cols(), b.cols()) << msg;
  const auto& a_ref = math::to_ref(a);
  const auto& b_ref = math::to_ref(b);
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      expect_eq(a_ref(i, j), b_ref(i, j), msg);
    }
  }
}

template <typename T>
void expect_eq(const std::vector<T>& a, const std::vector<T>& b,
               const char* msg) {
  EXPECT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    expect_eq(a[i], b[i], msg);
  }
}

template <typename T, require_not_st_var<T>* = nullptr>
void expect_adj_eq(const T& a, const T& b, const char* msg) {}

void expect_adj_eq(math::var a, math::var b, const char* msg) {
  EXPECT_EQ(a.adj(), b.adj()) << msg;
}

template <typename T1, typename T2, require_all_eigen_t<T1, T2>* = nullptr,
          require_vt_same<T1, T2>* = nullptr>
void expect_adj_eq(const T1& a, const T2& b, const char* msg) {
  EXPECT_EQ(a.rows(), b.rows()) << msg;
  EXPECT_EQ(a.cols(), b.cols()) << msg;
  const auto& a_ref = math::to_ref(a);
  const auto& b_ref = math::to_ref(b);
  for (int i = 0; i < a.rows(); i++) {
    for (int j = 0; j < a.cols(); j++) {
      expect_adj_eq(a_ref(i, j), b_ref(i, j), msg);
    }
  }
}

template <typename T>
void expect_adj_eq(const std::vector<T>& a, const std::vector<T>& b,
                   const char* msg) {
  EXPECT_EQ(a.size(), b.size()) << msg;
  for (int i = 0; i < a.size(); i++) {
    expect_adj_eq(a[i], b[i], msg);
  }
}

#define TO_STRING_(x) #x
#define TO_STRING(x) TO_STRING_(x)
#define EXPECT_STAN_EQ(a, b) \
  stan::test::expect_eq(     \
      a, b, "Error in file: " __FILE__ ", on line: " TO_STRING(__LINE__))
#define EXPECT_STAN_ADJ_EQ(a, b) \
  stan::test::expect_adj_eq(     \
      a, b, "Error in file: " __FILE__ ", on line: " TO_STRING(__LINE__))

}  // namespace test

namespace math {

template <typename T>
auto bad_no_expressions(const Eigen::Matrix<T, -1, -1>& a) {
  return a;
}

template <typename T>
auto bad_multiple_evaluations(const T& a) {
  return a + a;
}

template <typename T>
auto bad_wrong_value(const T& a) {
  if (std::is_same<T, plain_type_t<T>>::value) {
    return a(0, 0);
  }
  return a(0, 0) + 1;
}

template <typename T>
auto bad_wrong_derivatives(const T& a) {
  operands_and_partials<T> ops(a);
  if (!is_constant<T>::value && std::is_same<T, plain_type_t<T>>::value) {
    ops.edge1_.partials_[0] = 1234;
  }
  return ops.build(0);
}

}  // namespace math
}  // namespace stan
