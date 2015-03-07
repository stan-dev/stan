#include <stdexcept>
#include <stan/math/prim/mat/fun/sort_indices.hpp>
#include <stan/math/prim/mat/fun/sort_indices_asc.hpp>
#include <stan/math/prim/mat/fun/sort_indices_desc.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <gtest/gtest.h>

template <typename T>
void test_sort_indices_asc() {
  using stan::math::sort_indices_asc;
  typedef typename stan::math::index_type<T>::type idx_t;

  T c(1);
  c[0] = 1.7;
  std::vector<int> d = sort_indices_asc(c);
  EXPECT_EQ(c.size(),idx_t(d.size()));
  EXPECT_EQ(d.at(0), 1);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  std::vector<int> f = sort_indices_asc(e);
  EXPECT_EQ(e.size(), idx_t(f.size()));
  EXPECT_EQ(f.at(0), 2);
  EXPECT_EQ(f.at(1), 1);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  std::vector<int> h = sort_indices_asc(g);
  EXPECT_EQ(g.size(), idx_t(h.size()));
  EXPECT_EQ(h.at(0), 2);
  EXPECT_EQ(h.at(1), 1);
  EXPECT_EQ(h.at(2), 3);

  T z; 
  EXPECT_NO_THROW(sort_indices_asc(z));
  EXPECT_EQ(idx_t(0), z.size());
}

TEST(MathMatrix,sort_indices_asc) {
  using stan::math::sort_indices_asc;

  EXPECT_EQ(0U, sort_indices_asc(std::vector<int>(0)).size());

  test_sort_indices_asc<std::vector<double> >();
  test_sort_indices_asc<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_sort_indices_asc<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}


template <typename T>
void test_sort_indices_desc() {
  using stan::math::sort_indices_desc;
  typedef typename stan::math::index_type<T>::type idx_t;

  T c(1);
  c[0] = 1.7;
  std::vector<int> d = sort_indices_desc(c);
  EXPECT_EQ(c.size(), idx_t(d.size()));
  EXPECT_EQ(d.at(0), 1);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  std::vector<int> f = sort_indices_desc(e);
  EXPECT_EQ(e.size(), idx_t(f.size()));
  EXPECT_EQ(f.at(0), 1);
  EXPECT_EQ(f.at(1), 2);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  std::vector<int> h = sort_indices_desc(g);
  EXPECT_EQ(g.size(), idx_t(h.size()));
  EXPECT_EQ(h.at(0), 3);
  EXPECT_EQ(h.at(1), 1);
  EXPECT_EQ(h.at(2), 2);

  T z; 
  EXPECT_NO_THROW(sort_indices_desc(z));
  EXPECT_EQ(idx_t(0), z.size());

}

TEST(MathMatrix,sort_indices_desc) {
  using stan::math::sort_indices_desc;

  EXPECT_EQ(0U, sort_indices_desc(std::vector<int>(0)).size());

  test_sort_indices_desc<std::vector<double> >();
  test_sort_indices_desc<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_sort_indices_desc<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}
