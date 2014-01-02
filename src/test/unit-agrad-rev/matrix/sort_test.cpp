#include <stan/agrad/rev/matrix/sort.hpp>
#include <test/unit/agrad/util.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/rev.hpp>
#include <stan/math/matrix/sort.hpp>

void test_sort_asc(VEC val) {
  using stan::math::sort_asc;
  using stan::agrad::sort_asc;
  
  AVEC x;
  for(size_t i=0U; i<val.size(); i++)
    x.push_back(AVAR(val[i]));
  
  VEC val_sorted = sort_asc(val);
  AVEC x_sorted = sort_asc(x);
  
  for(size_t i=0U; i<val.size(); i++)
    EXPECT_EQ(val_sorted[i],x_sorted[i].val());

  for(size_t i=0U; i<val.size(); i++)
    for(size_t j=0U; j<val.size(); j++)
      if(val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i],x[j]);
      else
        EXPECT_FALSE(x_sorted[i]==x[j]);
}

void test_sort_desc(VEC val) {
  using stan::math::sort_desc;
  using stan::agrad::sort_desc;
  
  AVEC x;
  for(size_t i=0U; i<val.size(); i++)
    x.push_back(AVAR(val[i]));
  
  VEC val_sorted = sort_desc(val);
  AVEC x_sorted = sort_desc(x);
  
  for(size_t i=0U; i<val.size(); i++)
    EXPECT_EQ(val_sorted[i],x_sorted[i].val());

  for(size_t i=0U; i<val.size(); i++)
    for(size_t j=0U; j<val.size(); j++)
      if(val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i],x[j]);
      else
        EXPECT_FALSE(x_sorted[i]==x[j]);
}

template <typename T, int R, int C>
void test_sort_asc(Eigen::Matrix<T,R,C> val) {
  using stan::math::sort_asc;
  using stan::agrad::sort_asc;

  typedef Eigen::Matrix<AVAR,R,C> AVEC;
  typedef Eigen::Matrix<double,R,C> VEC;
  
  const size_t size = val.size();

  AVEC x(size);
  for(size_t i=0U; i<size; i++)
    x.data()[i] = AVAR(val[i]);
  
  VEC val_sorted = sort_asc(val);
  AVEC x_sorted = sort_asc(x);
  
  for(size_t i=0U; i<size; i++)
    EXPECT_EQ(val_sorted.data()[i],x_sorted.data()[i].val());

  for(size_t i=0U; i<size; i++)
    for(size_t j=0U; j<size; j++)
      if(val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i],x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i]==x.data()[j]);
}

template <typename T, int R, int C>
void test_sort_desc(Eigen::Matrix<T,R,C> val) {
  using stan::math::sort_desc;
  using stan::agrad::sort_desc;

  typedef Eigen::Matrix<AVAR,R,C> AVEC;
  typedef Eigen::Matrix<double,R,C> VEC;
  
  const size_t size = val.size();

  AVEC x(size);
  for(size_t i=0U; i<size; i++)
    x.data()[i] = AVAR(val[i]);
  
  VEC val_sorted = sort_desc(val);
  AVEC x_sorted = sort_desc(x);
  
  for(size_t i=0U; i<size; i++)
    EXPECT_EQ(val_sorted.data()[i],x_sorted.data()[i].val());

  for(size_t i=0U; i<size; i++)
    for(size_t j=0U; j<size; j++)
      if(val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i],x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i]==x.data()[j]);
}



TEST(AgradRev, sort) {
  VEC a;
  a.push_back(1); a.push_back(2); a.push_back(2); a.push_back(3);
  test_sort_asc(a);
  test_sort_desc(a);

  VEC b;
  b.push_back(1.1); b.push_back(2.2); ; b.push_back(33.1); b.push_back(-12.1); b.push_back(33.1);
  test_sort_asc(b);
  test_sort_desc(b);
  
  VEC c;
  c.push_back(1.1); c.push_back(-2); c.push_back(2.1); c.push_back(3); c.push_back(2.1);
  test_sort_asc(c);
  test_sort_desc(c);
  
  Eigen::RowVectorXd vec1(4);
  vec1 << 1, -33.1, 2.1, -33.1;
  test_sort_asc(vec1);
  test_sort_desc(vec1);

  Eigen::RowVectorXd vec2(5);
  vec2 << 1.1e-6, -2.3, 31.1, 1, -10.1;
  test_sort_asc(vec2);
  test_sort_desc(vec2);
  
  Eigen::VectorXd vec3(4);
  vec3 << -11.1, 2.2, -3.6, 2.2;
  test_sort_asc(vec3);
  test_sort_desc(vec3);
  
  Eigen::VectorXd vec4(3);
  vec4 << -10.1, 2.12, 3.102;
  test_sort_asc(vec4);
  test_sort_desc(vec4);

  Eigen::RowVectorXd vec5 = Eigen::RowVectorXd::Random(1,10);
  test_sort_asc(vec5);
  test_sort_desc(vec5);
  
  Eigen::VectorXd vec6 = Eigen::VectorXd::Random(20,1);
  test_sort_asc(vec6);
  test_sort_desc(vec6);
}

TEST(AgradRev, sort_no_thrown) {
  AVEC vec0;
  EXPECT_EQ(0U, vec0.size());
  EXPECT_NO_THROW(sort_asc(vec0));
  EXPECT_NO_THROW(sort_desc(vec0));
  
  Eigen::Matrix<AVAR,Eigen::Dynamic,1> vec1;
  EXPECT_EQ(0, vec1.size());
  EXPECT_NO_THROW(sort_asc(vec1));
  EXPECT_NO_THROW(sort_desc(vec1));

  Eigen::Matrix<AVAR,1,Eigen::Dynamic> vec2;
  EXPECT_EQ(0, vec2.size());
  EXPECT_NO_THROW(sort_asc(vec2));
  EXPECT_NO_THROW(sort_desc(vec2));
}
