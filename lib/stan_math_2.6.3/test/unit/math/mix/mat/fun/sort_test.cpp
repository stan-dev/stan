#include <gtest/gtest.h>

#include <stan/math/fwd/mat/fun/sort_asc.hpp>
#include <stan/math/fwd/mat/fun/sort_desc.hpp>
#include <stan/math/prim/mat/fun/sort.hpp>

#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

typedef std::vector<double> VEC;
using stan::math::fvar;
using stan::math::var;

void test_sort_asc2(std::vector<double> val) {
  using stan::math::sort_asc;
  using stan::math::sort_asc;
  
  std::vector<fvar<var> > x;
  for(size_t i=0U; i<val.size(); i++)
    x.push_back(fvar<var>(val[i]));
  
  std::vector<double> val_sorted = sort_asc(val);
  std::vector<fvar<var> > x_sorted = sort_asc(x);
  
  for(size_t i=0U; i<val.size(); i++)
    EXPECT_EQ(val_sorted[i],x_sorted[i].val().val());

  for(size_t i=0U; i<val.size(); i++)
    for(size_t j=0U; j<val.size(); j++)
      if(val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i],x[j]);
      else
        EXPECT_FALSE(x_sorted[i]==x[j]);
}
void test_sort_asc4(std::vector<double> val) {
  using stan::math::sort_asc;
  using stan::math::sort_asc;
  
  std::vector<fvar<fvar<var> > > x;
  for(size_t i=0U; i<val.size(); i++)
    x.push_back(fvar<fvar<var> >(val[i]));
  
  std::vector<double> val_sorted = sort_asc(val);
  std::vector<fvar<fvar<var> > > x_sorted = sort_asc(x);
  
  for(size_t i=0U; i<val.size(); i++)
    EXPECT_EQ(val_sorted[i],x_sorted[i].val().val().val());

  for(size_t i=0U; i<val.size(); i++)
    for(size_t j=0U; j<val.size(); j++)
      if(val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i],x[j]);
      else
        EXPECT_FALSE(x_sorted[i]==x[j]);
}
void test_sort_desc2(VEC val) {
  using stan::math::sort_desc;
  using stan::math::sort_desc;
  
  std::vector<fvar<var> > x;
  for(size_t i=0U; i<val.size(); i++)
    x.push_back(fvar<var>(val[i]));
  
  VEC val_sorted = sort_desc(val);
  std::vector<fvar<var> > x_sorted = sort_desc(x);
  
  for(size_t i=0U; i<val.size(); i++)
    EXPECT_EQ(val_sorted[i],x_sorted[i].val().val());

  for(size_t i=0U; i<val.size(); i++)
    for(size_t j=0U; j<val.size(); j++)
      if(val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i],x[j]);
      else
        EXPECT_FALSE(x_sorted[i]==x[j]);
}
void test_sort_desc4(VEC val) {
  using stan::math::sort_desc;
  using stan::math::sort_desc;
  
  std::vector<fvar<fvar<var> > > x;
  for(size_t i=0U; i<val.size(); i++)
    x.push_back(fvar<fvar<var> >(val[i]));
  
  VEC val_sorted = sort_desc(val);
  std::vector<fvar<fvar<var> > > x_sorted = sort_desc(x);
  
  for(size_t i=0U; i<val.size(); i++)
    EXPECT_EQ(val_sorted[i],x_sorted[i].val().val().val());

  for(size_t i=0U; i<val.size(); i++)
    for(size_t j=0U; j<val.size(); j++)
      if(val_sorted[i] == val[j])
        EXPECT_EQ(x_sorted[i],x[j]);
      else
        EXPECT_FALSE(x_sorted[i]==x[j]);
}
template <typename T, int R, int C>
void test_sort_asc2(Eigen::Matrix<T,R,C> val) {
  using stan::math::sort_asc;
  using stan::math::sort_asc;

  typedef Eigen::Matrix<fvar<var>,R,C> AVEC;
  typedef Eigen::Matrix<double,R,C> VEC;
  
  const size_t size = val.size();

  AVEC x(size);
  for(size_t i=0U; i<size; i++)
    x.data()[i] = fvar<var>(val[i]);
  
  VEC val_sorted = sort_asc(val);
  AVEC x_sorted = sort_asc(x);
  
  for(size_t i=0U; i<size; i++)
    EXPECT_EQ(val_sorted.data()[i],x_sorted.data()[i].val().val());

  for(size_t i=0U; i<size; i++)
    for(size_t j=0U; j<size; j++)
      if(val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i],x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i]==x.data()[j]);
}
template <typename T, int R, int C>
void test_sort_asc4(Eigen::Matrix<T,R,C> val) {
  using stan::math::sort_asc;
  using stan::math::sort_asc;

  typedef Eigen::Matrix<fvar<fvar<var> >,R,C> AVEC;
  typedef Eigen::Matrix<double,R,C> VEC;
  
  const size_t size = val.size();

  AVEC x(size);
  for(size_t i=0U; i<size; i++)
    x.data()[i] = fvar<fvar<var> >(val[i]);
  
  VEC val_sorted = sort_asc(val);
  AVEC x_sorted = sort_asc(x);
  
  for(size_t i=0U; i<size; i++)
    EXPECT_EQ(val_sorted.data()[i],x_sorted.data()[i].val().val().val());

  for(size_t i=0U; i<size; i++)
    for(size_t j=0U; j<size; j++)
      if(val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i],x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i]==x.data()[j]);
}
template <typename T, int R, int C>
void test_sort_desc2(Eigen::Matrix<T,R,C> val) {
  using stan::math::sort_desc;
  using stan::math::sort_desc;

  typedef Eigen::Matrix<fvar<var>,R,C> AVEC;
  typedef Eigen::Matrix<double,R,C> VEC;
  
  const size_t size = val.size();

  AVEC x(size);
  for(size_t i=0U; i<size; i++)
    x.data()[i] = fvar<var>(val[i]);
  
  VEC val_sorted = sort_desc(val);
  AVEC x_sorted = sort_desc(x);
  
  for(size_t i=0U; i<size; i++)
    EXPECT_EQ(val_sorted.data()[i],x_sorted.data()[i].val().val());

  for(size_t i=0U; i<size; i++)
    for(size_t j=0U; j<size; j++)
      if(val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i],x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i]==x.data()[j]);
}
template <typename T, int R, int C>
void test_sort_desc4(Eigen::Matrix<T,R,C> val) {
  using stan::math::sort_desc;
  using stan::math::sort_desc;

  typedef Eigen::Matrix<fvar<fvar<var> >,R,C> AVEC;
  typedef Eigen::Matrix<double,R,C> VEC;
  
  const size_t size = val.size();

  AVEC x(size);
  for(size_t i=0U; i<size; i++)
    x.data()[i] = fvar<fvar<var> >(val[i]);
  
  VEC val_sorted = sort_desc(val);
  AVEC x_sorted = sort_desc(x);
  
  for(size_t i=0U; i<size; i++)
    EXPECT_EQ(val_sorted.data()[i],x_sorted.data()[i].val().val().val());

  for(size_t i=0U; i<size; i++)
    for(size_t j=0U; j<size; j++)
      if(val_sorted.data()[i] == val.data()[j])
        EXPECT_EQ(x_sorted.data()[i],x.data()[j]);
      else
        EXPECT_FALSE(x_sorted.data()[i]==x.data()[j]);
}

TEST(AgradMixSort, var) {
  VEC a;
  a.push_back(1); a.push_back(2); a.push_back(2); a.push_back(3);
  test_sort_asc2(a);
  test_sort_desc2(a);

  VEC b;
  b.push_back(1.1); b.push_back(2.2); ; b.push_back(33.1); b.push_back(-12.1); b.push_back(33.1);
  test_sort_asc2(b);
  test_sort_desc2(b);
  
  VEC c;
  c.push_back(1.1); c.push_back(-2); c.push_back(2.1); c.push_back(3); c.push_back(2.1);
  test_sort_asc2(c);
  test_sort_desc2(c);
  
  Eigen::RowVectorXd vec1(4);
  vec1 << 1, -33.1, 2.1, -33.1;
  test_sort_asc2(vec1);
  test_sort_desc2(vec1);

  Eigen::RowVectorXd vec2(5);
  vec2 << 1.1e-6, -2.3, 31.1, 1, -10.1;
  test_sort_asc2(vec2);
  test_sort_desc2(vec2);
  
  Eigen::VectorXd vec3(4);
  vec3 << -11.1, 2.2, -3.6, 2.2;
  test_sort_asc2(vec3);
  test_sort_desc2(vec3);
  
  Eigen::VectorXd vec4(3);
  vec4 << -10.1, 2.12, 3.102;
  test_sort_asc2(vec4);
  test_sort_desc2(vec4);

  Eigen::RowVectorXd vec5 = Eigen::RowVectorXd::Random(1,10);
  test_sort_asc2(vec5);
  test_sort_desc2(vec5);
  
  Eigen::VectorXd vec6 = Eigen::VectorXd::Random(20,1);
  test_sort_asc2(vec6);
  test_sort_desc2(vec6);

}

TEST(AgradMixSort, fv_no_thrown) {
  Eigen::Matrix<fvar<var>,Eigen::Dynamic,1> vec1;
  EXPECT_EQ(0, vec1.size());
  EXPECT_NO_THROW(sort_asc(vec1));
  EXPECT_NO_THROW(sort_desc(vec1));

  Eigen::Matrix<fvar<var>,1,Eigen::Dynamic> vec2;
  EXPECT_EQ(0, vec2.size());
  EXPECT_NO_THROW(sort_asc(vec2));
  EXPECT_NO_THROW(sort_desc(vec2));
}
TEST(AgradMixSort, ffv_sort) {
  VEC a;
  a.push_back(1); a.push_back(2); a.push_back(2); a.push_back(3);
  test_sort_asc4(a);
  test_sort_desc4(a);

  VEC b;
  b.push_back(1.1); b.push_back(2.2); ; b.push_back(33.1); b.push_back(-12.1); b.push_back(33.1);
  test_sort_asc4(b);
  test_sort_desc4(b);
  
  VEC c;
  c.push_back(1.1); c.push_back(-2); c.push_back(2.1); c.push_back(3); c.push_back(2.1);
  test_sort_asc4(c);
  test_sort_desc4(c);
  
  Eigen::RowVectorXd vec1(4);
  vec1 << 1, -33.1, 2.1, -33.1;
  test_sort_asc4(vec1);
  test_sort_desc4(vec1);

  Eigen::RowVectorXd vec2(5);
  vec2 << 1.1e-6, -2.3, 31.1, 1, -10.1;
  test_sort_asc4(vec2);
  test_sort_desc4(vec2);
  
  Eigen::VectorXd vec3(4);
  vec3 << -11.1, 2.2, -3.6, 2.2;
  test_sort_asc4(vec3);
  test_sort_desc4(vec3);
  
  Eigen::VectorXd vec4(3);
  vec4 << -10.1, 2.12, 3.102;
  test_sort_asc4(vec4);
  test_sort_desc4(vec4);

  Eigen::RowVectorXd vec5 = Eigen::RowVectorXd::Random(1,10);
  test_sort_asc4(vec5);
  test_sort_desc4(vec5);
  
  Eigen::VectorXd vec6 = Eigen::VectorXd::Random(20,1);
  test_sort_asc4(vec6);
  test_sort_desc4(vec6);

}

TEST(AgradMixSort, ffv_no_thrown) {
  Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1> vec1;
  EXPECT_EQ(0, vec1.size());
  EXPECT_NO_THROW(sort_asc(vec1));
  EXPECT_NO_THROW(sort_desc(vec1));

  Eigen::Matrix<fvar<fvar<var> >,1,Eigen::Dynamic> vec2;
  EXPECT_EQ(0, vec2.size());
  EXPECT_NO_THROW(sort_asc(vec2));
  EXPECT_NO_THROW(sort_desc(vec2));
}
