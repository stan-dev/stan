#include <vector>
#include <stan/math/indexing/rvalue.hpp>
#include <gtest/gtest.h>

using stan::math::nil_index_list;
using stan::math::cons_index_list;
using stan::math::index_uni;
using stan::math::index_multi;
using stan::math::index_omni;
using stan::math::index_min;
using stan::math::index_max;
using stan::math::index_min_max;
using stan::meta::nil;
using stan::meta::cons;

TEST(MathIndexing, rvalue_vector_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);

  std::vector<double> rx = rvalue(x, nil_index_list());
  EXPECT_EQ(2,rx.size());
  EXPECT_FLOAT_EQ(1.1, rx[0]);
  EXPECT_FLOAT_EQ(2.2, rx[1]);
}
TEST(MathIndexing, rvalue_vector_uni_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < x.size(); ++k)
    EXPECT_EQ(x[k],
              rvalue(x, cons_index_list<index_uni, nil_index_list>(index_uni(k), 
                                                                   nil_index_list())));
}
TEST(MathIndexing, rvalue_vector_multi_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);
  x.push_back(4.4);
  x.push_back(5.5);
  x.push_back(6.6);

  std::vector<int> idxs;
  idxs.push_back(0);
  idxs.push_back(3);
  idxs.push_back(4);
  
  std::vector<double> rx
    = rvalue(x, cons_index_list<index_multi, nil_index_list>(index_multi(idxs), 
                                                             nil_index_list()));
  EXPECT_FLOAT_EQ(3, rx.size());
  EXPECT_FLOAT_EQ(x[0], rx[0]);
  EXPECT_FLOAT_EQ(x[3], rx[1]);
  EXPECT_FLOAT_EQ(x[4], rx[2]);
}
TEST(MathIndexing, rvalue_vector_omni_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  std::vector<double> rx
    = rvalue(x, cons_index_list<index_omni, nil_index_list>(index_omni(), 
                                                             nil_index_list()));
  EXPECT_FLOAT_EQ(3, rx.size());
  for (int n = 0; n < rx.size(); ++n)
    EXPECT_FLOAT_EQ(x[n], rx[n]);
}
TEST(MathIndexing, rvalue_vector_min_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < 4; ++k) {
    std::vector<double> rx
      = rvalue(x, cons_index_list<index_min, nil_index_list>(index_min(k), 
                                                             nil_index_list()));
    EXPECT_FLOAT_EQ(3 - k, rx.size());
    for (int n = 0; n < rx.size(); ++n)
      EXPECT_FLOAT_EQ(x[n + k], rx[n]);
  }
}
TEST(MathIndexing, rvalue_vector_max_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < 3; ++k) {
    std::vector<double> rx
      = rvalue(x, cons_index_list<index_max, nil_index_list>(index_max(k), 
                                                             nil_index_list()));
    EXPECT_FLOAT_EQ(k + 1, rx.size());
    for (int n = 0; n < rx.size(); ++n)
      EXPECT_FLOAT_EQ(x[n], rx[n]);
  }
}
TEST(MathIndexing, rvalue_vector_min_max_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);
  x.push_back(4.4);

  for (int mn = 0; mn < 4; ++mn) {
    for (int mx = mn; mx < 4; ++mx) {
      std::vector<double> rx
        = rvalue(x, cons_index_list<index_min_max, nil_index_list>(index_min_max(mn,mx), 
                                                                   nil_index_list()));
      EXPECT_FLOAT_EQ(mx - mn + 1, rx.size());
      for (int n = mn; n <= mx; ++n)
        EXPECT_FLOAT_EQ(x[n], rx[n - mn]);
    }
  }
}












