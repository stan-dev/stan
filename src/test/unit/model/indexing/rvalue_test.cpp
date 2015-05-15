#include <iostream>
#include <vector>
#include <stan/model/indexing/rvalue.hpp>
#include <gtest/gtest.h>

using stan::model::nil_index_list;
using stan::model::cons_index_list;
using stan::model::index_uni;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_min;
using stan::model::index_max;
using stan::model::index_min_max;

TEST(ModelIndexing, rvalue_vector_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);

  std::vector<double> rx = rvalue(x, nil_index_list());
  EXPECT_EQ(2,rx.size());
  EXPECT_FLOAT_EQ(1.1, rx[0]);
  EXPECT_FLOAT_EQ(2.2, rx[1]);
}
TEST(ModelIndexing, rvalue_vector_uni_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (size_t k = 0; k < x.size(); ++k)
    EXPECT_EQ(x[k], rvalue(x, index_list(index_uni(k))));
}
TEST(ModelIndexing, rvalue_vector_multi_nil) {
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
  
  std::vector<double> rx = rvalue(x, index_list(index_multi(idxs)));
  EXPECT_FLOAT_EQ(3, rx.size());
  EXPECT_FLOAT_EQ(x[0], rx[0]);
  EXPECT_FLOAT_EQ(x[3], rx[1]);
  EXPECT_FLOAT_EQ(x[4], rx[2]);
}
TEST(ModelIndexing, rvalue_vector_omni_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  std::vector<double> rx = rvalue(x, index_list(index_omni()));
  EXPECT_FLOAT_EQ(3, rx.size());
  for (size_t n = 0; n < rx.size(); ++n)
    EXPECT_FLOAT_EQ(x[n], rx[n]);
}
TEST(ModelIndexing, rvalue_vector_min_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < 4; ++k) {
    std::vector<double> rx = rvalue(x, index_list(index_min(k)));
    EXPECT_FLOAT_EQ(3 - k, rx.size());
    for (size_t n = 0; n < rx.size(); ++n)
      EXPECT_FLOAT_EQ(x[n + k], rx[n]);
  }
}
TEST(ModelIndexing, rvalue_vector_max_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);

  for (int k = 0; k < 3; ++k) {
    std::vector<double> rx = rvalue(x, index_list(index_max(k)));
    EXPECT_FLOAT_EQ(k + 1, rx.size());
    for (size_t n = 0; n < rx.size(); ++n)
      EXPECT_FLOAT_EQ(x[n], rx[n]);
  }
}
TEST(ModelIndexing, rvalue_vector_min_max_nil) {
  std::vector<double> x;
  x.push_back(1.1);
  x.push_back(2.2);
  x.push_back(3.3);
  x.push_back(4.4);

  for (int mn = 0; mn < 4; ++mn) {
    for (int mx = mn; mx < 4; ++mx) {
      std::vector<double> rx = rvalue(x, index_list(index_min_max(mn, mx)));
      EXPECT_FLOAT_EQ(mx - mn + 1, rx.size());
      for (int n = mn; n <= mx; ++n)
        EXPECT_FLOAT_EQ(x[n], rx[n - mn]);
    }
  }
}

TEST(ModelIndexing, rvalue_doubless_uni_uni) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  
  for (int m = 0; m < 2; ++m)
    for (int n = 0; n < 2; ++n)
      EXPECT_FLOAT_EQ(m + n / 10.0, 
                      rvalue(x, index_list(index_uni(m), index_uni(n))));
}

TEST(ModelIndexing, rvalue_doubless_uni_multi) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);
  x0.push_back(0.2);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);
  x1.push_back(1.2);

  vector<double> x2;
  x2.push_back(2.0);
  x2.push_back(2.1);
  x2.push_back(2.2);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);

  vector<double> y = rvalue(x, index_list(index_uni(0), index_min(1)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.1, y[0]);
  EXPECT_FLOAT_EQ(0.2, y[1]);

  y = rvalue(x, index_list(index_uni(1), index_max(1)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y[0]);
  EXPECT_FLOAT_EQ(1.1, y[1]);

  y = rvalue(x, index_list(index_uni(1), index_min_max(1,2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);
  EXPECT_FLOAT_EQ(1.2, y[1]);

  y = rvalue(x, index_list(index_uni(1), index_min_max(1,1)));
  EXPECT_EQ(1, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);

  y = rvalue(x, index_list(index_uni(2), index_omni()));
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(2.0, y[0]);
  EXPECT_FLOAT_EQ(2.1, y[1]);
  EXPECT_FLOAT_EQ(2.2, y[2]);

  vector<int> ns;
  ns.push_back(2);
  ns.push_back(0);
  y = rvalue(x, index_list(index_uni(0), index_multi(ns)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.2, y[0]);
  EXPECT_FLOAT_EQ(0.0, y[1]);
}


TEST(ModelIndexing, rvalue_doubless_multi_uni) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);
  x0.push_back(0.2);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);
  x1.push_back(1.2);

  vector<double> x2;
  x2.push_back(2.0);
  x2.push_back(2.1);
  x2.push_back(2.2);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);

  vector<double> y = rvalue(x, index_list(index_min(1), index_uni(0)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.0, y[0]);
  EXPECT_FLOAT_EQ(2.0, y[1]);

  y = rvalue(x, index_list(index_max(1), index_uni(2)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(0.2, y[0]);
  EXPECT_FLOAT_EQ(1.2, y[1]);

  y = rvalue(x, index_list(index_min_max(1,2), index_uni(1)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);
  EXPECT_FLOAT_EQ(2.1, y[1]);

  y = rvalue(x, index_list(index_min_max(1,1), index_uni(1)));
  EXPECT_EQ(1, y.size());
  EXPECT_FLOAT_EQ(1.1, y[0]);

  y = rvalue(x, index_list(index_omni(), index_uni(2)));
  EXPECT_EQ(3, y.size());
  EXPECT_FLOAT_EQ(0.2, y[0]);
  EXPECT_FLOAT_EQ(1.2, y[1]);
  EXPECT_FLOAT_EQ(2.2, y[2]);

  vector<int> ns;
  ns.push_back(2);
  ns.push_back(0);
  y = rvalue(x, index_list(index_multi(ns), index_uni(0)));
  EXPECT_EQ(2, y.size());
  EXPECT_FLOAT_EQ(2.0, y[0]);
  EXPECT_FLOAT_EQ(0.0, y[1]);
}

TEST(ModelIndexing, rvalue_doubless_multi_multi) {
  using std::vector;
  
  vector<double> x0;
  x0.push_back(0.0);
  x0.push_back(0.1);
  x0.push_back(0.2);

  vector<double> x1;
  x1.push_back(1.0);
  x1.push_back(1.1);
  x1.push_back(1.2);

  vector<double> x2;
  x2.push_back(2.0);
  x2.push_back(2.1);
  x2.push_back(2.2);

  vector<vector<double> > x;
  x.push_back(x0);
  x.push_back(x1);
  x.push_back(x2);

  vector<vector<double> > y = rvalue(x, index_list(index_max(1), index_min(1)));
  EXPECT_EQ(2, y.size());
  EXPECT_EQ(2, y[0].size());
  EXPECT_EQ(2, y[1].size());
  EXPECT_FLOAT_EQ(0.1, y[0][0]);
  EXPECT_FLOAT_EQ(0.2, y[0][1]);
  EXPECT_FLOAT_EQ(1.1, y[1][0]);
  EXPECT_FLOAT_EQ(1.2, y[1][1]);
}
