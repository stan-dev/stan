#include <stan/model/indexing/rvalue_index_size.hpp>
#include <gtest/gtest.h>
#include <vector>

// error checking is during indexing, not during index construction
// so no tests here for out of bounds

TEST(modelIndexingRvalueIndexSize, multi) {
  using stan::model::index_multi;
  using stan::model::rvalue_index_size;

  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  
  index_multi idx(ns);
  EXPECT_EQ(2, rvalue_index_size(idx, 10));
}

TEST(modelIndexingRvalueIndexSize, omni) {
  using stan::model::index_omni;
  using stan::model::rvalue_index_size;

  index_omni idx;
  EXPECT_EQ(10, rvalue_index_size(idx, 10));
}

TEST(modelIndexingRvalueIndexSize, min) {
  using stan::model::index_min;
  using stan::model::rvalue_index_size;

  index_min idx(3);
  EXPECT_EQ(8, rvalue_index_size(idx, 10));
}

TEST(modelIndexingRvalueIndexSize, max) {
  using stan::model::index_max;
  using stan::model::rvalue_index_size;

  index_max idx(5);
  EXPECT_EQ(5, rvalue_index_size(idx, 10));
}

TEST(modelIndexingRvalueIndexSize, minMax) {
  using stan::model::index_min_max;
  using stan::model::rvalue_index_size;
  index_min_max mm(1, 3);
  EXPECT_EQ(3, rvalue_index_size(mm, 10));

  index_min_max mm2(3, 3);
  EXPECT_EQ(1, rvalue_index_size(mm2, 10));

  index_min_max mm3(3, 1);
  EXPECT_EQ(0, rvalue_index_size(mm3, 10));

  index_min_max mm4(1, 0);
  EXPECT_EQ(0, rvalue_index_size(mm3, 10));
}
