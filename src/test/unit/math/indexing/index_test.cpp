#include <vector>
#include <boost/type_traits/is_same.hpp> 
#include <stan/math/indexing/index.hpp>
#include <gtest/gtest.h>

using stan::math::index_uni;
using stan::math::index_multi;
using stan::math::index_omni;
using stan::math::index_min;
using stan::math::index_max;
using stan::math::index_min_max;

using stan::meta::uni_index;
using stan::meta::multi_index;

TEST(MathIndexingIndex, index_uni) {
  EXPECT_TRUE(( boost::is_same<uni_index, index_uni::index_type>::value ));

  index_uni idx(17);
  EXPECT_EQ(17, idx.n_);
}

TEST(MathIndexingIndex, index_multi) {
  EXPECT_TRUE(( boost::is_same<multi_index, index_multi::index_type>::value ));

  std::vector<int> ns;
  ns.push_back(3);
  ns.push_back(23);

  index_multi idx(ns);
  EXPECT_EQ(2, idx.ns_.size());
  for (int i = 0; i < ns.size(); ++i)
    EXPECT_EQ(ns[i], idx.ns_[i]);
}

TEST(MathIndexingIndex, index_omni) {
  EXPECT_TRUE(( boost::is_same<multi_index, index_omni::index_type>::value ));

  index_omni idx;
  (void) idx;  // just to silence compiler griping about idx being unused
}

TEST(MathIndexingIndex, index_min) {
  EXPECT_TRUE(( boost::is_same<multi_index, index_min::index_type>::value ));

  index_min idx(3);
  EXPECT_EQ(3, idx.min_);
}

TEST(MathIndexingIndex, index_max) {
  EXPECT_TRUE(( boost::is_same<multi_index, index_max::index_type>::value ));

  index_max idx(912);
  EXPECT_EQ(912, idx.max_);
}

TEST(MathIndexingIndex, index_min_max) {
  EXPECT_TRUE(( boost::is_same<multi_index, index_min_max::index_type>::value ));

  index_min_max idx(401,912);
  EXPECT_EQ(401, idx.min_);
  EXPECT_EQ(912, idx.max_);
}

