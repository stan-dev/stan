#include <vector>
#include <stan/model/indexing/index_list.hpp>
#include <stan/model/indexing/index.hpp>
#include <gtest/gtest.h>

using stan::model::nil_index_list;
using stan::model::cons_index_list;

using stan::model::index_uni;
using stan::model::index_multi;

TEST(MathIndexingIndexList, cons_index_list) {
  nil_index_list empty;  // ()

  index_uni idx_u(7);  

  cons_index_list<index_uni, nil_index_list>  cil(idx_u, empty);
  EXPECT_EQ(index_uni(7).n_, cil.head_.n_);

  std::vector<int> ns;
  ns.push_back(3);
  ns.push_back(17);
  index_multi idx_m(ns);

  cons_index_list<index_multi, 
                  cons_index_list<index_uni, 
                                  nil_index_list> > cil2(idx_m, cil); 
  EXPECT_EQ(2, cil2.head_.ns_.size());
  EXPECT_EQ(17,cil2.head_.ns_[1]);
  EXPECT_EQ(7, cil2.tail_.head_.n_);
}
