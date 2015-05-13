#include <vector>
#include <stan/model/indexing/indexed_type.hpp>
#include <stan/model/indexing/typelist.hpp>
#include <test/unit/meta/util.hpp>
#include <gtest/gtest.h>

TEST(ModelIndexedType, testDouble) {
  using stan::model::nil;
  using stan::model::indexed_type;

  // double -- ()
  expect_eq_indexed<double,
                    indexed_type<double,nil> >();
}

TEST(ModelIndexedType, testVec) {
  using stan::model::nil;
  using stan::model::cons;
  using stan::model::indexed_type;
  using stan::model::uni_index;
  using stan::model::multi_index;
  using stan::model::typelist;
  using std::vector;

  // double[] --  (uni)
  expect_eq_indexed<double,
                    indexed_type<vector<double>,
                                 typelist<uni_index>::type > >();

  // double[] -- (uni)
  expect_eq_indexed<double,
                    indexed_type<vector<double>,
                                 typelist<uni_index>::type> >();

  // double[] -- (multi)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<double>,
                                 typelist<multi_index>::type> >();

  // double[] -- nil
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<double>,
                                 typelist<>::type> >();
}

TEST(ModelIndexedType, testVecVec) {
  using stan::model::nil;
  using stan::model::cons;
  using stan::model::indexed_type;
  using stan::model::uni_index;
  using stan::model::multi_index;
  using std::vector;

  using stan::model::typelist;

  // double[,] -- ()
  expect_eq_indexed<vector<vector<double> >,
                    indexed_type<vector<vector<double> >,
                                 typelist<>::type> >();

  // double[,] -- (uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<vector<double> >,
                                 typelist<uni_index>::type > >();

  // double[,] -- (multi)
  expect_eq_indexed<vector<vector<double> >,
                    indexed_type<vector<vector<double> >,
                                 typelist<multi_index>::type > >();

  // double[,] -- (uni,uni)
  expect_eq_indexed<double,
                    indexed_type<vector<vector<double> >,
                                 typelist<uni_index,uni_index>::type > >();

  // double[,] -- (uni,multi)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<vector<double> >,
                                 typelist<uni_index,multi_index>::type > >();

  // double[,] -- (multi,uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<vector<double> >,
                                 typelist<multi_index,uni_index>::type > >();

  // double[,] -- (multi,multi)
  expect_eq_indexed<vector<vector<double> >,
                    indexed_type<vector<vector<double> >,
                                 typelist<multi_index,multi_index>::type > >();
}
