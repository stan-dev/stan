#include <vector>
#include <stan/meta/indexed_type.hpp>
#include <stan/meta/typelist.hpp>

#include <boost/type_traits/is_same.hpp> 
#include <gtest/gtest.h>

template <typename T, typename I>
void expect_eq_indexed() {
  EXPECT_TRUE(( boost::is_same<T,typename I::type>::value ));
}

TEST(MetaTypelist, testDouble) {
  using stan::meta::nil;
  using stan::meta::indexed_type;

  // double -- ()
  expect_eq_indexed<double,
                    indexed_type<double,nil> >();
}

TEST(MetaTypelist, testVec) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::indexed_type;
  using stan::meta::uni_index;
  using stan::meta::multi_index;
  using stan::meta::typelist;
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

TEST(MetaTypelist, testVecVec) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::indexed_type;
  using stan::meta::uni_index;
  using stan::meta::multi_index;
  using std::vector;

  using stan::meta::typelist;

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
