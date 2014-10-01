#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/indexed_type.hpp>
#include <stan/meta/typelist.hpp>
#include <stan/meta/matrix/indexed_type.hpp>

#include <test/unit/meta/util.hpp>

#include <gtest/gtest.h>

// **************************** VECTORS **********************************

TEST(MetaMatrixIndexedType, vector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;

  // vector -- (uni)
  expect_eq_indexed<double,
                    indexed_type<Matrix<double,Dynamic,1>, 
                                 typelist<uni_index>::type > >();

  // vector -- (multi)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<Matrix<double,Dynamic,1>, 
                                 typelist<multi_index>::type > >();
}
TEST(MetaMatrixIndexedType, vectorArray) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;

  // vector[] -- (uni)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<vector<Matrix<double,Dynamic,1> >,
                                 typelist<uni_index>::type > >();

  // vector[] -- (multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<Matrix<double,Dynamic,1> >,
                                 typelist<multi_index>::type > >();

  // vector[] -- (uni,uni)
  expect_eq_indexed<double,
                    indexed_type<vector<Matrix<double,Dynamic,1> >,
                                 typelist<uni_index, uni_index>::type > >();

  // vector[] -- (uni,multi)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<vector<Matrix<double,Dynamic,1> >,
                                 typelist<uni_index, multi_index>::type > >();


  // vector[] -- (multi,uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<Matrix<double,Dynamic,1> >,
                                 typelist<multi_index, uni_index>::type > >();

  // vector[] -- (multi,multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<Matrix<double,Dynamic,1> >,
                                 typelist<multi_index, multi_index>::type > >();
}

TEST(MetaMatrixIndexedType, vectorArray2) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;

  // vector[,] - (uni)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index>::type > >();

  // vector[,] - (multi)
  expect_eq_indexed<vector<vector<Matrix<double,Dynamic,1> > >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index>::type > >();

  // vector[,] - (uni,uni)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index,uni_index>::type > >();

  // vector[,] - (uni,multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index,multi_index>::type > >();

  // vector[,] - (multi,uni)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index,uni_index>::type > >();

  // vector[,] - (multi,multi)
  expect_eq_indexed<vector<vector<Matrix<double,Dynamic,1> > >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index,multi_index>::type > >();

  // vector[,] -- (uni,uni,uni)
  expect_eq_indexed<double,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index, uni_index, uni_index>::type > >();

  // vector[,] -- (uni,uni,multi)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index, uni_index, multi_index>::type > >();

  // vector[,] -- (uni,multi,uni)
  expect_eq_indexed<vector<double>, 
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index, multi_index, uni_index>::type > >();


  // vector[,] -- (uni,multi,multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<uni_index, multi_index, multi_index>::type > >();


  // vector[,] -- (multi,uni,uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index, uni_index, uni_index>::type > >();

  // vector[,] -- (multi,uni,multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index, uni_index, multi_index>::type > >();

  // vector[,] -- (multi,multi,uni)
  expect_eq_indexed<vector<vector<double> >, 
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index, multi_index, uni_index>::type > >();


  // vector[,] -- (multi,multi,multi)
  expect_eq_indexed<vector< vector<Matrix<double,Dynamic,1> > >,
                    indexed_type<vector<vector<Matrix<double,Dynamic,1> > >,
                                 typelist<multi_index, multi_index, multi_index>::type > >();
}


// **************************** ROW VECTORS **********************************

TEST(MetaMatrixIndexedType, rowVector) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;

  // rowVector -- (uni)
  expect_eq_indexed<double,
                    indexed_type<Matrix<double,1,Dynamic>, 
                                 typelist<uni_index>::type > >();

  // rowVector -- (multi)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<Matrix<double,1,Dynamic>, 
                                 typelist<multi_index>::type > >();
}
TEST(MetaMatrixIndexedType, rowVectorArray) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;

  // rowVector[] -- (uni)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<vector<Matrix<double,1,Dynamic> >,
                                 typelist<uni_index>::type > >();

  // rowVector[] -- (multi)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<Matrix<double,1,Dynamic> >,
                                 typelist<multi_index>::type > >();

  // rowVector[] -- (uni,uni)
  expect_eq_indexed<double,
                    indexed_type<vector<Matrix<double,1,Dynamic> >,
                                 typelist<uni_index, uni_index>::type > >();

  // rowVector[] -- (uni,multi)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<vector<Matrix<double,1,Dynamic> >,
                                 typelist<uni_index, multi_index>::type > >();


  // rowVector[] -- (multi,uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<Matrix<double,1,Dynamic> >,
                                 typelist<multi_index, uni_index>::type > >();

  // rowVector[] -- (multi,multi)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<Matrix<double,1,Dynamic> >,
                                 typelist<multi_index, multi_index>::type > >();
}

TEST(MetaMatrixIndexedType, rowVectorArray2) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;

  // rowVector[,] - (uni)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index>::type > >();

  // rowVector[,] - (multi)
  expect_eq_indexed<vector<vector<Matrix<double,1,Dynamic> > >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index>::type > >();

  // rowVector[,] - (uni,uni)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index,uni_index>::type > >();

  // rowVector[,] - (uni,multi)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index,multi_index>::type > >();

  // rowVector[,] - (multi,uni)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index,uni_index>::type > >();

  // rowVector[,] - (multi,multi)
  expect_eq_indexed<vector<vector<Matrix<double,1,Dynamic> > >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index,multi_index>::type > >();

  // rowVector[,] -- (uni,uni,uni)
  expect_eq_indexed<double,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index, uni_index, uni_index>::type > >();

  // rowVector[,] -- (uni,uni,multi)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index, uni_index, multi_index>::type > >();

  // rowVector[,] -- (uni,multi,uni)
  expect_eq_indexed<vector<double>, 
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index, multi_index, uni_index>::type > >();


  // rowVector[,] -- (uni,multi,multi)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<uni_index, multi_index, multi_index>::type > >();


  // rowVector[,] -- (multi,uni,uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index, uni_index, uni_index>::type > >();

  // rowVector[,] -- (multi,uni,multi)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index, uni_index, multi_index>::type > >();

  // rowVector[,] -- (multi,multi,uni)
  expect_eq_indexed<vector<vector<double> >, 
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index, multi_index, uni_index>::type > >();


  // rowVector[,] -- (multi,multi,multi)
  expect_eq_indexed<vector< vector<Matrix<double,1,Dynamic> > >,
                    indexed_type<vector<vector<Matrix<double,1,Dynamic> > >,
                                 typelist<multi_index, multi_index, multi_index>::type > >();
}


TEST(MetaMatrixIndexedType,matrix) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;
  
  // matrix -- ()
  expect_eq_indexed<Matrix<double,Dynamic,Dynamic>,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<>::type > >();

  // matrix -- (uni)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<uni_index>::type > >();

  // matrix -- (multi)
  expect_eq_indexed<Matrix<double,Dynamic,Dynamic>,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<multi_index>::type > >();

  // matrix -- (uni,uni)
  expect_eq_indexed<double,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<uni_index, uni_index>::type > >();

  // matrix -- (uni,multi)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<uni_index, multi_index>::type > >();

  // matrix -- (multi,uni)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<multi_index, uni_index>::type > >();
  // matrix -- (multi,multi)
  expect_eq_indexed<Matrix<double,Dynamic,Dynamic>,
                    indexed_type<Matrix<double,Dynamic,Dynamic>,
                                 typelist<multi_index, multi_index>::type > >();
}

TEST(MetaMatrixIndexedType,matrixArray) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::meta::indexed_type;
  using stan::meta::typelist;
  using stan::meta::uni_index;
  using stan::meta::multi_index;
  
  // matrix[] -- ()
  expect_eq_indexed<vector<Matrix<double,Dynamic,Dynamic> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<>::type > >();

  // matrix[] -- (uni)
  expect_eq_indexed<Matrix<double,Dynamic,Dynamic>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index>::type > >();

  // matrix[] -- (multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,Dynamic> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index>::type > >();

  // matrix[] -- (uni,uni)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index,uni_index>::type > >();

  // matrix[] -- (uni,multi)
  expect_eq_indexed<Matrix<double,Dynamic,Dynamic>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index,multi_index>::type > >();

  // matrix[] -- (multi,uni)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index,uni_index>::type > >();

  // matrix[] -- (multi,multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,Dynamic> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index,multi_index>::type > >();


  // matrix[] -- (uni,uni,uni)
  expect_eq_indexed<double,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index,uni_index,uni_index>::type > >();

  // matrix[] -- (uni,uni,multi)
  expect_eq_indexed<Matrix<double,1,Dynamic>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index,uni_index,multi_index>::type > >();

  // matrix[] -- (uni,multi,uni)
  expect_eq_indexed<Matrix<double,Dynamic,1>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index,multi_index,uni_index>::type > >();

  // matrix[] -- (uni,multi,multi)
  expect_eq_indexed<Matrix<double,Dynamic,Dynamic>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<uni_index,multi_index,multi_index>::type > >();

  // matrix[] -- (multi,uni,uni)
  expect_eq_indexed<vector<double>,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index,uni_index,uni_index>::type > >();

  // matrix[] -- (multi,uni,multi)
  expect_eq_indexed<vector<Matrix<double,1,Dynamic> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index,uni_index,multi_index>::type > >();

  // matrix[] -- (multi,multi,uni)
  expect_eq_indexed<vector<Matrix<double,Dynamic,1> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index,multi_index,uni_index>::type > >();

  // matrix[] -- (multi,multi,multi)
  expect_eq_indexed<vector<Matrix<double,Dynamic,Dynamic> >,
                    indexed_type<vector<Matrix<double,Dynamic,Dynamic> >,
                                 typelist<multi_index,multi_index,multi_index>::type > >();

}




