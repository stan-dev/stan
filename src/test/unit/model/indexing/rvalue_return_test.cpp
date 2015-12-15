#include <typeinfo>
#include <boost/type_traits/is_same.hpp> 
#include <stan/model/indexing/rvalue_return.hpp>
#include <gtest/gtest.h>

using stan::model::nil_index_list;
using stan::model::cons_index_list;
using stan::model::index_uni;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::index_min;
using stan::model::index_max;
using stan::model::index_min_max;
using stan::model::rvalue_return;

typedef std::vector<double> doubles;
typedef std::vector<doubles> doubless;

typedef Eigen::MatrixXd mat;
typedef std::vector<mat> mats;
typedef std::vector<mats> matss;

typedef Eigen::VectorXd vec;
typedef std::vector<vec> vecs;
typedef std::vector<vecs> vecss;

typedef Eigen::RowVectorXd rowvec;
typedef std::vector<rowvec> rowvecs;
typedef std::vector<rowvecs> rowvecss;

typedef cons_index_list<index_uni, nil_index_list> uni;
typedef cons_index_list<index_multi, nil_index_list> multi;

typedef cons_index_list<index_uni, uni> uni_uni;
typedef cons_index_list<index_uni, multi> uni_multi;
typedef cons_index_list<index_multi, uni> multi_uni;
typedef cons_index_list<index_multi, multi> multi_multi;

typedef cons_index_list<index_uni, uni_uni> uni_uni_uni;
typedef cons_index_list<index_uni, uni_multi> uni_uni_multi;
typedef cons_index_list<index_uni, multi_uni> uni_multi_uni;
typedef cons_index_list<index_uni, multi_multi> uni_multi_multi;
typedef cons_index_list<index_multi, uni_uni> multi_uni_uni;
typedef cons_index_list<index_multi, uni_multi> multi_uni_multi;
typedef cons_index_list<index_multi, multi_uni> multi_multi_uni;
typedef cons_index_list<index_multi, multi_multi> multi_multi_multi;

template <typename T, typename C, typename I>
void expect_same() {
  EXPECT_TRUE(( boost::is_same<T, 
                               typename rvalue_return<C, I>::type>::value )) 
    << "type(T)=" << typeid(T).name() << std::endl
    << "type(C)=" << typeid(C).name() << std::endl
    << "type(I)=" << typeid(I).name() << std::endl
    << "rvalue_return=" << typeid(typename rvalue_return<C, I>::type).name() 
    << std::endl
    << std::endl;
}

// EXHAUSTIVE TESTS OF ALL INDEXING UP TO 2nd ORDER

TEST(modelIndexing,rvalueReturnNil) {
  expect_same<double, 
              double, nil_index_list>();
  expect_same<doubles, 
              doubles, nil_index_list>();
  expect_same<doubless,
              doubless, nil_index_list>();
  expect_same<mat, 
              mat, nil_index_list>();
  expect_same<mats, 
              mats, nil_index_list>();
  expect_same<matss,
              matss, nil_index_list>();
  expect_same<vec, 
              vec, nil_index_list>();
  expect_same<vecs, 
              vecs, nil_index_list>();
  expect_same<vecss,
              vecss, nil_index_list>();
  expect_same<rowvec, 
              rowvec, nil_index_list>();
  expect_same<rowvecs, 
              rowvecs, nil_index_list>();
  expect_same<rowvecss,
              rowvecss, nil_index_list>();
}

TEST(modelIndex, rvalueReturnUni) {
  expect_same<double,
              doubles, uni>();
  expect_same<doubles,
              doubless, uni>();
  expect_same<double,
              vec, uni>();
  expect_same<vec,
              vecs, uni>();
  expect_same<vecs,
              vecss, uni>();
  expect_same<double,
              rowvec, uni>();
  expect_same<rowvec,
              rowvecs, uni>();
  expect_same<rowvecs,
              rowvecss, uni>();
  expect_same<rowvec,
              mat, uni>();
  expect_same<mat,
              mats, uni>();
  expect_same<mats,
              matss, uni>();
}

TEST(modelIndex, rvalueReturnMulti) {
  expect_same<doubles,
              doubles, multi>();
  expect_same<doubless,
              doubless, multi>();
  expect_same<mats,
              mats, multi>();
  expect_same<matss,
              matss, multi>();
  expect_same<vecs,
              vecs, multi>();
  expect_same<vecss,
              vecss, multi>();
  expect_same<rowvecs,
              rowvecs, multi>();
  expect_same<rowvecss,
              rowvecss, multi>();
}
    
TEST(modelIndex, rvalueReturnUniUni) {
  expect_same<double,
              doubless, uni_uni>();
  expect_same<double,
              vecs, uni_uni>();
  expect_same<vec,
              vecss, uni_uni>();
  expect_same<double,
              rowvecs, uni_uni>();
  expect_same<rowvec,
              rowvecss, uni_uni>();
  expect_same<double,
              mat, uni_uni>();
  expect_same<rowvec,
              mats, uni_uni>();
  expect_same<mat,
              matss, uni_uni>();
}

TEST(modelIndex, rvalueReturnUniMulti) {
  expect_same<doubles,
              doubless, uni_multi>();
  expect_same<vec,
              vecs, uni_multi>();
  expect_same<vecs,
              vecss, uni_multi>();
  expect_same<rowvec,
              rowvecs, uni_multi>();
  expect_same<rowvecs,
              rowvecss, uni_multi>();
  expect_same<rowvec,
              mat, uni_multi>();
  expect_same<mat,
              mats, uni_multi>();
  expect_same<mats,
              matss, uni_multi>();
}

TEST(modelIndex, rvalueReturnMultiUni) {
  expect_same<doubles,
              doubless, multi_uni>();
  expect_same<doubles,
              vecs, multi_uni>();
  expect_same<vecs,
              vecss, multi_uni>();
  expect_same<doubles,
              rowvecs, multi_uni>();
  expect_same<rowvecs,
              rowvecss, multi_uni>();
  expect_same<vec,
              mat, multi_uni>();
  expect_same<rowvecs,
              mats, multi_uni>();
  expect_same<mats,
              matss, multi_uni>();
}

TEST(modelIndex, rvalueReturnMultiMulti) {
  expect_same<doubless,
              doubless, multi_multi>();
  expect_same<vecs,
              vecs, multi_multi>();
  expect_same<vecss,
              vecss, multi_multi>();
  expect_same<rowvecs,
              rowvecs, multi_multi>();
  expect_same<rowvecss,
              rowvecss, multi_multi>();
  expect_same<mat,
              mat, multi_multi>();
  expect_same<mats,
              mats, multi_multi>();
  expect_same<matss,
              matss, multi_multi>();
}
