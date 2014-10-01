#include <stan/meta/typelist.hpp>

#include <boost/type_traits/is_same.hpp> 
#include <gtest/gtest.h>

template <typename T1, typename T2>
void expect_same_type() {
  EXPECT_TRUE(( boost::is_same<T1,T2>::value ));
}
template <typename T1, typename T2>
void expect_diff_type() {
  EXPECT_FALSE(( boost::is_same<T1,T2>::value ));
}

TEST(MetaTypelist, typelist0) {
  using stan::meta::nil;
  using stan::meta::typelist;
  expect_same_type<nil, 
                   typelist< >::type>();
}
TEST(MetaTypelist, typelist1) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::typelist;

  expect_same_type<cons<double,nil>,
                   typelist<double>::type>();
  expect_diff_type<typelist<double>::type, 
                   typelist<int>::type>();
}
TEST(MetaTypelist, typelist2) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::typelist;

  expect_same_type<cons<double,cons<int,nil> >,
                   typelist<double,int>::type>();

  expect_diff_type<typelist<int,double>::type,
                   typelist<double,int>::type>();
}
TEST(MetaTypelist, typelist3) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::typelist;

  expect_same_type<cons<double,cons<int,cons<short,nil> > >,
                   typelist<double,int,short>::type>();
  expect_diff_type<typelist<double,int,short>::type,
                   typelist<double,int>::type>();
}
TEST(MetaTypelist, typelist4) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::typelist;
  
  expect_same_type<cons<double,cons<int,cons<short,cons<int,nil> > > >,
                   typelist<double,int,short,int>::type>();
  expect_diff_type<typelist<double,int,short,int>::type, 
                   typelist<double,int,short,short>::type>();
}

TEST(MetaTypelist, typelist5) {
  using stan::meta::nil;
  using stan::meta::cons;
  using stan::meta::typelist;
  expect_same_type<cons<double,cons<int,cons<short,cons<int,cons<long,nil> > > > >,
                   typelist<double,int,short,int,long>::type>();
  expect_diff_type<typelist<double,int,short,int,long>::type, 
                   typelist<double,int,short,short>::type>();
}
