#include <rstan/io/rlist_var_context.hpp> 

void EXPECT_TRUE(bool a) {
  if (a) Rprintf("Pass.\n"); 
  else Rprintf("Fail.\n"); 
} 

void EXPECT_FALSE(bool a) {
  if (!a) Rprintf("Pass.\n"); 
  else Rprintf("Fail.\n"); 
} 
void EXPECT_EQ(size_t a, size_t b) {
  if (a == b) Rprintf("Pass.\n"); 
  else Rprintf("Fail.\n"); 
} 

void EXPECT_FLOAT_EQ(double a, double b) {
  if (std::fabs(a - b) < .0000001) Rprintf("Pass.\n"); 
  else Rprintf("Fail.\n"); 
} 

void EXPECT_FLOAT_EQ(double a, size_t b) {
  if (std::fabs(a - b) < .0000001) Rprintf("Pass.\n"); 
  else Rprintf("Fail.\n"); 
} 

// test for list(foo = c(1L,2L)
RcppExport SEXP test_rlist_var_context1(SEXP params) {
  BEGIN_RCPP; 
  Rcpp::List rparam(params);  
  rstan::io::rlist_var_context  rlist_(rparam); 

  Rprintf("Test1 started.\n-------\n"); 

  // test for list(foo = c(1L,2L)
  EXPECT_TRUE(rlist_.contains_i("foo"));
  EXPECT_TRUE(rlist_.contains_r("foo"));

  EXPECT_EQ(2U,rlist_.vals_i("foo").size());

  EXPECT_EQ(1,rlist_.vals_i("foo")[0]);
  EXPECT_EQ(2,rlist_.vals_i("foo")[1]);
  EXPECT_FLOAT_EQ(1.0,rlist_.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,rlist_.vals_r("foo")[1]);

  EXPECT_TRUE(rlist_.remove("foo"));
  EXPECT_FALSE(rlist_.remove("foo"));

  EXPECT_FALSE(rlist_.contains_i("foo"));
  EXPECT_FALSE(rlist_.contains_r("foo"));
  return Rcpp::wrap(int(0)); 
  Rprintf("Test1 ended.\n-------\n"); 
  END_RCPP; 
} 

// test for list(foo = c(1L, 2L), bar = 1.0)
RcppExport SEXP test_rlist_var_context2(SEXP params) {
  BEGIN_RCPP;
  Rcpp::List rparam(params);  
  rstan::io::rlist_var_context  rlist_(rparam); 

  Rprintf("Test2 started.\n-------\n"); 
  EXPECT_TRUE(rlist_.contains_i("foo"));
  EXPECT_TRUE(rlist_.contains_r("foo"));  

  EXPECT_FALSE(rlist_.contains_i("bar"));
  EXPECT_TRUE(rlist_.contains_r("bar"));  

  EXPECT_FALSE(rlist_.contains_i("bing"));
  EXPECT_FALSE(rlist_.contains_r("bing"));


  EXPECT_EQ(2U,rlist_.vals_i("foo").size());
  EXPECT_EQ(1U,rlist_.dims_i("foo").size());
  EXPECT_EQ(2U,rlist_.dims_i("foo")[0]);
  EXPECT_EQ(1,rlist_.vals_i("foo")[0]);
  EXPECT_EQ(2,rlist_.vals_i("foo")[1]);

  EXPECT_EQ(2U,rlist_.vals_r("foo").size());
  EXPECT_EQ(1U,rlist_.dims_r("foo").size());
  EXPECT_EQ(2U,rlist_.dims_r("foo")[0]);
  EXPECT_FLOAT_EQ(1.0,rlist_.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,rlist_.vals_r("foo")[1]);

  EXPECT_EQ(1U,rlist_.vals_r("bar").size());
  EXPECT_EQ(0U,rlist_.dims_r("bar").size());
  EXPECT_FLOAT_EQ(1.0,rlist_.vals_r("bar")[0]);

  EXPECT_EQ(0U,rlist_.vals_i("bar").size());
  EXPECT_EQ(0U,rlist_.dims_i("bar").size());

  EXPECT_EQ(0U,rlist_.vals_r("bing").size());
  EXPECT_FALSE(rlist_.contains_r("bing"));
  EXPECT_EQ(0U,rlist_.vals_i("bing").size());
  EXPECT_FALSE(rlist_.contains_i("bing"));

  EXPECT_EQ(0U,rlist_.dims_r("bing").size());
  EXPECT_FALSE(rlist_.contains_r("bing"));
  EXPECT_EQ(0U,rlist_.dims_i("bing").size());
  EXPECT_FALSE(rlist_.contains_i("bing"));
  Rprintf("Test2 ended.\n-------\n"); 
  return Rcpp::wrap(int(0)); 
  END_RCPP;
} 

// test for 

// lst <- list(foo = c(1L, 2L),
//             bar = 1L, 
//             bing = structure(c(1.0, 2.0, 2.0, 5.0, 3.0, 6.0), .Dim = c(2, 3))); 
   
RcppExport SEXP test_rlist_var_context3(SEXP params) {
  BEGIN_RCPP;
  Rcpp::List rparam(params);  
  rstan::io::rlist_var_context  rlist_(rparam); 
  Rprintf("Test3 started.\n-------\n"); 
  EXPECT_TRUE(rlist_.contains_i("foo"));
  EXPECT_TRUE(rlist_.contains_r("foo"));
  EXPECT_TRUE(rlist_.contains_r("bar"));
  EXPECT_FALSE(rlist_.contains_r("baz"));
  EXPECT_FALSE(rlist_.contains_i("bingz"));

  EXPECT_EQ(2U,rlist_.vals_i("foo").size());
  EXPECT_EQ(1U,rlist_.vals_r("bar").size());
  EXPECT_EQ(1,rlist_.vals_i("foo")[0]);
  EXPECT_EQ(2,rlist_.vals_i("foo")[1]);
  EXPECT_FLOAT_EQ(1.0,rlist_.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,rlist_.vals_r("foo")[1]);
  EXPECT_FLOAT_EQ(1.0,rlist_.vals_r("bar")[0]);
  EXPECT_EQ(6U,rlist_.vals_r("bing").size());
  EXPECT_FLOAT_EQ(2.0,rlist_.vals_r("bing")[2]);

  EXPECT_EQ(2U, rlist_.dims_r("bing").size());
  EXPECT_EQ(2U, rlist_.dims_r("bing")[0]);
  EXPECT_EQ(3U, rlist_.dims_r("bing")[1]);

  EXPECT_TRUE(rlist_.remove("bing"));
  EXPECT_FALSE(rlist_.remove("bing"));
  Rprintf("Test3 ended.\n-------\n"); 
  return Rcpp::wrap(int(0)); 
  END_RCPP;
} 

RCPP_MODULE(rstantest){
  using namespace Rcpp;
  function("test_rlist_var_context1", &test_rlist_var_context1); 
  function("test_rlist_var_context2", &test_rlist_var_context2); 
  function("test_rlist_var_context3", &test_rlist_var_context3); 
}


