#include <stan/io/dump.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

void test_list3(stan::io::dump_reader& reader,
               const std::vector<double>& vals) {
  std::vector<double> vals2 = reader.double_values();
  EXPECT_EQ(vals.size(),vals2.size());
  for (size_t i = 0; i < vals.size(); ++i) {
    if (boost::math::isnan(vals[i]))
      EXPECT_TRUE(boost::math::isnan(vals2[i]));
    else
      EXPECT_FLOAT_EQ(vals[i],vals2[i]);
  }
}
void test_list3(stan::io::dump_reader& reader,
               const std::vector<int>& vals) {
  std::vector<int> vals2 = reader.int_values();
  EXPECT_EQ(vals.size(),vals2.size());
  for (size_t i = 0; i < vals.size(); ++i)
    EXPECT_EQ(vals[i],vals2[i]);
}
template <typename T>
void test_list2(stan::io::dump_reader& reader,
                const std::string& name,
                const std::vector<T>& vals,
                const std::vector<size_t>& dims) {
  bool has_next = reader.next();
  EXPECT_EQ(true,has_next);
  EXPECT_EQ(name,reader.name());
  EXPECT_EQ(dims.size(), reader.dims().size());
  for (size_t i = 0; i < dims.size(); ++i)
    EXPECT_EQ(dims[i],reader.dims()[i]);
  test_list3(reader,vals);
}


template <typename T>
void test_list(const std::string& name, 
               const std::vector<T>& vals, 
               const std::string& s) {
  std::stringstream in(s);
  stan::io::dump_reader reader(in);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(vals.size());
  test_list2(reader,name,vals,expected_dims);
}


template <typename T>
void test_val(std::string name, T val, std::string s) {
  std::stringstream in(s);
  stan::io::dump_reader reader(in);
  std::vector<T> vals;
  vals.push_back(val);
  std::vector<size_t> expected_dims;
  test_list2(reader,name,vals,expected_dims);
}

TEST(io_dump, reader_double) {
  test_val("a",-5.0,"a <- -5.0");
  test_val("a",5.0,"a <- 5.0");
  test_val("a",0.0,"a <- 0.0");
}

TEST(io_dump, reader_int) {
  test_val("a",5,"a <- 5");
  test_val("a",-1,"a <- -1");
  test_val("a",8,"a <- 8L");
}

TEST(io_dump, reader_doubles) {
  std::vector<double> vs;
  test_list("a",vs,"a <- c()");

  vs.clear();
  vs.push_back(5.0);
  vs.push_back(-6.0);
  test_list("a",vs,"a <- c(5.0,-6.0)");

  vs.clear();
  vs.push_back(0.0001);
  test_list("xYz",vs,"xYz <- c(.0001)");

  vs.clear();
  vs.push_back(-0.0001);
  test_list("xYz",vs,"xYz <- c(-.0001)");

  vs.clear();
  vs.push_back(-5);
  vs.push_back(-2.12);
  vs.push_back(3.0);
  vs.push_back(0.0);
  test_list("b12",vs,"b12 <- c(-5.0, -2.12, 3.0, 0.0)");
}

TEST(io_dump, reader_ints) {
  std::vector<int> vs;
  test_list("a",vs,"a <- c()");

  vs.clear();
  vs.push_back(5);
  vs.push_back(-6);
  test_list("a",vs,"a <- c(5,-6)");

  vs.clear();
  vs.push_back(0);
  test_list("xYz",vs,"xYz <- c(0)");

  vs.clear();
  vs.push_back(-5);
  vs.push_back(-2);
  vs.push_back(3);
  vs.push_back(0);
  test_list("b12",vs,"b12 <- c(-5, -2L, 3, 0l)");

  vs.clear();
  vs.push_back(1);
  vs.push_back(2);
  vs.push_back(3);
  vs.push_back(4);
  vs.push_back(5);
  test_list("z98",vs,"z98 <- 1:5");
  
  vs.clear();
  vs.push_back(9);
  vs.push_back(8);
  test_list("iroc",vs,"iroc <- 9:8");
}


TEST(io_dump, reader_vec_data) {
  std::vector<int> expected_vals;
  for (int i = 1; i <= 6; ++i)
    expected_vals.push_back(i);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2U);
  expected_dims.push_back(3U);

  std::string txt = "foo <- structure(1:6, .Dim = c(2,3))";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"foo",expected_vals,expected_dims);
}
TEST(io_dump, reader_vec_data_backward) {
  std::vector<int> expected_vals;
  for (int i = 20; i >= 1; --i)
    expected_vals.push_back(i);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(5U);
  expected_dims.push_back(4U);

  std::string txt = "foo <- structure(20:1, .Dim = c(5,4))";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"foo",expected_vals,expected_dims);
}
TEST(io_dump, reader_vec_data_long_suffix) {
  std::vector<int> expected_vals;
  for (int i = 10; i >= -9; --i)
    expected_vals.push_back(i);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(20U);
  std::string txt = "a <-\nc(10L, 9L, 8L, 7L, 6L, 5L, 4L, 3L, 2L, 1L, 0L, -1L, -2L, -3L, \n-4L, -5L, -6L, -7L, -8L, -9L)";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"a",expected_vals,expected_dims);
}
TEST(io_dump, reader_nan_inf) {
  std::string txt = "a <- c(-1.0, Inf, -Inf, 0, Infinity, 129, NaN, -4)"; 
  std::vector<double> expected_vals;
  expected_vals.push_back(-1.0);
  expected_vals.push_back(std::numeric_limits<double>::infinity());
  expected_vals.push_back(-std::numeric_limits<double>::infinity());
  expected_vals.push_back(0.0);
  expected_vals.push_back(std::numeric_limits<double>::infinity());
  expected_vals.push_back(129.0);
  expected_vals.push_back(std::numeric_limits<double>::quiet_NaN());
  expected_vals.push_back(-4);

  std::vector<size_t> expected_dims;
  expected_dims.push_back(expected_vals.size());
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"a",expected_vals,expected_dims);
}

TEST(io_dump, reader_vec_double) {
  std::vector<double> expected_vals;
  expected_vals.push_back(1.0);
  expected_vals.push_back(4.0);
  expected_vals.push_back(2.0);
  expected_vals.push_back(5.0);
  expected_vals.push_back(3.0);
  expected_vals.push_back(6.0);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2U);
  expected_dims.push_back(3U);
  std::string txt = "foo <- structure(c(1.0,4.0,2.0,5.0,3.0,6.0), .Dim = c(2,3))";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"foo",expected_vals,expected_dims);
}
TEST(io_dump, reader_vec_double_dots) {
  std::vector<double> expected_vals;
  expected_vals.push_back(1.0);
  expected_vals.push_back(4.0);
  expected_vals.push_back(2.0);
  expected_vals.push_back(5.0);
  expected_vals.push_back(3.0);
  expected_vals.push_back(6.0);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2U);
  expected_dims.push_back(3U);
  std::string txt = "foo <- structure(c(1.0,4.0,2.0,5.0,3.0,6.0), .Dim = 2:3))";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"foo",expected_vals,expected_dims);
}
TEST(io_dump, reader_vec_double_dots_rev) {
  std::vector<double> expected_vals;
  expected_vals.push_back(1.0);
  expected_vals.push_back(4.0);
  expected_vals.push_back(2.0);
  expected_vals.push_back(5.0);
  expected_vals.push_back(3.0);
  expected_vals.push_back(6.0);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(3U);
  expected_dims.push_back(2U);
  std::string txt = "foo <- structure(c(1.0,4.0,2.0,5.0,3.0,6.0), .Dim = 3:2))";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"foo",expected_vals,expected_dims);
}



TEST(io_dump, reader_vec_int) {
  std::vector<int> expected_vals;
  expected_vals.push_back(1);
  expected_vals.push_back(4);
  expected_vals.push_back(2);
  expected_vals.push_back(5);
  expected_vals.push_back(3);
  expected_vals.push_back(6);
  std::vector<size_t> expected_dims;
  expected_dims.push_back(2U);
  expected_dims.push_back(3U);
  std::string txt = "foo <- structure(c(1,4,2,5,3,6), .Dim = c(2,3))";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  test_list2(reader,"foo",expected_vals,expected_dims);
}

TEST(io_dump, reader_sequence) {
  std::string txt = "foo <- 1\nbar <- 2";
  std::stringstream in(txt);
  stan::io::dump_reader reader(in);
  EXPECT_TRUE(reader.next());
  EXPECT_EQ("foo",reader.name());
  EXPECT_TRUE(reader.next());
  EXPECT_EQ("bar",reader.name());
  EXPECT_FALSE(reader.next());
}

TEST(io_dump,two_lines) {
  std::string txt = "foo <- 3\nbar <- 4";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("foo"));
  EXPECT_TRUE(dump.contains_i("bar"));

  std::string txt2 = "foo <- 3\nloo <- 4";
  std::stringstream in2(txt2);
  stan::io::dump dump2(in2);
  EXPECT_TRUE(dump2.contains_i("foo"));
  EXPECT_FALSE(dump2.contains_i("oo"));
  EXPECT_TRUE(dump2.contains_i("loo"));
}

TEST(io_dump,dump) {
  std::string txt = "foo <- c(1,2)\nbar<-1.0\n\"bing\"<-\nstructure(c(1.0,4.0,2.0,5.0,3.0,6.0), .Dim = c(2,3))\nqux <- 2.0\nquux<-structure(c(1.0,2.0,3.0,4.0), .Dim = c(2L, 2L))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("foo"));
  EXPECT_TRUE(dump.contains_r("foo"));
  EXPECT_TRUE(dump.contains_r("bar"));
  EXPECT_TRUE(dump.contains_r("qux"));
  EXPECT_TRUE(dump.contains_r("quux"));
  EXPECT_FALSE(dump.contains_r("baz"));
  EXPECT_FALSE(dump.contains_i("bingz"));

  EXPECT_EQ(2U,dump.vals_i("foo").size());
  EXPECT_EQ(1U,dump.vals_r("bar").size());
  EXPECT_EQ(1,dump.vals_i("foo")[0]);
  EXPECT_EQ(2,dump.vals_i("foo")[1]);
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,dump.vals_r("foo")[1]);
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("bar")[0]);
  EXPECT_EQ(6U,dump.vals_r("bing").size());
  EXPECT_FLOAT_EQ(2.0,dump.vals_r("bing")[2]);
  EXPECT_EQ(1U,dump.vals_r("qux").size());
  EXPECT_EQ(4U,dump.vals_r("quux").size());
  EXPECT_FLOAT_EQ(4.0,dump.vals_r("quux")[3]);
  
  EXPECT_EQ(2U, dump.dims_r("bing").size());
  EXPECT_EQ(2U, dump.dims_r("bing")[0]);
  EXPECT_EQ(3U, dump.dims_r("bing")[1]);

  EXPECT_EQ(2U, dump.dims_r("quux").size());
  EXPECT_EQ(2U, dump.dims_r("quux")[0]);
  EXPECT_EQ(2U, dump.dims_r("quux")[1]);

  EXPECT_TRUE(dump.remove("bing"));
  EXPECT_FALSE(dump.remove("bing"));
}

TEST(io_dump,dump_contains_i) {
  std::string txt = "foo <- c(1,2)";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("foo"));
  EXPECT_TRUE(dump.contains_r("foo"));

  EXPECT_EQ(2U,dump.vals_i("foo").size());

  EXPECT_EQ(1,dump.vals_i("foo")[0]);
  EXPECT_EQ(2,dump.vals_i("foo")[1]);
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,dump.vals_r("foo")[1]);

  EXPECT_TRUE(dump.remove("foo"));
  EXPECT_FALSE(dump.remove("foo"));

  EXPECT_FALSE(dump.contains_i("foo"));
  EXPECT_FALSE(dump.contains_r("foo"));
}

TEST(io_dump, dump_safety) {
  std::string txt = "foo <- c(1,2)\nbar<-1.0";
  //std::string txt = "bar<-1.0\nfoo <- c(1,2)\n";
  std::stringstream in(txt);
  stan::io::dump dump(in);

  EXPECT_TRUE(dump.contains_i("foo"));
  EXPECT_TRUE(dump.contains_r("foo"));  

  EXPECT_FALSE(dump.contains_i("bar"));
  EXPECT_TRUE(dump.contains_r("bar"));  

  EXPECT_FALSE(dump.contains_i("bing"));
  EXPECT_FALSE(dump.contains_r("bing"));

  
  EXPECT_EQ(2U,dump.vals_i("foo").size());
  EXPECT_EQ(1U,dump.dims_i("foo").size());
  EXPECT_EQ(2U,dump.dims_i("foo")[0]);
  EXPECT_EQ(1,dump.vals_i("foo")[0]);
  EXPECT_EQ(2,dump.vals_i("foo")[1]);

  EXPECT_EQ(2U,dump.vals_r("foo").size());
  EXPECT_EQ(1U,dump.dims_r("foo").size());
  EXPECT_EQ(2U,dump.dims_r("foo")[0]);
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,dump.vals_r("foo")[1]);

  EXPECT_EQ(1U,dump.vals_r("bar").size());
  EXPECT_EQ(0U,dump.dims_r("bar").size());
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("bar")[0]);

  EXPECT_EQ(0U,dump.vals_i("bar").size());
  EXPECT_EQ(0U,dump.dims_i("bar").size());

  EXPECT_EQ(0U,dump.vals_r("bing").size());
  EXPECT_FALSE(dump.contains_r("bing"));
  EXPECT_EQ(0U,dump.vals_i("bing").size());
  EXPECT_FALSE(dump.contains_i("bing"));

  EXPECT_EQ(0U,dump.dims_r("bing").size());
  EXPECT_FALSE(dump.contains_r("bing"));
  EXPECT_EQ(0U,dump.dims_i("bing").size());
  EXPECT_FALSE(dump.contains_i("bing"));
}

TEST(io_dump, dump_abs_ref) {
  std::string txt = "\"N\" <-\n5\n\"y\" <-\n c(1.0, 2.0, 1.2, -0.2, 2.7)";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("N"));
  stan::io::var_context& context = dump;
  EXPECT_TRUE(context.contains_i("N"));
}

TEST(io_dump, product) {
  std::vector<size_t>dims;
  dims.push_back(1);
  
  EXPECT_FLOAT_EQ(1.0, stan::io::product(dims));
  dims.push_back(3);
  dims.push_back(4);
  EXPECT_FLOAT_EQ(12.0, stan::io::product(dims));
}


// thanks to ksvanhorn for pointing out this test case
// which failed in Stan 1.3
TEST(io_dump, it_sign_ksvanhorn) {
  using std::vector;
  std::string txt
    = "N <- 5\ny <- c(2, 1, 1, 2, -3.4)\nsigma <- 3\n";
  vector<size_t> dims;
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("N"));
  EXPECT_TRUE(dump.contains_r("N"));
  EXPECT_TRUE(dump.contains_r("y"));
  EXPECT_TRUE(dump.contains_i("sigma"));
  EXPECT_TRUE(dump.contains_r("sigma"));

  vector<int> N_values = dump.vals_i("N");
  EXPECT_EQ(1U,N_values.size());
  EXPECT_EQ(5,N_values[0]);

  vector<double> y_values = dump.vals_r("y");
  EXPECT_EQ(5U,y_values.size());
  EXPECT_FLOAT_EQ(2,y_values[0]);
  EXPECT_FLOAT_EQ(1,y_values[1]);
  EXPECT_FLOAT_EQ(1,y_values[2]);
  EXPECT_FLOAT_EQ(2,y_values[3]);
  EXPECT_FLOAT_EQ(-3.4,y_values[4]);

  vector<double> sigma_values = dump.vals_r("sigma");
  EXPECT_EQ(1U,sigma_values.size());
  EXPECT_FLOAT_EQ(3,sigma_values[0]);
  
}
