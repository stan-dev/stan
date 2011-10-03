#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include <stan/io/dump.hpp>
#include <stan/maths/special_functions.hpp>

void test_list3(stan::io::dump_reader& reader,
	       const std::vector<double>& vals) {
  std::vector<double> vals2 = reader.double_values();
  EXPECT_EQ(vals.size(),vals2.size());
  for (unsigned int i = 0; i < vals.size(); ++i)
    EXPECT_FLOAT_EQ(vals[i],vals2[i]);
}
void test_list3(stan::io::dump_reader& reader,
	       const std::vector<int>& vals) {
  std::vector<int> vals2 = reader.int_values();
  EXPECT_EQ(vals.size(),vals2.size());
  for (unsigned int i = 0; i < vals.size(); ++i)
    EXPECT_EQ(vals[i],vals2[i]);
}
template <typename T>
void test_list2(stan::io::dump_reader& reader,
		const std::string& name,
		const std::vector<T>& vals,
		const std::vector<unsigned int>& dims) {
  bool has_next = reader.next();
  EXPECT_EQ(true,has_next);
  EXPECT_EQ(name,reader.name());
  EXPECT_EQ(dims.size(), reader.dims().size());
  for (unsigned int i = 0; i < dims.size(); ++i)
    EXPECT_EQ(dims[i],reader.dims()[i]);
  test_list3(reader,vals);
}


template <typename T>
void test_list(const std::string& name, 
	       const std::vector<T>& vals, 
	       const std::string& s) {
  std::stringstream in(s);
  stan::io::dump_reader reader(in);
  std::vector<unsigned int> expected_dims;
  expected_dims.push_back(vals.size());
  test_list2(reader,name,vals,expected_dims);
}


template <typename T>
void test_val(std::string name, T val, std::string s) {
  std::stringstream in(s);
  stan::io::dump_reader reader(in);
  std::vector<T> vals;
  vals.push_back(val);
  std::vector<unsigned int> expected_dims;
  test_list2(reader,name,vals,expected_dims);
}

TEST(io_dump, reader_double) {
  test_val("a",5.0,"a <- 5.0");
  test_val("a",0.0,"a <- 0.0");
  test_val("a",-5.0,"a <- -5.0");
}

TEST(io_dump, reader_int) {
  test_val("a",5,"a <- 5");
  test_val("a",-1,"a <- -1");
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
  test_list("b12",vs,"b12 <- c(-5, -2, 3, 0)");
}

TEST(io_dump, reader_vec_double) {
  std::vector<double> expected_vals;
  expected_vals.push_back(1.0);
  expected_vals.push_back(4.0);
  expected_vals.push_back(2.0);
  expected_vals.push_back(5.0);
  expected_vals.push_back(3.0);
  expected_vals.push_back(6.0);
  std::vector<unsigned int> expected_dims;
  expected_dims.push_back(2U);
  expected_dims.push_back(3U);
  std::string txt = "foo <- structure(c(1.0,4.0,2.0,5.0,3.0,6.0), .Dim = c(2,3))";
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
  std::vector<unsigned int> expected_dims;
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

TEST(io_dump,dump) {
  std::string txt = "foo <- c(1,2)\nbar<-1.0\n\"bing\"<-\nstructure(c(1.0,4.0,2.0,5.0,3.0,6.0), .Dim = c(2,3))";
  std::stringstream in(txt);
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("foo"));
  EXPECT_TRUE(dump.contains_r("foo"));
  EXPECT_TRUE(dump.contains_r("bar"));
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
  
  EXPECT_EQ(2U, dump.dims_r("bing").size());
  EXPECT_EQ(2U, dump.dims_r("bing")[0]);
  EXPECT_EQ(3U, dump.dims_r("bing")[1]);

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

  EXPECT_EQ(2,dump.vals_i("foo").size());
  EXPECT_EQ(1,dump.dims_i("foo").size());
  EXPECT_EQ(2,dump.dims_i("foo")[0]);
  EXPECT_EQ(1,dump.vals_i("foo")[0]);
  EXPECT_EQ(2,dump.vals_i("foo")[1]);

  EXPECT_EQ(2,dump.vals_r("foo").size());
  EXPECT_EQ(1,dump.dims_r("foo").size());
  EXPECT_EQ(2,dump.dims_r("foo")[0]);
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("foo")[0]);
  EXPECT_FLOAT_EQ(2.0,dump.vals_r("foo")[1]);

  EXPECT_EQ(1,dump.vals_r("bar").size());
  EXPECT_EQ(0,dump.dims_r("bar").size());
  EXPECT_FLOAT_EQ(1.0,dump.vals_r("bar")[0]);

  EXPECT_EQ(0,dump.vals_i("bar").size());
  EXPECT_EQ(0,dump.dims_i("bar").size());

  EXPECT_EQ(0,dump.vals_r("bing").size());
  EXPECT_FALSE(dump.contains_r("bing"));
  EXPECT_EQ(0,dump.vals_i("bing").size());
  EXPECT_FALSE(dump.contains_i("bing"));

  EXPECT_EQ(0,dump.dims_r("bing").size());
  EXPECT_FALSE(dump.contains_r("bing"));
  EXPECT_EQ(0,dump.dims_i("bing").size());
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

TEST(io_dump, dump_file) {
  std::fstream in("src/models/normal_estimate/normal_estimate.data");
  stan::io::dump dump(in);
  EXPECT_TRUE(dump.contains_i("N"));
  stan::io::var_context& context = dump;
  EXPECT_TRUE(context.contains_i("N"));
}
