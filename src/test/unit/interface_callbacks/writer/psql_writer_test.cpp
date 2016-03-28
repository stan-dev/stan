#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/interface_callbacks/writer/psql_writer.hpp>

class StanInterfaceCallbacksPQXXWriter: public ::testing::Test {
public:
  StanInterfaceCallbacksPQXXWriter() :
    writer("","TEST")  {}

  void SetUp() {
  }
  void TearDown() { }

  stan::interface_callbacks::writer::psql_writer writer;
};

TEST_F(StanInterfaceCallbacksPQXXWriter, key_double) {
  EXPECT_NO_THROW(writer("key", 5.2));
//  EXPECT_EQ("key = 5.2\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_int) {
  EXPECT_NO_THROW(writer("key", 5));
//  EXPECT_EQ("key = 5\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_string) {
  EXPECT_NO_THROW(writer("key", "five"));
//  EXPECT_EQ("key = five\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_vector) {
  const int N = 5;
  double x[N];
  for (int n = 0; n < N; ++n) x[n] = n;

  EXPECT_NO_THROW(writer("key", x, N));
//  EXPECT_EQ("key: 0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_matrix) {
  const int n_cols = 2;
  const int n_rows = 3;
  double x[n_cols * n_rows];
  for (int i = 0; i < n_cols; ++i)
    for (int j = 0; j < n_rows; ++j)
      x[i * n_rows + j] = i - j;

  EXPECT_NO_THROW(writer("key", x, n_rows, n_cols));
//  EXPECT_EQ("key\n0,-1\n-2,1\n0,-1\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, double_vector) {
  const int N = 5;
  std::vector<double> x;
    for (int n = 0; n < N; ++n) x.push_back(n);

  EXPECT_NO_THROW(writer(x));
//  EXPECT_EQ("0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, string_vector) {
  const int N = 5;
  std::vector<std::string> x;
    for (int n = 0; n < N; ++n)
      x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer(x));
//  EXPECT_EQ("0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, null) {
  EXPECT_NO_THROW(writer());
//  EXPECT_EQ("\n", ss.str());
}

TEST_F(StanInterfaceCallbacksPQXXWriter, string) {
  EXPECT_NO_THROW(writer("message"));
//  EXPECT_EQ("message\n", ss.str());
}
