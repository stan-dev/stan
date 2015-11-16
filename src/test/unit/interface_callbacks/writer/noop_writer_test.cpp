#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/interface_callbacks/writer/noop_writer.hpp>

class StanInterfaceCallbacksNoopWriter: public ::testing::Test {
public:
  void SetUp() { }
  void TearDown() { }
  stan::interface_callbacks::writer::noop_writer writer;
};

TEST_F(StanInterfaceCallbacksNoopWriter, key_double) {
  EXPECT_NO_THROW(writer("key", 5.0));
}

TEST_F(StanInterfaceCallbacksNoopWriter, key_int) {
  EXPECT_NO_THROW(writer("key", 5));
}

TEST_F(StanInterfaceCallbacksNoopWriter, key_string) {
  EXPECT_NO_THROW(writer("key", "five"));
}

TEST_F(StanInterfaceCallbacksNoopWriter, key_vector) {
  const int N = 5;
  double x[N];
  for (int n = 0; n < N; ++n) x[n] = n;

  EXPECT_NO_THROW(writer("key", x, N));
}

TEST_F(StanInterfaceCallbacksNoopWriter, key_matrix) {
  const int n_cols = 2;
  const int n_rows = 3;
  double x[n_cols * n_rows];
  for (int i = 0; i < n_cols; ++i)
    for (int j = 0; j < n_rows; ++j)
      x[i * n_rows + j] = i - j;

  EXPECT_NO_THROW(writer("key", x, n_rows, n_cols));
}

TEST_F(StanInterfaceCallbacksNoopWriter, double_vector) {
  const int N = 5;
  std::vector<double> x;
    for (int n = 0; n < N; ++n) x.push_back(n);

  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceCallbacksNoopWriter, string_vector) {
  const int N = 5;
  std::vector<std::string> x;
    for (int n = 0; n < N; ++n)
      x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceCallbacksNoopWriter, null) {
  EXPECT_NO_THROW(writer());
}

TEST_F(StanInterfaceCallbacksNoopWriter, string) {
  EXPECT_NO_THROW(writer("message"));
}
