#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>

class StanInterfaceCallbacksStreamWriter: public ::testing::Test {
public:
  StanInterfaceCallbacksStreamWriter() :
    ss(), writer(ss), writer_prefix(ss, "# ") {}

  void SetUp() {
    ss.str(std::string());
    ss.clear();
  }
  void TearDown() { }

  std::stringstream ss;
  stan::interface_callbacks::writer::stream_writer writer;
  stan::interface_callbacks::writer::stream_writer writer_prefix;
};

TEST_F(StanInterfaceCallbacksStreamWriter, key_double) {
  EXPECT_NO_THROW(writer("key", 5.2));
  EXPECT_EQ("key = 5.2\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_double_prefix) {
  EXPECT_NO_THROW(writer_prefix("key", 5.2));
  EXPECT_EQ("# key = 5.2\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_int) {
  EXPECT_NO_THROW(writer("key", 5));
  EXPECT_EQ("key = 5\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_int_prefix) {
  EXPECT_NO_THROW(writer_prefix("key", 5));
  EXPECT_EQ("# key = 5\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_string) {
  EXPECT_NO_THROW(writer("key", "five"));
  EXPECT_EQ("key = five\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_string_prefix) {
  EXPECT_NO_THROW(writer_prefix("key", "five"));
  EXPECT_EQ("# key = five\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_vector) {
  const int N = 5;
  double x[N];
  for (int n = 0; n < N; ++n) x[n] = n;

  EXPECT_NO_THROW(writer("key", x, N));
  EXPECT_EQ("key: 0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_vector_prefix) {
  const int N = 5;
  double x[N];
  for (int n = 0; n < N; ++n) x[n] = n;

  EXPECT_NO_THROW(writer_prefix("key", x, N));
  EXPECT_EQ("# key: 0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_matrix) {
  const int n_cols = 2;
  const int n_rows = 3;
  double x[n_cols * n_rows];
  for (int i = 0; i < n_cols; ++i)
    for (int j = 0; j < n_rows; ++j)
      x[i * n_rows + j] = i - j;

  EXPECT_NO_THROW(writer("key", x, n_rows, n_cols));
  EXPECT_EQ("key\n0,-1\n-2,1\n0,-1\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, key_matrix_prefix) {
  const int n_cols = 2;
  const int n_rows = 3;
  double x[n_cols * n_rows];
  for (int i = 0; i < n_cols; ++i)
    for (int j = 0; j < n_rows; ++j)
      x[i * n_rows + j] = i - j;

  EXPECT_NO_THROW(writer_prefix("key", x, n_rows, n_cols));
  EXPECT_EQ("# key\n# 0,-1\n# -2,1\n# 0,-1\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, double_vector) {
  const int N = 5;
  std::vector<double> x;
    for (int n = 0; n < N; ++n) x.push_back(n);

  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, string_vector) {
  const int N = 5;
  std::vector<std::string> x;
    for (int n = 0; n < N; ++n)
      x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("0,1,2,3,4\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, null) {
  EXPECT_NO_THROW(writer());
  EXPECT_EQ("\n", ss.str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, string) {
  EXPECT_NO_THROW(writer("message"));
  EXPECT_EQ("message\n", ss.str());
}
