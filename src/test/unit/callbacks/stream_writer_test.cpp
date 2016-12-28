#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/callbacks/stream_writer.hpp>

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
  stan::callbacks::stream_writer writer;
  stan::callbacks::stream_writer writer_prefix;
};

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
