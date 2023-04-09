#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>

class StanInterfaceCallbacksStreamWriter : public ::testing::Test {
 public:
  StanInterfaceCallbacksStreamWriter()
      : ss(),
        writer(std::unique_ptr<std::stringstream>(&ss)) {}
  
  void SetUp() {
    ss.str(std::string());
    ss.clear();
  }
  void TearDown() {}

  std::stringstream ss;
  stan::callbacks::unique_stream_writer<std::stringstream> writer;
};

TEST_F(StanInterfaceCallbacksStreamWriter, double_vector) {
  const int N = 5;
  std::vector<double> x;
  for (int n = 0; n < N; ++n)
    x.push_back(n);

  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("0,1,2,3,4\n", writer.get_stream().str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, double_vector_precision2) {
  ss << std::setprecision(2);
  const int N = 5;
  std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("1.2,2.3,3.5,4.6\n", writer.get_stream().str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, double_vector_precision3) {
  ss << std::setprecision(3);
  const int N = 5;
  std::vector<double> x{1.23456789, 2.3456789, 3.45678910, 4.567890123};
  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("1.23,2.35,3.46,4.57\n", writer.get_stream().str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, string_vector) {
  const int N = 5;
  std::vector<std::string> x;
  for (int n = 0; n < N; ++n)
    x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("0,1,2,3,4\n", writer.get_stream().str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, null) {
  EXPECT_NO_THROW(writer());
  EXPECT_EQ("\n", writer.get_stream().str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, string) {
  EXPECT_NO_THROW(writer("message"));
  EXPECT_EQ("message\n",
            writer.get_stream().str());
}
