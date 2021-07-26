#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>

class StanInterfaceCallbacksStreamWriter : public ::testing::Test {
 public:
  StanInterfaceCallbacksStreamWriter()
      : writer(std::make_unique<std::stringstream>(std::stringstream{})) {}

  void SetUp() {
    static_cast<std::stringstream&>(writer.get_stream()).str(std::string());
    static_cast<std::stringstream&>(writer.get_stream()).clear();
  }
  void TearDown() {}

  stan::callbacks::unique_stream_writer<std::ostream> writer;
};

TEST_F(StanInterfaceCallbacksStreamWriter, double_vector) {
  const int N = 5;
  std::vector<double> x;
  for (int n = 0; n < N; ++n)
    x.push_back(n);

  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("0,1,2,3,4\n",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, string_vector) {
  const int N = 5;
  std::vector<std::string> x;
  for (int n = 0; n < N; ++n)
    x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer(x));
  EXPECT_EQ("0,1,2,3,4\n",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, null) {
  EXPECT_NO_THROW(writer());
  EXPECT_EQ("\n", static_cast<std::stringstream&>(writer.get_stream()).str());
}

TEST_F(StanInterfaceCallbacksStreamWriter, string) {
  EXPECT_NO_THROW(writer("message"));
  EXPECT_EQ("message\n",
            static_cast<std::stringstream&>(writer.get_stream()).str());
}
