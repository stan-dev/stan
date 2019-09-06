#include <gtest/gtest.h>
#include <boost/lexical_cast.hpp>
#include <stan/callbacks/writer.hpp>

class StanInterfaceCallbacksWriter: public ::testing::Test {
public:
  void SetUp() { }
  void TearDown() { }
  stan::callbacks::writer writer;
};

TEST_F(StanInterfaceCallbacksWriter, double_vector) {
  const int N = 5;
  std::vector<double> x;
    for (int n = 0; n < N; ++n) x.push_back(n);

  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceCallbacksWriter, string_vector) {
  const int N = 5;
  std::vector<std::string> x;
    for (int n = 0; n < N; ++n)
      x.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceCallbacksWriter, null) {
  EXPECT_NO_THROW(writer());
}

TEST_F(StanInterfaceCallbacksWriter, string) {
  EXPECT_NO_THROW(writer("message"));
}
