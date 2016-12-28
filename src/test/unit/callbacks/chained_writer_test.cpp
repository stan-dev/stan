#include <stan/callbacks/chained_writer.hpp>
#include <gtest/gtest.h>

namespace test {
  class mock_writer : public stan::callbacks::writer {
  public:
    int N;

    mock_writer() : N(0) { }

    void operator()(const std::vector<std::string>& names) {
      ++N;
    }

    void operator()(const std::vector<double>& state) {
      ++N;
    }

    void operator()() {
      ++N;
    }

    void operator()(const std::string& message) {
      ++N;
    }
  };
}

class StanCallbacksChainedWriter : public ::testing::Test {
public:
  StanCallbacksChainedWriter()
    : writer1(), writer2(),
      chained_writer(writer1, writer2) { }

  test::mock_writer writer1, writer2;
  stan::callbacks::chained_writer chained_writer;
};

TEST_F(StanCallbacksChainedWriter, names) {
  std::vector<std::string> names;

  chained_writer(names);
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}

TEST_F(StanCallbacksChainedWriter, state) {
  std::vector<double> state;

  chained_writer(state);
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}

TEST_F(StanCallbacksChainedWriter, message) {
  chained_writer("message");
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}
