#include <stan/callbacks/tee_writer.hpp>
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

class StanCallbacksTeeWriter : public ::testing::Test {
public:
  StanCallbacksTeeWriter()
    : writer1(), writer2(),
      tee_writer(writer1, writer2) { }

  test::mock_writer writer1, writer2;
  stan::callbacks::tee_writer tee_writer;
};

TEST_F(StanCallbacksTeeWriter, names) {
  std::vector<std::string> names;

  tee_writer(names);
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}

TEST_F(StanCallbacksTeeWriter, state) {
  std::vector<double> state;

  tee_writer(state);
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}

TEST_F(StanCallbacksTeeWriter, message) {
  tee_writer("message");
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}
