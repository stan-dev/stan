#include <stan/callbacks/tee_writer.hpp>
#include <gtest/gtest.h>

namespace test {
class mock_writer : public stan::callbacks::writer {
 public:
  int N;
  bool empty_;

  mock_writer() : N(0) {}
  mock_writer(bool is_empty) : N(0), empty_(is_empty) {}

  void operator()(const std::vector<std::string>& names) {
    if (!empty_) {
      ++N;
    }
  }

  void operator()(const std::vector<double>& state) {
    if (!empty_) {
      ++N;
    }
  }

  void operator()() {
    if (!empty_) {
      ++N;
    }
  }

  void operator()(const std::string& message) {
    if (!empty_) {
      ++N;
    }
  }

  inline bool is_empty() const noexcept {
    return false;
  }
};
}  // namespace test

class StanCallbacksTeeWriter : public ::testing::Test {
 public:
  StanCallbacksTeeWriter()
      : writer1(),
        writer2(),
        tee_writer(writer1, writer2),
        empty_writer1(true),
        empty_writer2(true),
        empty_tee_writer(empty_writer1, empty_writer2) {}

  test::mock_writer writer1, writer2;
  stan::callbacks::tee_writer tee_writer;
  test::mock_writer empty_writer1, empty_writer2;
  stan::callbacks::tee_writer empty_tee_writer;
};

TEST_F(StanCallbacksTeeWriter, names) {
  std::vector<std::string> names;

  tee_writer(names);
  EXPECT_EQ(1, writer1.N);
  EXPECT_EQ(1, writer2.N);
}

TEST_F(StanCallbacksTeeWriter, empty_names) {
  std::vector<std::string> names;

  empty_tee_writer(names);
  EXPECT_EQ(0, empty_writer1.N);
  EXPECT_EQ(0, empty_writer2.N);
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
