#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/sum_values.hpp>
#include <sstream>
#include <vector>

class StanInterfaceWriter : public ::testing::Test {
public:
  StanInterfaceWriter() :
    N_(10),
    skip_(5),
    writer1(N_), 
    writer2(N_, skip_) { }
  
  void SetUp() {
  }
  
  void TearDown() {
  }
  
  size_t N_;
  size_t skip_;
  stan::interface_callbacks::writer::sum_values writer1;
  stan::interface_callbacks::writer::sum_values writer2;
};

TEST_F(StanInterfaceWriter, sum_values_constructor) {
  EXPECT_EQ(0U, writer1.called());
  EXPECT_EQ(0U, writer2.called());

  EXPECT_EQ(0U, writer1.recorded());
  EXPECT_EQ(0U, writer2.recorded());

  for (size_t n = 0; n < N_; n++) {
    EXPECT_FLOAT_EQ(0, writer1.sum()[n]);
  }
  for (size_t n = 0; n < N_; n++) {
    EXPECT_FLOAT_EQ(0, writer2.sum()[n]);
  }
}


TEST_F(StanInterfaceWriter, sum_values_vector_double) {
  std::vector<double> x;

  for (size_t i = 1; i <= N_; i++) 
    x.push_back(i);

  for (size_t i = 1; i <= 100; i++) {
    EXPECT_NO_THROW(writer1(x));
    EXPECT_EQ(i, writer1.called());
    EXPECT_EQ(i, writer1.recorded());
    
    for (size_t n = 0; n < N_; n++) {
      EXPECT_FLOAT_EQ(x[n] * i, writer1.sum()[n]);
    }
  }
  

  for (size_t i = 1; i <= skip_; i++) {
    EXPECT_NO_THROW(writer2(x));
    EXPECT_EQ(i, writer2.called());
    EXPECT_EQ(0U, writer2.recorded());
    
    for (size_t n = 0; n < N_; n++) {
      EXPECT_FLOAT_EQ(0.0, writer2.sum()[n]);
    }
  }
  for (size_t i = 1; i <= 100; i++) {
    EXPECT_NO_THROW(writer2(x));
    EXPECT_EQ(i, writer2.recorded());
    EXPECT_EQ(i+skip_, writer2.called());
    
    for (size_t n = 0; n < N_; n++) {
      EXPECT_FLOAT_EQ(x[n] * i, writer2.sum()[n]);
    }
  }
}

TEST_F(StanInterfaceWriter, csv_vector_string) {
  std::vector<std::string> y;
  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(writer1(y));
  EXPECT_NO_THROW(writer2(y));

  EXPECT_EQ(0U, writer1.called());
  EXPECT_EQ(0U, writer2.called());

  EXPECT_EQ(0U, writer1.recorded());
  EXPECT_EQ(0U, writer2.recorded());
}

TEST_F(StanInterfaceWriter, sum_values_string) {
  std::string x;
  x = "abcd";

  EXPECT_NO_THROW(writer1(x));
  EXPECT_NO_THROW(writer2(x));

  EXPECT_EQ(0U, writer1.called());
  EXPECT_EQ(0U, writer2.called());

  EXPECT_EQ(0U, writer1.recorded());
  EXPECT_EQ(0U, writer2.recorded());

}

TEST_F(StanInterfaceWriter, sum_values_noargs) {
  EXPECT_NO_THROW(writer1());
  EXPECT_EQ(0U, writer1.called());
  EXPECT_EQ(0U, writer1.recorded());
  
  EXPECT_NO_THROW(writer2());
  EXPECT_EQ(0U, writer2.called());
  EXPECT_EQ(0U, writer2.recorded());
}

TEST_F(StanInterfaceWriter, csv_is_writing) {
  EXPECT_TRUE(writer1.is_writing());
  EXPECT_TRUE(writer2.is_writing());
}
