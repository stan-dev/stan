#include <gtest/gtest.h>
#include <stan/interface/recorder/sum_values.hpp>
#include <sstream>
#include <vector>

class StanInterfaceRecorder : public ::testing::Test {
public:
  StanInterfaceRecorder() :
    N_(10),
    skip_(5),
    recorder1(N_), 
    recorder2(N_, skip_) { }
  
  void SetUp() {
  }
  
  void TearDown() {
  }
  
  size_t N_;
  size_t skip_;
  stan::interface::recorder::sum_values recorder1;
  stan::interface::recorder::sum_values recorder2;
};

TEST_F(StanInterfaceRecorder, sum_values_constructor) {
  EXPECT_EQ(0U, recorder1.called());
  EXPECT_EQ(0U, recorder2.called());

  EXPECT_EQ(0U, recorder1.recorded());
  EXPECT_EQ(0U, recorder2.recorded());

  for (size_t n = 0; n < N_; n++) {
    EXPECT_FLOAT_EQ(0, recorder1.sum()[n]);
  }
  for (size_t n = 0; n < N_; n++) {
    EXPECT_FLOAT_EQ(0, recorder2.sum()[n]);
  }
}


TEST_F(StanInterfaceRecorder, sum_values_vector_double) {
  std::vector<double> x;

  for (size_t i = 1; i <= N_; i++) 
    x.push_back(i);

  for (size_t i = 1; i <= 100; i++) {
    EXPECT_NO_THROW(recorder1(x));
    EXPECT_EQ(i, recorder1.called());
    EXPECT_EQ(i, recorder1.recorded());
    
    for (size_t n = 0; n < N_; n++) {
      EXPECT_FLOAT_EQ(x[n] * i, recorder1.sum()[n]);
    }
  }
  

  for (size_t i = 1; i <= skip_; i++) {
    EXPECT_NO_THROW(recorder2(x));
    EXPECT_EQ(i, recorder2.called());
    EXPECT_EQ(0U, recorder2.recorded());
    
    for (size_t n = 0; n < N_; n++) {
      EXPECT_FLOAT_EQ(0.0, recorder2.sum()[n]);
    }
  }
  for (size_t i = 1; i <= 100; i++) {
    EXPECT_NO_THROW(recorder2(x));
    EXPECT_EQ(i, recorder2.recorded());
    EXPECT_EQ(i+skip_, recorder2.called());
    
    for (size_t n = 0; n < N_; n++) {
      EXPECT_FLOAT_EQ(x[n] * i, recorder2.sum()[n]);
    }
  }
}

TEST_F(StanInterfaceRecorder, csv_vector_string) {
  std::vector<std::string> y;
  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(recorder1(y));
  EXPECT_NO_THROW(recorder2(y));

  EXPECT_EQ(0U, recorder1.called());
  EXPECT_EQ(0U, recorder2.called());

  EXPECT_EQ(0U, recorder1.recorded());
  EXPECT_EQ(0U, recorder2.recorded());
}

TEST_F(StanInterfaceRecorder, sum_values_string) {
  std::string x;
  x = "abcd";

  EXPECT_NO_THROW(recorder1(x));
  EXPECT_NO_THROW(recorder2(x));

  EXPECT_EQ(0U, recorder1.called());
  EXPECT_EQ(0U, recorder2.called());

  EXPECT_EQ(0U, recorder1.recorded());
  EXPECT_EQ(0U, recorder2.recorded());

}

TEST_F(StanInterfaceRecorder, sum_values_noargs) {
  EXPECT_NO_THROW(recorder1());
  EXPECT_EQ(0U, recorder1.called());
  EXPECT_EQ(0U, recorder1.recorded());
  
  EXPECT_NO_THROW(recorder2());
  EXPECT_EQ(0U, recorder2.called());
  EXPECT_EQ(0U, recorder2.recorded());
}

TEST_F(StanInterfaceRecorder, csv_is_recording) {
  EXPECT_TRUE(recorder1.is_recording());
  EXPECT_TRUE(recorder2.is_recording());
}
