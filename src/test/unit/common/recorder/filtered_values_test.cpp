#include <gtest/gtest.h>
#include <stan/common/recorder/filtered_values.hpp>
#include <sstream>
#include <vector>

class StanCommonRecorder : public ::testing::Test {
public:
  StanCommonRecorder() :
    N_filter(2),
    N(3), M(5), filter(N_filter) { }

  void SetUp() { 
    filter[0] = 2;
    filter[1] = 1;

    recorder_ptr = new stan::common::recorder::filtered_values
      <std::vector<double> >(N, M, filter);
    for (int n = 0; n < N_filter; n++)
      preallocated_values.push_back(std::vector<double>(M));

    for (int n = 0; n < N_filter; n++)
      for (int m = 0; m < M; m++)
        preallocated_values[n][m] = -1.0;
  }
  
  void TearDown() { 
    delete recorder_ptr;
  }

  int N_filter;
  int N;
  int M;
  stan::common::recorder::filtered_values<std::vector<double> > *recorder_ptr;
  std::vector<std::vector<double> > preallocated_values;
  std::vector<size_t> filter;
};

TEST_F(StanCommonRecorder, filtered_values_vector_double) {
  stan::common::recorder::filtered_values<std::vector<double> > 
    recorder = *recorder_ptr;
  std::vector<double> x(N);
  for (int n = 0; n < N; n++) {
    x.at(n) = n;
  }
  EXPECT_NO_THROW(recorder(x));
  EXPECT_FLOAT_EQ(2.0, recorder.x()[0][0]);
  EXPECT_FLOAT_EQ(1.0, recorder.x()[1][0]);
  for (int n = 0; n < N_filter; n++) 
    for (int m = 1; m < M; m++)
      EXPECT_FLOAT_EQ(0.0, recorder.x()[n][m]);
  
  x.clear();
  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  EXPECT_THROW(recorder(x), std::length_error);
  
  x.clear();
  x.push_back(1);
  EXPECT_THROW(recorder(x), std::length_error);

  x.clear();
  for (int n = 0; n < N; n++) 
    x.push_back(10*n);
  EXPECT_NO_THROW(recorder(x));
  
  EXPECT_FLOAT_EQ(2.0, recorder.x()[0][0]);
  EXPECT_FLOAT_EQ(1.0, recorder.x()[1][0]);
  EXPECT_FLOAT_EQ(20.0, recorder.x()[0][1]);
  EXPECT_FLOAT_EQ(10.0, recorder.x()[1][1]);
  for (int m = 2; m < M; m++)
    for (int n = 0; n < N_filter; n++) 
      EXPECT_FLOAT_EQ(0.0, recorder.x()[n][m]);

  x.clear();
  for (int n = 0; n < N; n++) 
    x.push_back(100*n);
  EXPECT_NO_THROW(recorder(x));

  EXPECT_FLOAT_EQ(2.0, recorder.x()[0][0]);
  EXPECT_FLOAT_EQ(1.0, recorder.x()[1][0]);
  EXPECT_FLOAT_EQ(20.0, recorder.x()[0][1]);
  EXPECT_FLOAT_EQ(10.0, recorder.x()[1][1]);
  EXPECT_FLOAT_EQ(200.0, recorder.x()[0][2]);
  EXPECT_FLOAT_EQ(100.0, recorder.x()[1][2]);
  
  EXPECT_NO_THROW(recorder(x));
  EXPECT_NO_THROW(recorder(x));
  EXPECT_THROW(recorder(x), std::out_of_range);
}

TEST_F(StanCommonRecorder, filtered_values_vector_string) {
  stan::common::recorder::filtered_values<std::vector<double> > 
    recorder = *recorder_ptr;

  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(recorder(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(recorder(y));
}

TEST_F(StanCommonRecorder, filtered_values_string) {
  stan::common::recorder::filtered_values<std::vector<double> > 
    recorder = *recorder_ptr;

  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(recorder(x));
}

TEST_F(StanCommonRecorder, filtered_values_noargs) {
  stan::common::recorder::filtered_values<std::vector<double> > 
    recorder = *recorder_ptr;

  EXPECT_NO_THROW(recorder());
  EXPECT_NO_THROW(recorder());
}

TEST_F(StanCommonRecorder, filtered_values_is_recording) {
  stan::common::recorder::filtered_values<std::vector<double> > 
    recorder = *recorder_ptr;

  std::vector<double> x(N);
  for (int n = 0; n < N; n++) 
    x.at(n) = n;

  for (int m = 0; m < M; m++) {
    EXPECT_TRUE(recorder.is_recording());
    recorder(x);
  }
  EXPECT_FALSE(recorder.is_recording());
}
