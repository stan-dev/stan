#include <gtest/gtest.h>
#include <stan/common/recorder/values.hpp>
#include <sstream>
#include <vector>

class StanCommonRecorder : public ::testing::Test {
public:
  StanCommonRecorder() :
    N(3), M(5),
    recorder(N, M) { }

  void SetUp() { 
    for (int n = 0; n < N; n++)
      preallocated_values.push_back(std::vector<double>(M));

    for (int n = 0; n < N; n++)
      for (int m = 0; m < M; m++)
        preallocated_values[n][m] = -1.0;
  }
  
  void TearDown() { }
  
  int N;
  int M;
  stan::common::recorder::values<std::vector<double> > recorder;
  std::vector<std::vector<double> > preallocated_values;
};

TEST_F(StanCommonRecorder, values_vector_double) {
  std::vector<double> x;
  for (int n = 1; n <= N; n++) 
    x.push_back(n);
  EXPECT_NO_THROW(recorder(x));
  for (int n = 0; n < N; n++) {
    EXPECT_FLOAT_EQ(x[n], recorder.x()[n][0]);
  }
  for (int n = 0; n < N; n++) 
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
  for (int n = 1; n <= N; n++) 
    x.push_back(10*n);
  EXPECT_NO_THROW(recorder(x));
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, recorder.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), recorder.x()[n][1]);
  for (int m = 2; m < M; m++)
    for (int n = 0; n < N; n++) 
      EXPECT_FLOAT_EQ(0.0, recorder.x()[n][m]);


  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(100*n);
  EXPECT_NO_THROW(recorder(x));

  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, recorder.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), recorder.x()[n][1]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(100*(n+1), recorder.x()[n][2]);
  
  EXPECT_NO_THROW(recorder(x));
  EXPECT_NO_THROW(recorder(x));
  EXPECT_THROW(recorder(x), std::out_of_range);

}

TEST_F(StanCommonRecorder, values_vector_string) {
  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(recorder(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(recorder(y));
}

TEST_F(StanCommonRecorder, values_string) {
  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(recorder(x));
}

TEST_F(StanCommonRecorder, values_noargs) {
  EXPECT_NO_THROW(recorder());
  EXPECT_NO_THROW(recorder());
}

TEST_F(StanCommonRecorder, values_is_recording) {
  std::vector<double> x(N);
  for (int n = 0; n < N; n++) 
    x.at(n) = n;

  for (int m = 0; m < M; m++) {
    EXPECT_TRUE(recorder.is_recording());
    recorder(x);
  }
  EXPECT_FALSE(recorder.is_recording());
}

TEST_F(StanCommonRecorder, values_preallocated_vector_double) {
  stan::common::recorder::values<std::vector<double> > 
    recorder_preallocated(preallocated_values);

  std::vector<double> x;
  for (int n = 1; n <= N; n++) 
    x.push_back(n);
  EXPECT_NO_THROW(recorder_preallocated(x));
  for (int n = 0; n < N; n++) {
    EXPECT_FLOAT_EQ(x[n], recorder_preallocated.x()[n][0]);
  }
  for (int n = 0; n < N; n++) 
    for (int m = 1; m < M; m++)
      EXPECT_FLOAT_EQ(-1.0, recorder_preallocated.x()[n][m]);

  x.clear();

  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  EXPECT_THROW(recorder_preallocated(x), std::length_error);
  
  x.clear();
  x.push_back(1);
  EXPECT_THROW(recorder_preallocated(x), std::length_error);

  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(10*n);
  EXPECT_NO_THROW(recorder_preallocated(x));
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, recorder_preallocated.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), recorder_preallocated.x()[n][1]);
  for (int m = 2; m < M; m++)
    for (int n = 0; n < N; n++) 
      EXPECT_FLOAT_EQ(-1.0, recorder_preallocated.x()[n][m]);


  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(100*n);
  EXPECT_NO_THROW(recorder_preallocated(x));

  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, recorder_preallocated.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), recorder_preallocated.x()[n][1]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(100*(n+1), recorder_preallocated.x()[n][2]);
  
  EXPECT_NO_THROW(recorder_preallocated(x));
  EXPECT_NO_THROW(recorder_preallocated(x));
  EXPECT_THROW(recorder_preallocated(x), std::out_of_range);
}

TEST_F(StanCommonRecorder, values_preallocated_vector_string) {
  stan::common::recorder::values<std::vector<double> > 
    recorder_preallocated(preallocated_values);

  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(recorder_preallocated(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(recorder_preallocated(y));
}

TEST_F(StanCommonRecorder, values_preallocated_string) {
  stan::common::recorder::values<std::vector<double> > 
    recorder_preallocated(preallocated_values);

  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(recorder_preallocated(x));
}

TEST_F(StanCommonRecorder, values_preallocated_noargs) {
  stan::common::recorder::values<std::vector<double> > 
    recorder_preallocated(preallocated_values);

  EXPECT_NO_THROW(recorder_preallocated());
  EXPECT_NO_THROW(recorder_preallocated());
}

TEST_F(StanCommonRecorder, values_preallocated_is_recording) {
  stan::common::recorder::values<std::vector<double> > 
    recorder_preallocated(preallocated_values);

  std::vector<double> x(N);
  for (int n = 0; n < N; n++) 
    x.at(n) = n;

  for (int m = 0; m < M; m++) {
    EXPECT_TRUE(recorder_preallocated.is_recording());
    recorder_preallocated(x);
  }
  EXPECT_FALSE(recorder_preallocated.is_recording());
}
