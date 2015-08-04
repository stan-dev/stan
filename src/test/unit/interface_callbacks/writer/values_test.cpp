#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/values.hpp>
#include <sstream>
#include <vector>

class StanInterfaceWriter : public ::testing::Test {
public:
  StanInterfaceWriter() :
    N(3), M(5),
    writer(N, M) { }

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
  stan::interface_callbacks::writer::values<std::vector<double> > writer;
  std::vector<std::vector<double> > preallocated_values;
};

TEST_F(StanInterfaceWriter, values_vector_double) {
  std::vector<double> x;
  for (int n = 1; n <= N; n++) 
    x.push_back(n);
  EXPECT_NO_THROW(writer(x));
  for (int n = 0; n < N; n++) {
    EXPECT_FLOAT_EQ(x[n], writer.x()[n][0]);
  }
  for (int n = 0; n < N; n++) 
    for (int m = 1; m < M; m++)
      EXPECT_FLOAT_EQ(0.0, writer.x()[n][m]);

  x.clear();

  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  EXPECT_THROW(writer(x), std::length_error);
  
  x.clear();
  x.push_back(1);
  EXPECT_THROW(writer(x), std::length_error);

  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(10*n);
  EXPECT_NO_THROW(writer(x));
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, writer.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), writer.x()[n][1]);
  for (int m = 2; m < M; m++)
    for (int n = 0; n < N; n++) 
      EXPECT_FLOAT_EQ(0.0, writer.x()[n][m]);


  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(100*n);
  EXPECT_NO_THROW(writer(x));

  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, writer.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), writer.x()[n][1]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(100*(n+1), writer.x()[n][2]);
  
  EXPECT_NO_THROW(writer(x));
  EXPECT_NO_THROW(writer(x));
  EXPECT_THROW(writer(x), std::out_of_range);

}

TEST_F(StanInterfaceWriter, values_vector_string) {
  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(writer(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(writer(y));
}

TEST_F(StanInterfaceWriter, values_string) {
  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceWriter, values_noargs) {
  EXPECT_NO_THROW(writer());
  EXPECT_NO_THROW(writer());
}

TEST_F(StanInterfaceWriter, values_is_writing) {
  std::vector<double> x(N);
  for (int n = 0; n < N; n++) 
    x.at(n) = n;

  for (int m = 0; m < M; m++) {
    EXPECT_TRUE(writer.is_writing());
    writer(x);
  }
  EXPECT_FALSE(writer.is_writing());
}

TEST_F(StanInterfaceWriter, values_preallocated_vector_double) {
  stan::interface_callbacks::writer::values<std::vector<double> > 
    writer_preallocated(preallocated_values);

  std::vector<double> x;
  for (int n = 1; n <= N; n++) 
    x.push_back(n);
  EXPECT_NO_THROW(writer_preallocated(x));
  for (int n = 0; n < N; n++) {
    EXPECT_FLOAT_EQ(x[n], writer_preallocated.x()[n][0]);
  }
  for (int n = 0; n < N; n++) 
    for (int m = 1; m < M; m++)
      EXPECT_FLOAT_EQ(-1.0, writer_preallocated.x()[n][m]);

  x.clear();

  for (int i = 1; i < 10; i++) 
    x.push_back(10*i);
  EXPECT_THROW(writer_preallocated(x), std::length_error);
  
  x.clear();
  x.push_back(1);
  EXPECT_THROW(writer_preallocated(x), std::length_error);

  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(10*n);
  EXPECT_NO_THROW(writer_preallocated(x));
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, writer_preallocated.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), writer_preallocated.x()[n][1]);
  for (int m = 2; m < M; m++)
    for (int n = 0; n < N; n++) 
      EXPECT_FLOAT_EQ(-1.0, writer_preallocated.x()[n][m]);


  x.clear();
  for (int n = 1; n <= N; n++) 
    x.push_back(100*n);
  EXPECT_NO_THROW(writer_preallocated(x));

  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(n+1, writer_preallocated.x()[n][0]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(10*(n+1), writer_preallocated.x()[n][1]);
  for (int n = 0; n < N; n++) 
    EXPECT_FLOAT_EQ(100*(n+1), writer_preallocated.x()[n][2]);
  
  EXPECT_NO_THROW(writer_preallocated(x));
  EXPECT_NO_THROW(writer_preallocated(x));
  EXPECT_THROW(writer_preallocated(x), std::out_of_range);
}

TEST_F(StanInterfaceWriter, values_preallocated_vector_string) {
  stan::interface_callbacks::writer::values<std::vector<double> > 
    writer_preallocated(preallocated_values);

  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(writer_preallocated(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(writer_preallocated(y));
}

TEST_F(StanInterfaceWriter, values_preallocated_string) {
  stan::interface_callbacks::writer::values<std::vector<double> > 
    writer_preallocated(preallocated_values);

  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(writer_preallocated(x));
}

TEST_F(StanInterfaceWriter, values_preallocated_noargs) {
  stan::interface_callbacks::writer::values<std::vector<double> > 
    writer_preallocated(preallocated_values);

  EXPECT_NO_THROW(writer_preallocated());
  EXPECT_NO_THROW(writer_preallocated());
}

TEST_F(StanInterfaceWriter, values_preallocated_is_writing) {
  stan::interface_callbacks::writer::values<std::vector<double> > 
    writer_preallocated(preallocated_values);

  std::vector<double> x(N);
  for (int n = 0; n < N; n++) 
    x.at(n) = n;

  for (int m = 0; m < M; m++) {
    EXPECT_TRUE(writer_preallocated.is_writing());
    writer_preallocated(x);
  }
  EXPECT_FALSE(writer_preallocated.is_writing());
}
