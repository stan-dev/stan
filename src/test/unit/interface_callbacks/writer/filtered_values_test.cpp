#include <gtest/gtest.h>
#include <stan/interface_callbacks/writer/filtered_values.hpp>
#include <sstream>
#include <vector>

class StanInterfaceWriter : public ::testing::Test {
public:
  StanInterfaceWriter() :
    N_filter(2),
    N(3), M(5), filter(N_filter) { }

  void SetUp() { 
    filter[0] = 2;
    filter[1] = 1;

    writer_ptr = new stan::interface_callbacks::writer::filtered_values
      <std::vector<double> >(N, M, filter);
    for (int n = 0; n < N_filter; n++)
      preallocated_values.push_back(std::vector<double>(M));

    for (int n = 0; n < N_filter; n++)
      for (int m = 0; m < M; m++)
        preallocated_values[n][m] = -1.0;
  }
  
  void TearDown() { 
    delete writer_ptr;
  }

  int N_filter;
  int N;
  int M;
  stan::interface_callbacks::writer::filtered_values<std::vector<double> > *writer_ptr;
  std::vector<std::vector<double> > preallocated_values;
  std::vector<size_t> filter;
};

TEST_F(StanInterfaceWriter, filtered_values_vector_double) {
  stan::interface_callbacks::writer::filtered_values<std::vector<double> > 
    writer = *writer_ptr;
  std::vector<double> x(N);
  for (int n = 0; n < N; n++) {
    x.at(n) = n;
  }
  EXPECT_NO_THROW(writer(x));
  EXPECT_FLOAT_EQ(2.0, writer.x()[0][0]);
  EXPECT_FLOAT_EQ(1.0, writer.x()[1][0]);
  for (int n = 0; n < N_filter; n++) 
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
  for (int n = 0; n < N; n++) 
    x.push_back(10*n);
  EXPECT_NO_THROW(writer(x));
  
  EXPECT_FLOAT_EQ(2.0, writer.x()[0][0]);
  EXPECT_FLOAT_EQ(1.0, writer.x()[1][0]);
  EXPECT_FLOAT_EQ(20.0, writer.x()[0][1]);
  EXPECT_FLOAT_EQ(10.0, writer.x()[1][1]);
  for (int m = 2; m < M; m++)
    for (int n = 0; n < N_filter; n++) 
      EXPECT_FLOAT_EQ(0.0, writer.x()[n][m]);

  x.clear();
  for (int n = 0; n < N; n++) 
    x.push_back(100*n);
  EXPECT_NO_THROW(writer(x));

  EXPECT_FLOAT_EQ(2.0, writer.x()[0][0]);
  EXPECT_FLOAT_EQ(1.0, writer.x()[1][0]);
  EXPECT_FLOAT_EQ(20.0, writer.x()[0][1]);
  EXPECT_FLOAT_EQ(10.0, writer.x()[1][1]);
  EXPECT_FLOAT_EQ(200.0, writer.x()[0][2]);
  EXPECT_FLOAT_EQ(100.0, writer.x()[1][2]);
  
  EXPECT_NO_THROW(writer(x));
  EXPECT_NO_THROW(writer(x));
  EXPECT_THROW(writer(x), std::out_of_range);
}

TEST_F(StanInterfaceWriter, filtered_values_vector_string) {
  stan::interface_callbacks::writer::filtered_values<std::vector<double> > 
    writer = *writer_ptr;

  std::vector<std::string> y;

  y.push_back("abc");
  y.push_back("def");
  EXPECT_NO_THROW(writer(y));
  
  y.clear();
  y.push_back("ghi");
  EXPECT_NO_THROW(writer(y));
}

TEST_F(StanInterfaceWriter, filtered_values_string) {
  stan::interface_callbacks::writer::filtered_values<std::vector<double> > 
    writer = *writer_ptr;

  std::string x;

  x = "abcd";
  EXPECT_NO_THROW(writer(x));
}

TEST_F(StanInterfaceWriter, filtered_values_noargs) {
  stan::interface_callbacks::writer::filtered_values<std::vector<double> > 
    writer = *writer_ptr;

  EXPECT_NO_THROW(writer());
  EXPECT_NO_THROW(writer());
}

TEST_F(StanInterfaceWriter, filtered_values_is_writing) {
  stan::interface_callbacks::writer::filtered_values<std::vector<double> > 
    writer = *writer_ptr;

  std::vector<double> x(N);
  for (int n = 0; n < N; n++) 
    x.at(n) = n;

  for (int m = 0; m < M; m++) {
    EXPECT_TRUE(writer.is_writing());
    writer(x);
  }
  EXPECT_FALSE(writer.is_writing());
}
