#include <gtest/gtest.h>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <stan/interface_callbacks/writer/psql_writer.hpp>

class StanInterfaceCallbacksPQXXWriter: public ::testing::Test {
public:
  StanInterfaceCallbacksPQXXWriter() :
    conn(new pqxx::connection())  {
      pqxx::work TT(*conn, "clean_tables");
      TT.exec("DROP TABLE IF EXISTS runs CASCADE; DROP TABLE IF EXISTS key_value;" 
        "DROP TABLE IF EXISTS messages; DROP TABLE IF EXISTS parameter_names; DROP TABLE IF EXISTS parameter_samples;");
      TT.commit();
      writer = new stan::interface_callbacks::writer::psql_writer("","TEST");
  }

  ~StanInterfaceCallbacksPQXXWriter() {
    delete conn;
  }

  void SetUp() {
  }
  void TearDown() { }

  stan::interface_callbacks::writer::psql_writer* writer;
  pqxx::connection* conn;
};

TEST_F(StanInterfaceCallbacksPQXXWriter, key_double) {
  double in, out;
  in = 5.2;
  EXPECT_NO_THROW((*writer)("key_double_value_test", in));
  pqxx::work T(*conn, "check_key_double");
  pqxx::result R = T.exec("SELECT * FROM key_value WHERE key = 'key_double_value_test';");
  R[0]["double"].to(out);
  EXPECT_DOUBLE_EQ(in, out); 
  T.commit();
  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_int) {
  int in, out;
  in = 5;
  EXPECT_NO_THROW((*writer)("key_integer_value_test", in));
  pqxx::work T(*conn, "check_key_integer");
  pqxx::result R = T.exec("SELECT * FROM key_value WHERE key = 'key_integer_value_test';");
  R[0]["integer"].to(out);
  EXPECT_EQ(in, out); 
  T.commit();
  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_string) {
  std::string in, out;
  in = "five";
  EXPECT_NO_THROW((*writer)("key_string_value_test", in));
  pqxx::work T(*conn, "check_key_string");
  pqxx::result R = T.exec("SELECT * FROM key_value WHERE key = 'key_string_value_test';");
  R[0]["string"].to(out);
  EXPECT_EQ(in, out); 
  T.commit();
  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_vector) {
  const int N = 5;
  double x[N];
  double out;
  for (int n = 0; n < N; ++n) x[n] = n;

  EXPECT_NO_THROW((*writer)("key_vector_idx_value_test", x, N));
  conn->prepare("check_key_vector_idx_value", "SELECT * FROM key_value WHERE "
    "key = 'key_vector_idx_value_test' AND idx = $1;");
  for (int n = 0; n < N; ++n) {
    pqxx::work T(*conn, "check_key_vector");
    pqxx::result R = T.prepared("check_key_vector_idx_value")(n).exec();
    R[0]["double"].to(out);
    EXPECT_EQ(x[n], out);
    T.commit();
  }

  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, key_matrix) {
  const int n_rows = 3;
  const int n_cols = 2;
  double x[n_rows * n_cols];
  double out;
  for (int i = 0; i < n_rows; ++i)
    for (int j = 0; j < n_cols; ++j)
      x[i * n_cols + j] = i - j;

  EXPECT_NO_THROW((*writer)("key_matrix_idx_value_test", x, n_rows, n_cols));
  conn->prepare("check_key_matrix_idx_value", "SELECT * FROM key_value WHERE "
    "key = 'key_matrix_idx_value_test' AND row_idx = $1 AND col_idx = $2;");

  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      pqxx::work T(*conn, "check_key_matrix");
      pqxx::result R = T.prepared("check_key_matrix_idx_value")(i)(j).exec();
      R[0]["double"].to(out);
      EXPECT_EQ(x[i * n_cols + j], out);
      T.commit();
    }
  }
  delete writer;

}

TEST_F(StanInterfaceCallbacksPQXXWriter, string_vector) {
  const int N = 5;
  std::string out;
  std::vector<std::string> x;
    for (int n = 0; n < N; ++n)
      x.push_back("NAME " + boost::lexical_cast<std::string>(n));
  EXPECT_NO_THROW((*writer)(x));

  pqxx::work T(*conn, "check_name");
  pqxx::result R = T.exec("SELECT name FROM parameter_names;");
  for (int n = 0; n < N; ++n) {
    R[n]["name"].to(out);
    EXPECT_EQ(x[n], out);
  }
  T.commit();
  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, double_vector) {
  const int N = 1000;
  std::vector<std::string> s;
  std::vector<double> x;
  for (int n = 0; n < N; ++n) 
    x.push_back(n);
  for (int n = 0; n < N; ++n)
    s.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW((*writer)(s));
  for (int i = 0; i < 100; ++i) {
    EXPECT_NO_THROW((*writer)(x));
  }
  delete writer;
  conn->prepare("check_parameter_value", "SELECT * FROM parameter_samples WHERE "
    "name = $1 ORDER BY iteration;");
  std::string name_out;
  double value_out;
  int iter_out;
  for (int n = 0; n < N; ++n) {
    pqxx::work T(*conn, "check_parameter");
    pqxx::result R = T.prepared("check_parameter_value")(n).exec();
    for (int i = 0; i < 100; ++i) {
      R[i]["name"].to(name_out);
      EXPECT_EQ(boost::lexical_cast<std::string>(n), name_out);
      R[i]["value"].to(value_out);
      EXPECT_DOUBLE_EQ(n, value_out);
      R[i]["iteration"].to(iter_out);
      EXPECT_EQ(i+1, iter_out);
    }
    T.commit();
  }

}

TEST_F(StanInterfaceCallbacksPQXXWriter, double_vector_100k) {
  const int N = 10;
  std::vector<std::string> s;
  std::vector<double> x;
  for (int n = 0; n < N; ++n) 
    x.push_back(n);
  for (int n = 0; n < N; ++n)
    s.push_back(boost::lexical_cast<std::string>(n));

  EXPECT_NO_THROW((*writer)(s));
  for (int i = 0; i < 10; ++i) {
    EXPECT_NO_THROW((*writer)(x));
  }
  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, null) {
  EXPECT_NO_THROW((*writer)());
  delete writer;
}

TEST_F(StanInterfaceCallbacksPQXXWriter, string) {
  std::string in = "message, I have a message for you!";
  std::string out;
  EXPECT_NO_THROW((*writer)(in));
  pqxx::work T(*conn, "check_message");
  pqxx::result R = T.exec("SELECT * FROM messages;");
  R[0]["message"].to(out);
  EXPECT_EQ(in, out); 
  T.commit();
  delete writer;
}
