#include <stan/io/csv_writer.hpp>
#include <gtest/gtest.h>

TEST(io_csv_writer, csv_writer_int) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  writer.write(5);
  writer.write(2);
  writer.write(-12);
  EXPECT_EQ("5,2,-12",s.str());
  writer.newline();
  EXPECT_EQ("5,2,-12\n",s.str());
  writer.write(3);
  writer.write(4);
  writer.newline();
  EXPECT_EQ("5,2,-12\n3,4\n",s.str());
}

TEST(io_csv_writer, csv_writer_double) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  writer.write(1.2);
  writer.write(5.7);
  writer.newline();
  EXPECT_EQ("1.2,5.7\n",s.str());
}

TEST(io_csv_writer, csv_writer_vector) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  Eigen::Matrix<double,Eigen::Dynamic,1> v1(2);
  v1 << 1.2, 5.7;
  Eigen::Matrix<double,Eigen::Dynamic,1> v2(2);
  v2 << 5.7, 1.2;
  writer.write(v1);
  writer.write(v2);
  writer.newline();
  EXPECT_EQ("1.2,5.7,5.7,1.2\n",s.str());
}

TEST(io_csv_writer, csv_writer_row_vector) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  Eigen::Matrix<double,1,Eigen::Dynamic> v1(2);
  v1 << 1.2, 5.7;
  Eigen::Matrix<double,1,Eigen::Dynamic> v2(2);
  v2 << 5.7, 1.2;
  writer.write(v1);
  writer.write(v2);
  writer.newline();
  EXPECT_EQ("1.2,5.7,5.7,1.2\n",s.str());
}


TEST(io_csv_writer, csv_writer_matrix_row_major) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m(2,3);
  m << 1.2, 5.7, 1.0,
       2.0, 3.0, 4.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> v2(1,2);
  v2 << 5.7, 1.2;
  writer.write_row_major(m);
  writer.write_row_major(v2);
  writer.newline();
  EXPECT_EQ("1.2,5.7,1,2,3,4,5.7,1.2\n", s.str());
}
TEST(io_csv_writer, csv_writer_matrix_col_major) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m(2,3);
  m << 1.2, 5.7, 1.0,
       2.0, 3.0, 4.0;
  writer.write_col_major(m);
  writer.newline();
  EXPECT_EQ("1.2,2,5.7,3,1,4\n",s.str());
}
TEST(io_csv_writer, csv_writer_matrix) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m(2,3);
  m << 1.2, 5.7, 1.0,
       2.0, 3.0, 4.0;
  writer.write(m);
  writer.newline();
  EXPECT_EQ("1.2,2,5.7,3,1,4\n",s.str());
}
TEST(io_csv_writer, csv_writer_strings) {
  std::stringstream s;
  stan::io::csv_writer writer(s);
  writer.write("foo");
  writer.write("bar");
  EXPECT_EQ("\"foo\",\"bar\"",s.str());
  writer.newline();
  EXPECT_EQ("\"foo\",\"bar\"\n",s.str());
}



