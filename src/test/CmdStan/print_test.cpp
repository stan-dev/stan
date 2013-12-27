#include <stan/command/print.hpp>
#include <gtest/gtest.h>
#include <test/CmdStan/models/utility.hpp>

TEST(CommandPrint, next_index_1d) {
  std::vector<int> dims(1);
  std::vector<int> index(1,1);
  dims[0] = 100;

  ASSERT_EQ(1U, index.size());
  EXPECT_EQ(1, index[0]);
  for (int n = 1; n <= 100; n++) {
    if (n == 1)
      continue;
    next_index(index, dims);
    ASSERT_EQ(1U, index.size());
    EXPECT_EQ(n, index[0]);
  }
  
  index[0] = 100;
  EXPECT_THROW(next_index(index, dims), std::domain_error);
  
  index[0] = 1000;
  EXPECT_THROW(next_index(index, dims), std::domain_error);
}

TEST(CommandPrint, next_index_2d) {
  std::vector<int> dims(2);
  std::vector<int> index(2,1);
  dims[0] = 100;
  dims[1] = 3;

  ASSERT_EQ(2U, index.size());
  EXPECT_EQ(1, index[0]);
  EXPECT_EQ(1, index[1]);
  for (int i = 1; i <= 100; i++) 
    for (int j = 1; j <= 3; j++) {
      if (i == 1 && j == 1)
        continue;
      next_index(index, dims);
      ASSERT_EQ(2U, index.size());
      EXPECT_EQ(i, index[0]);
      EXPECT_EQ(j, index[1]);
    }
  
  index[0] = 100;
  index[1] = 3;
  EXPECT_THROW(next_index(index, dims), std::domain_error);
  
  index[0] = 1000;
  index[1] = 1;
  EXPECT_THROW(next_index(index, dims), std::domain_error);

  index[0] = 10;
  index[1] = 4;
  EXPECT_NO_THROW(next_index(index, dims))
    << "this will correct the index and set the next element to (11,1)";
  EXPECT_EQ(11, index[0]);
  EXPECT_EQ(1, index[1]);
}


TEST(CommandPrint, matrix_index_1d) {
  std::vector<int> dims(1);
  std::vector<int> index(1,1);
  dims[0] = 100;
  
  EXPECT_EQ(0, matrix_index(index, dims));
  
  index[0] = 50;
  EXPECT_EQ(49, matrix_index(index, dims));
  
  index[0] = 100;
  EXPECT_EQ(99, matrix_index(index, dims));
  
  index[0] = 0;
  EXPECT_THROW(matrix_index(index, dims), std::domain_error);

  index[0] = 101;
  EXPECT_THROW(matrix_index(index, dims), std::domain_error);
}

TEST(CommandPrint, matrix_index_2d) {
  std::vector<int> dims(2);
  std::vector<int> index(2,1);
  dims[0] = 100;
  dims[1] = 3;
  
  EXPECT_EQ(0, matrix_index(index, dims));
  
  index[0] = 50;
  index[1] = 1;
  EXPECT_EQ(49, matrix_index(index, dims));

  index[0] = 100;
  index[1] = 1;
  EXPECT_EQ(99, matrix_index(index, dims));

  index[0] = 1;
  index[1] = 2;
  EXPECT_EQ(100, matrix_index(index, dims));

  index[0] = 1;
  index[1] = 3;
  EXPECT_EQ(200, matrix_index(index, dims));

  index[0] = 100;
  index[1] = 3;
  EXPECT_EQ(299, matrix_index(index, dims));

  index[0] = 1;
  index[1] = 0;
  EXPECT_THROW(matrix_index(index, dims), std::domain_error);

  index[0] = 0;
  index[1] = 1;
  EXPECT_THROW(matrix_index(index, dims), std::domain_error);

  index[0] = 101;
  index[1] = 1;
  EXPECT_THROW(matrix_index(index, dims), std::domain_error);

  index[0] = 1;
  index[1] = 4;
  EXPECT_THROW(matrix_index(index, dims), std::domain_error);
}

TEST(CommandPrint, functional_test__issue_342) {
  std::string path_separator;
  path_separator.push_back(get_path_separator());
  std::string command = "bin" + path_separator + "print";
  std::string csv_file 
    = "src" + path_separator 
    + "test" + path_separator
    + "CmdStan" + path_separator
    + "print_samples" + path_separator
    + "matrix_output.csv";

  run_command_output out = run_command(command + " " + csv_file);
  ASSERT_FALSE(out.hasError) 
    << "\"" << out.command << "\" quit with an error";
}
