#include <gtest/gtest.h>

#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/json_parser.hpp>

void test_rtl_2_ltr(size_t idx_rtl, 
                    size_t idx_ltr,
                    const std::vector<size_t>& dims) {
  stan::json::vars_map_r vars_r;
  stan::json::vars_map_i vars_i;
  stan::json::json_data_handler handler(vars_r, vars_i);
  size_t idx = handler.convert_offset_rtl_2_ltr(idx_rtl,dims);
  EXPECT_EQ(idx, idx_ltr);
}

void test_exception(size_t idx_rtl, 
                    const std::string& exception_text,
                    const std::vector<size_t>& dims) {
  stan::json::vars_map_r vars_r;
  stan::json::vars_map_i vars_i;
  stan::json::json_data_handler handler(vars_r, vars_i);
  try {
    handler.convert_offset_rtl_2_ltr(idx_rtl,dims);
  } catch (const std::exception& e) {
    EXPECT_EQ(e.what(), exception_text);
    return;
  }
  FAIL();  // didn't throw an exception as expected.
}


TEST(ioJson,rtl_2_ltr_1) {
  std::vector<size_t> dims(1);
  dims[0] = 7;
  test_rtl_2_ltr(0,0,dims);
  test_rtl_2_ltr(1,1,dims);
  test_rtl_2_ltr(2,2,dims);
  test_rtl_2_ltr(3,3,dims);
  test_rtl_2_ltr(4,4,dims);
  test_rtl_2_ltr(5,5,dims);
  test_rtl_2_ltr(6,6,dims);
}

//  row major:
//  11 12 13 14 21 22 23 24
//  column major:
//  11 21 12 22 13 23 14 24

TEST(ioJson,rtl_2_ltr_2) {
  std::vector<size_t> dims(2);
  dims[0] = 2;
  dims[1] = 4;
  test_rtl_2_ltr(0,0,dims);
  test_rtl_2_ltr(1,2,dims);
  test_rtl_2_ltr(2,4,dims);
  test_rtl_2_ltr(3,6,dims);
  test_rtl_2_ltr(4,1,dims);
  test_rtl_2_ltr(5,3,dims);
  test_rtl_2_ltr(6,5,dims);
  test_rtl_2_ltr(7,7,dims);
}

TEST(ioJson,rtl_2_ltr_err_1) {
  std::vector<size_t> dims(1);
  dims[0] = 7;
  test_exception(7,"variable: , unexpected error",dims);
}

TEST(ioJson,rtl_2_ltr_err_2) {
  std::vector<size_t> dims(2);
  dims[0] = 2;
  dims[1] = 4;
  test_exception(8,"variable: , unexpected error",dims);
}

TEST(ioJson,rtl_2_ltr_err_3n) {
  std::vector<size_t> dims(2);
  dims[0] = 2;
  dims[1] = 4;
  test_exception(11,"variable: , unexpected error",dims);
}
