#include <stan/io/json/json_data.hpp>
#include <stan/io/json/json_data_handler.hpp>
#include <stan/io/json/json_error.hpp>
#include <stan/io/json/json_handler.hpp>
#include <stan/io/json/rapidjson_parser.hpp>

#include <boost/limits.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

#include <complex>


// void test_tuple_var(stan::json::json_data &jdata, const std::string &text,
//                       const std::string &name,
//                     // pass in maps - vars_i and vars_r
//                       const std::vector<std::string>> &expected_slots,
//                     ///////////// stopped here
//                     ) {
//   return;
// }



TEST(ioJson, jsonData_tuple_int_real) {
  std::string txt = "{ \"foo\" : 3.214, "                               \
                    "\"bar\" : { \"1\" : 1, \"2\" : 6.428 } }";
  std::cout << txt << std::endl;
  std::stringstream in(txt);
  stan::json::json_data jdata(in);
  EXPECT_EQ(true, jdata.contains_r("foo"));
  EXPECT_EQ(true, jdata.contains_r("bar.1"));
  EXPECT_EQ(true, jdata.contains_r("bar.2"));
}
