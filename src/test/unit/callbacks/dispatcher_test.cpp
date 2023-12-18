#include <stan/callbacks/csv_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/info_type.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <string>

class CallbacksDispatcher : public ::testing::Test {
 public:
  CallbacksDispatcher()
      : ss_metric(),
        ss_draw_cnstrn(),
        writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric)),
        writer_draw_cnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_cnstrn)) {
  }

  void SetUp() {
    ss_metric.str(std::string());
    ss_metric.clear();
    ss_draw_cnstrn.str(std::string());
    ss_draw_cnstrn.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
        = std::make_shared<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(std::move(writer_metric));
    std::shared_ptr<stan::callbacks::structured_writer> writer_draw_cnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream, deleter_noop>>(std::move(writer_draw_cnstrn));
    dp.add_writer(stan::callbacks::struct_info_type::INV_METRIC, std::move(writer_metric_ptr));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_CONSTRAIN, std::move(writer_draw_cnstrn_ptr));
  }

  void TearDown() {}

  std::stringstream ss_metric;
  std::stringstream ss_draw_cnstrn;
  stan::callbacks::dispatcher  dp;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_cnstrn;
};


TEST_F(CallbacksDispatcher, write_metric_draw) {
  dp.begin_record(stan::callbacks::struct_info_type::INV_METRIC);
  dp.write(stan::callbacks::struct_info_type::INV_METRIC, "metric_type", "unit_e");
  //  dp.write(stan::callbacks::struct_info_type::INV_METRIC, "inv_metric", eigen vector, all values 1.0
  dp.end_record(stan::callbacks::struct_info_type::INV_METRIC);

  auto out_metric = output_sans_whitespace(ss_metric);
  EXPECT_EQ("{\"metric_type\":\"unit_e\"}", out_metric);

  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  dp.table_header(stan::callbacks::table_info_type::DRAW_CONSTRAIN, header);
  dp.table_row(stan::callbacks::table_info_type::DRAW_CONSTRAIN, values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss_draw_cnstrn.str());
}

TEST_F(CallbacksDispatcher, write_noops) {
  EXPECT_NO_THROW(dp.begin_record(stan::callbacks::struct_info_type::RUN_CONFIG));
  EXPECT_NO_THROW(dp.end_record(stan::callbacks::struct_info_type::RUN_CONFIG));
  EXPECT_NO_THROW(dp.write(stan::callbacks::struct_info_type::RUN_CONFIG, "foo", "bar"));
  ASSERT_TRUE(ss_metric.str().empty());
  ASSERT_TRUE(ss_draw_cnstrn.str().empty());
}

TEST_F(CallbacksDispatcher, csv_no_header) {
  std::vector<double> values = {1, 2, 3};
  dp.table_row(stan::callbacks::table_info_type::DRAW_CONSTRAIN, values);
  ASSERT_TRUE(ss_draw_cnstrn.str().empty());
}

TEST_F(CallbacksDispatcher, csv_mult_header) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  dp.table_header(stan::callbacks::table_info_type::DRAW_CONSTRAIN, header);
  dp.table_header(stan::callbacks::table_info_type::DRAW_CONSTRAIN, header);
  dp.table_row(stan::callbacks::table_info_type::DRAW_CONSTRAIN, values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss_draw_cnstrn.str());
}
