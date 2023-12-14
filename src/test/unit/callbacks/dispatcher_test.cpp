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
        ss_draws_cnstrn(),
        writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric)),
        writer_draws_cnstrn(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draws_cnstrn)) {
  }

  void SetUp() {
    ss_metric.str(std::string());
    ss_metric.clear();
    ss_draws_cnstrn.str(std::string());
    ss_draws_cnstrn.clear();
    std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
        = std::make_shared<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(std::move(writer_metric));
    std::shared_ptr<stan::callbacks::structured_writer> writer_draws_cnstrn_ptr
        = std::make_shared<stan::callbacks::csv_writer<std::stringstream, deleter_noop>>(std::move(writer_draws_cnstrn));
    dp.add_writer(stan::callbacks::info_type::METRIC, std::move(writer_metric_ptr));
    dp.add_writer(stan::callbacks::info_type::DRAW_CONSTRAINED, std::move(writer_draws_cnstrn_ptr));
  }

  void TearDown() {}

  std::stringstream ss_metric;
  std::stringstream ss_draws_cnstrn;
  stan::callbacks::dispatcher  dp;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draws_cnstrn;
};


TEST_F(CallbacksDispatcher, write_metric_draws) {
  dp.begin_record(stan::callbacks::info_type::METRIC);
  dp.write(stan::callbacks::info_type::METRIC, "metric_type", "unit_e");
  //  dp.write(stan::callbacks::info_type::METRIC, "inv_metric", eigen vector, all values 1.0
  dp.end_record(stan::callbacks::info_type::METRIC);

  auto out_metric = output_sans_whitespace(ss_metric);
  EXPECT_EQ("{\"metric_type\":\"unit_e\"}", out_metric);

  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  dp.table_header(stan::callbacks::info_type::DRAW_CONSTRAINED, header);
  dp.table_row(stan::callbacks::info_type::DRAW_CONSTRAINED, values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss_draws_cnstrn.str());
}

TEST_F(CallbacksDispatcher, write_noops) {
  EXPECT_NO_THROW(dp.begin_record(stan::callbacks::info_type::CONFIG));
  EXPECT_NO_THROW(dp.end_record(stan::callbacks::info_type::CONFIG));
  EXPECT_NO_THROW(dp.write(stan::callbacks::info_type::CONFIG, "foo", "bar"));
  ASSERT_TRUE(ss_metric.str().empty());
  ASSERT_TRUE(ss_draws_cnstrn.str().empty());
}

TEST_F(CallbacksDispatcher, csv_no_header) {
  std::vector<double> values = {1, 2, 3};
  dp.table_row(stan::callbacks::info_type::DRAW_CONSTRAINED, values);
  ASSERT_TRUE(ss_draws_cnstrn.str().empty());
}

TEST_F(CallbacksDispatcher, csv_mult_header) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  dp.table_header(stan::callbacks::info_type::DRAW_CONSTRAINED, header);
  dp.table_header(stan::callbacks::info_type::DRAW_CONSTRAINED, header);
  dp.table_row(stan::callbacks::info_type::DRAW_CONSTRAINED, values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss_draws_cnstrn.str());
}
