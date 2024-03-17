#include <stan/callbacks/csv_writer.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/info_type.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/table_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <string>

class CallbacksDispatcher : public ::testing::Test {
 public:
  CallbacksDispatcher() :
      ss_draw_sample(),
      ss_draw_warmup(),
      ss_uparams_sample(),
      ss_uparams_warmup(),
      ss_params_init(),
      ss_algo(),
      ss_metric(),
      ss_timing(),
      writer_draw_sample(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_sample)),
      writer_draw_warmup(std::unique_ptr<std::stringstream, deleter_noop>(&ss_draw_warmup)),
      writer_uparams_sample(std::unique_ptr<std::stringstream, deleter_noop>(&ss_uparams_sample)),
      writer_uparams_warmup(std::unique_ptr<std::stringstream, deleter_noop>(&ss_uparams_warmup)),
      writer_params_init(std::unique_ptr<std::stringstream, deleter_noop>(&ss_params_init)),
      writer_algo(std::unique_ptr<std::stringstream, deleter_noop>(&ss_algo)),
      writer_metric(std::unique_ptr<std::stringstream, deleter_noop>(&ss_metric)),
      writer_timing(std::unique_ptr<std::stringstream, deleter_noop>(&ss_timing)) {

    std::shared_ptr<stan::callbacks::table_writer> writer_draw_sample_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_draw_sample));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_SAMPLE,
                  std::move(writer_draw_sample_ptr));

    std::shared_ptr<stan::callbacks::table_writer> writer_draw_warmup_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_draw_warmup));
    dp.add_writer(stan::callbacks::table_info_type::DRAW_WARMUP,
                  std::move(writer_draw_warmup_ptr));

    std::shared_ptr<stan::callbacks::table_writer> writer_uparams_sample_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_uparams_sample));
    dp.add_writer(stan::callbacks::table_info_type::UPARAMS_SAMPLE,
                  std::move(writer_uparams_sample_ptr));

    std::shared_ptr<stan::callbacks::table_writer> writer_uparams_warmup_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_uparams_warmup));
    dp.add_writer(stan::callbacks::table_info_type::UPARAMS_WARMUP,
                  std::move(writer_uparams_warmup_ptr));

    std::shared_ptr<stan::callbacks::table_writer> writer_params_init_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_params_init));
    dp.add_writer(stan::callbacks::table_info_type::PARAMS_INIT,
                  std::move(writer_params_init_ptr));

    std::shared_ptr<stan::callbacks::table_writer> writer_algo_ptr
        = std::make_shared<stan::callbacks::csv_writer<
          std::stringstream, deleter_noop>>(std::move(writer_algo));
    dp.add_writer(stan::callbacks::table_info_type::ALGO_STATE,
                  std::move(writer_algo_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_metric_ptr
        = std::make_shared<stan::callbacks::json_writer<
          std::stringstream, deleter_noop>>(std::move(writer_metric));
    dp.add_writer(stan::callbacks::struct_info_type::INV_METRIC,
                  std::move(writer_metric_ptr));

    std::shared_ptr<stan::callbacks::structured_writer> writer_timing_ptr
        = std::make_shared<stan::callbacks::json_writer<
          std::stringstream, deleter_noop>>(std::move(writer_timing));
    dp.add_writer(stan::callbacks::struct_info_type::RUN_TIMING,
                  std::move(writer_timing_ptr));
  }

  void SetUp() {
    ss_draw_sample.str(std::string());
    ss_draw_sample.clear();

    ss_draw_warmup.str(std::string());
    ss_draw_warmup.clear();

    ss_uparams_sample.str(std::string());
    ss_uparams_sample.clear();

    ss_uparams_warmup.str(std::string());
    ss_uparams_warmup.clear();

    ss_params_init.str(std::string());
    ss_params_init.clear();

    ss_algo.str(std::string());
    ss_algo.clear();

    ss_metric.str(std::string());
    ss_metric.clear();

    ss_timing.str(std::string());
    ss_timing.clear();
  }

  void TearDown() {}

  std::stringstream ss_draw_sample;
  std::stringstream ss_draw_warmup;
  std::stringstream ss_uparams_sample;
  std::stringstream ss_uparams_warmup;
  std::stringstream ss_params_init;
  std::stringstream ss_algo;
  std::stringstream ss_metric;
  std::stringstream ss_timing;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_sample;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_draw_warmup;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_uparams_sample;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_uparams_warmup;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_params_init;
  stan::callbacks::csv_writer<std::stringstream, deleter_noop> writer_algo;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_metric;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_timing;
  stan::callbacks::dispatcher dp;
};


TEST_F(CallbacksDispatcher, write_metric_draw) {
  Eigen::VectorXd inv_metric =  Eigen::VectorXd::Ones(4);
  dp.begin_record(stan::callbacks::struct_info_type::INV_METRIC);
  dp.write(stan::callbacks::struct_info_type::INV_METRIC, "metric_type", "unit_e");
  dp.write(stan::callbacks::struct_info_type::INV_METRIC, "inv_metric", inv_metric);
  dp.end_record(stan::callbacks::struct_info_type::INV_METRIC);

  auto out_metric = output_sans_whitespace(ss_metric);
  EXPECT_EQ("{\"metric_type\":\"unit_e\",\"inv_metric\":[1,1,1,1]}", out_metric);
}

TEST_F(CallbacksDispatcher, write_noops) {
  EXPECT_NO_THROW(dp.begin_record(stan::callbacks::struct_info_type::RUN_CONFIG));
  EXPECT_NO_THROW(dp.end_record(stan::callbacks::struct_info_type::RUN_CONFIG));
  EXPECT_NO_THROW(dp.write(stan::callbacks::struct_info_type::RUN_CONFIG, "foo", "bar"));
  ASSERT_TRUE(ss_metric.str().empty());
  ASSERT_TRUE(ss_draw_sample.str().empty());
}

TEST_F(CallbacksDispatcher, csv_mult_header) {
  std::vector<std::string> header = {"mu", "sigma", "theta"};
  std::vector<double> values = {1, 2, 3};
  dp.write_header(stan::callbacks::table_info_type::DRAW_SAMPLE, header);
  dp.write_flat(stan::callbacks::table_info_type::DRAW_SAMPLE, values);
  EXPECT_EQ("mu, sigma, theta\n1, 2, 3\n", ss_draw_sample.str());
}
