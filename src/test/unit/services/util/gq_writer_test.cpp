#include <stan/services/util/gq_writer.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <sstream>
#include <test/test-models/good/services/test_gq.hpp>
#include <stan/io/empty_var_context.hpp>
#include <stan/services/util/create_rng.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>


class ServicesUtilGQWriter : public testing::Test {
public:
  ServicesUtilGQWriter()
    : model(context, &model_log) {}
  std::stringstream model_log;
  stan::io::empty_var_context context;
  std::stringstream sample_ss, logger_ss;
  stan_model model;
};

TEST_F(ServicesUtilGQWriter, t1) {
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  stan::callbacks::stream_logger logger(logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss);
  stan::services::util::gq_writer writer(sample_writer,logger,2);
  writer.write_gq_names(model);
  // model test_gq.stan gen quantities block has 3 params:  xqg, y_rep.1, y_rep.2
  EXPECT_EQ(count_matches("xgq",sample_ss.str()),1);
  EXPECT_EQ(count_matches("y_rep",sample_ss.str()),2);
}

TEST_F(ServicesUtilGQWriter, t2) {
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  stan::callbacks::stream_logger logger(logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss);
  boost::ecuyer1988 rng1 = stan::services::util::create_rng(0, 1);
  std::vector<double> draw;
  draw.push_back(-2.345);
  draw.push_back(-6.789);
  stan::services::util::gq_writer writer(sample_writer,logger,2);
  writer.write_gq_values(model, rng1, draw);
  // model test_gq.stan generates 3 values, 2 commas
  EXPECT_EQ(count_matches(",",sample_ss.str()), 2);
}
