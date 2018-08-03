#include <stan/services/sample/standalone_gqs.hpp>
#include <stan/callbacks/stream_writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <test/test-models/good/services/bym2_offset_only.hpp>
#include <test/unit/services/instrumented_callbacks.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <iostream>

typedef bym2_offset_only_model_namespace::bym2_offset_only_model model_class;

TEST(ServicesStandaloneGQ_big, genDraws_bym) {
  stan::test::unit::instrumented_interrupt interrupt;
  std::stringstream model_log;
  std::stringstream sample_ss, logger_ss;
  stan::callbacks::stream_writer sample_writer(sample_ss, "");
  stan::callbacks::stream_logger logger(logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss,
                                        logger_ss);

  // Get var_context for data
  std::fstream data_stream("src/test/test-models/good/services/bym2_inputs.data.R",
                           std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  // Instantiate model with data
  model_class model(data_var_context);

  int num_params = stan::services::num_constrained_params(model);
  EXPECT_EQ(num_params, 1420);

  // Get param data from dumpfile matrix 1000 rows, 1420 columns
  // using stan "var_context" - PITA!
  std::fstream params_stream("src/test/test-models/good/services/bym2_draws.data.R",
                           std::fstream::in);
  stan::io::dump params_var_context(params_stream);
  params_stream.close();
  std::vector<std::vector<double>> cdraws(1000);
  try {
    params_var_context.validate_dims("get params", "bym2_draws", "matrix",
                                     params_var_context.to_vec(1420, 1000));
    std::vector<double> bym2_pars_vals = params_var_context.vals_r("bym2_draws");
    int idx = 0;
    for (int i = 0; i < 1000; ++i) {
      std::vector<double> tmp(1420);
      for (int j = 0; j < 1420; ++j) {
        tmp[j] = bym2_pars_vals[idx++];
      }
      cdraws[i] = tmp;
    }
   } catch (const std::exception& e) {
     logger.error("Cannot read draws from file.");
     logger.error("Caught exception: ");
     logger.error(e.what());
     std::cerr << logger_ss.str() << std::endl;
     throw std::domain_error("Initialization failure");
   }
  
  stan::services::standalone_generate(model,
                                      cdraws,
                                      12345,
                                      interrupt,
                                      logger,
                                      sample_writer);
  // verify results...
  // std::ofstream gqs_file;
  // gqs_file.open ("src/test/test-models/good/services/bym2_gqs_v2.csv");
  // gqs_file << sample_ss.str();
  // gqs_file.close();
}
