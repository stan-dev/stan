#ifndef STAN_SERVICES_SAMPLE_STANDALONE_GQS_HPP
#define STAN_SERVICES_SAMPLE_STANDALONE_GQS_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/math/prim.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/gq_writer.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace stan {
namespace services {

/**
 * Given a set of draws from a fitted model, generate corresponding
 * quantities of interest which are written to callback writer.
 * Matrix of draws consists of one row per draw, one column per parameter.
 * Draws are processed one row at a time.
 * Return code indicates success or type of error.
 *
 * @tparam Model model class
 * @param[in] model instantiated model
 * @param[in] draws sequence of draws of constrained parameters
 * @param[in] seed seed to use for randomization
 * @param[in, out] interrupt called every iteration
 * @param[in, out] logger logger to which to write warning and error messages
 * @param[in, out] sample_writer writer to which draws are written
 * @return error code
 */
template <class Model>
int standalone_generate(const Model &model, const Eigen::MatrixXd &draws,
                        unsigned int seed, callbacks::interrupt &interrupt,
                        callbacks::logger &logger,
                        callbacks::writer &sample_writer) {
  if (draws.size() == 0) {
    logger.error("Empty set of draws from fitted model.");
    return error_codes::DATAERR;
  }

  std::vector<std::string> p_names;
  model.constrained_param_names(p_names, false, false);
  std::vector<std::string> gq_names;
  model.constrained_param_names(gq_names, false, true);
  if (!(p_names.size() < gq_names.size())) {
    logger.error("Model doesn't generate any quantities of interest.");
    return error_codes::CONFIG;
  }

  std::stringstream msg;
  if (p_names.size() != draws.cols()) {
    msg << "Wrong number of parameter values in draws from fitted model.  ";
    msg << "Expecting " << p_names.size() << " columns, ";
    msg << "found " << draws.cols() << " columns.";
    std::string msgstr = msg.str();
    logger.error(msgstr);
    return error_codes::DATAERR;
  }
  util::gq_writer writer(sample_writer, logger, p_names.size());
  writer.write_gq_names(model);

  stan::rng_t rng = util::create_rng(seed, 1);

  std::vector<double> unconstrained_params_r;
  std::vector<double> row(draws.cols());
  try {
    for (size_t i = 0; i < draws.rows(); ++i) {
      Eigen::Map<Eigen::VectorXd>(&row[0], draws.cols()) = draws.row(i);
      try {
        model.unconstrain_array(row, unconstrained_params_r, &msg);
      } catch (const std::exception &e) {
        if (msg.str().length() > 0)
          logger.error(msg);
        logger.error(e.what());
        return error_codes::DATAERR;
      }
      interrupt();  // call out to interrupt and fail
      writer.write_gq_values(model, rng, unconstrained_params_r);
    }
  } catch (const std::exception &e) {
    logger.error(e.what());
    return error_codes::SOFTWARE;
  }
  return error_codes::OK;
}

/**
 * Given a set of draws from a fitted model, generate corresponding
 * quantities of interest which are written to callback writer.
 * Matrix of draws consists of one row per draw, one column per parameter.
 * Draws are processed one row at a time.
 * Return code indicates success or type of error.
 *
 * @tparam Model model class
 * @tparam SampleWriter type of sample writer
 * @param[in] model instantiated model
 * @param[in] num_chains number of chains
 * @param[in] draws standard vector containing sequence of draws of constrained
 * parameters
 * @param[in] seed seed to use for randomization
 * @param[in, out] interrupt called every iteration
 * @param[in, out] logger logger to which to write warning and error messages
 * @param[in, out] sample_writers A vector of writers to which draws for each
 * chain are written
 * @return error code
 */
template <typename Model, typename SampleWriter>
int standalone_generate(const Model &model, const int num_chains,
                        const std::vector<Eigen::MatrixXd> &draws,
                        unsigned int seed, callbacks::interrupt &interrupt,
                        callbacks::logger &logger,
                        std::vector<SampleWriter> &sample_writers) {
  if (num_chains == 1) {
    return standalone_generate(model, draws[0], seed, interrupt, logger,
                               sample_writers[0]);
  }

  std::vector<std::string> p_names;
  model.constrained_param_names(p_names, false, false);
  std::vector<std::string> gq_names;
  model.constrained_param_names(gq_names, false, true);
  if (!(p_names.size() < gq_names.size())) {
    logger.error("Model doesn't generate any quantities of interest.");
    return error_codes::CONFIG;
  }
  std::vector<util::gq_writer> writers;
  writers.reserve(num_chains);
  std::vector<stan::rng_t> rngs;
  rngs.reserve(num_chains);
  for (int i = 0; i < num_chains; ++i) {
    if (draws[i].size() == 0) {
      logger.error("Empty set of draws from fitted model.");
      return error_codes::DATAERR;
    }
    if (p_names.size() != draws[i].cols()) {
      std::stringstream msg;
      msg << "Wrong number of parameter values in draws from fitted model.  ";
      msg << "Expecting " << p_names.size() << " columns, ";
      msg << "found " << draws[i].cols() << " columns in draws from chain " << i
          << ".";
      std::string msgstr = msg.str();
      logger.error(msgstr);
      return error_codes::DATAERR;
    }
    writers.emplace_back(sample_writers[i], logger, p_names.size());
    writers[i].write_gq_names(model);
    rngs.emplace_back(util::create_rng(seed, i + 1));
  }
  bool error_any = false;
  try {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_chains, 1),
        [&draws, &model, &logger, &interrupt, &writers, &rngs,
         &error_any](const tbb::blocked_range<size_t> &r) {
          Eigen::VectorXd unconstrained_params_r(draws[0].cols());
          Eigen::VectorXd row(draws[0].cols());
          std::stringstream msg;
          for (size_t slice_idx = r.begin(); slice_idx != r.end();
               ++slice_idx) {
            for (size_t i = 0; i < draws[slice_idx].rows(); ++i) {
              if (error_any)
                return;
              try {
                row = draws[slice_idx].row(i);
                model.unconstrain_array(row, unconstrained_params_r, &msg);
              } catch (const std::domain_error &e) {
                if (msg.str().length() > 0)
                  logger.error(msg);
                logger.error(e.what());
                error_any = true;
                return;
              }
              interrupt();  // call out to interrupt and fail
              writers[slice_idx].write_gq_values(model, rngs[slice_idx],
                                                 unconstrained_params_r);
            }
          }
        },
        tbb::simple_partitioner());
  } catch (const std::exception &e) {
    logger.error(e.what());
    return error_codes::SOFTWARE;
  }
  return error_any ? error_codes::DATAERR : error_codes::OK;
}

/**
 * DEPRECATED: This function assumes dimensions are rectangular,
 * a restriction which the Stan language may soon relax.
 *
 * Find the names, dimensions of the model parameters.
 * Assembles vectors of name, dimensions for the variables
 * declared in the parameters block.
 *
 * @tparam Model type of model
 * @param[in] model model to query
 * @param[in, out] param_names sequence of parameter names
 * @param[in, out] param_dimss sequence of variable dimensionalities
 */
template <class Model>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((deprecated))
#elif defined(_MSC_VER)
__declspec(deprecated)
#endif
void get_model_parameters(const Model &model,
                          std::vector<std::string> &param_names,
                          std::vector<std::vector<size_t>> &param_dimss) {
  std::vector<std::string> all_param_names;
  model.get_param_names(all_param_names, false, false);
  std::vector<std::vector<size_t>> dimss;
  model.get_dims(dimss, false, false);
  // remove zero-size
  for (size_t i = 0; i < all_param_names.size(); i++) {
    auto &v = dimss[i];
    if (std::find(v.begin(), v.end(), 0) == v.end()) {
      param_names.emplace_back(all_param_names[i]);
      param_dimss.emplace_back(dimss[i]);
    }
  }
}

}  // namespace services
}  // namespace stan
#endif
