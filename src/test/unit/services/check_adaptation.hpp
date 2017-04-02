#ifndef TEST__UNIT__CHECK_ADAPTATION_HPP
#define TEST__UNIT__CHECK_ADAPTATION_HPP

#include <test/unit/services/instrumented_callbacks.hpp>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

namespace stan {
  namespace test {
    namespace unit {

      void check_adaptation(const size_t& num_params,
                            const std::vector<double>& param_vals,
                            stan::test::unit::instrumented_writer& report,
                            const double& err_margin) {
        std::vector<std::string> param_strings = report.string_values();
        size_t offset = 0;
        for (size_t i = 0; i < param_strings.size(); i++) {
          offset++;
          if (param_strings[i].find("lements of inverse mass matrix:")
              != std::string::npos) {
            break;
          }
        }
        std::vector<std::string> strs;
        boost::split(strs, param_strings[offset], boost::is_any_of(", "), boost::token_compress_on);
        EXPECT_EQ(num_params, strs.size());
        for (size_t i = 0; i < num_params; i++) {
          ASSERT_NEAR(param_vals[i], std::stod(strs[i]), err_margin);
        }
      }

      void check_adaptation(const size_t& num_rows, const size_t& num_cols,
                            const std::vector<double>& param_vals,
                            stan::test::unit::instrumented_writer& report,
                            const double& err_margin) {
        std::vector<std::string> param_strings = report.string_values();
        size_t offset = 0;
        for (size_t i = 0; i < param_strings.size(); i++) {
          offset++;
          if (param_strings[i].find("lements of inverse mass matrix:")
              != std::string::npos) {
            break;
          }
        }
        for (size_t i = 0, ij = 0; i < num_rows; i++) {
          std::vector<std::string> strs;
          boost::split(strs, param_strings[offset+i], boost::is_any_of(", "), boost::token_compress_on);
          EXPECT_EQ(num_cols, strs.size());
          for (size_t j = 0; j < num_cols; j++, ij++) {
            ASSERT_NEAR(param_vals[ij], std::stod(strs[j]), err_margin);
          }
        }
      }

      void check_different(const size_t& num_params,
                           const std::vector<double>& param_vals,
                           stan::test::unit::instrumented_writer& report,
                            const double& margin) {
        std::vector<std::string> param_strings = report.string_values();
        size_t offset = 0;
        for (size_t i = 0; i < param_strings.size(); i++) {
          offset++;
          if (param_strings[i].find("lements of inverse mass matrix:")
              != std::string::npos) {
            break;
          }
        }
        std::vector<std::string> strs;
        boost::split(strs, param_strings[offset], boost::is_any_of(", "), boost::token_compress_on);
        EXPECT_EQ(num_params, strs.size());
        for (size_t i = 0; i < num_params; i++) {
          ASSERT_GT(fabs(param_vals[i]-std::stod(strs[i])), margin);
        }
      }

      void check_different(const size_t& num_rows, const size_t& num_cols,
                           const std::vector<double>& param_vals,
                           stan::test::unit::instrumented_writer& report,
                           const double& margin) {
        std::vector<std::string> param_strings = report.string_values();
        size_t offset = 0;
        for (size_t i = 0; i < param_strings.size(); i++) {
          offset++;
          if (param_strings[i].find("lements of inverse mass matrix:")
              != std::string::npos) {
            break;
          }
        }
        for (size_t i = 0, ij = 0; i < num_rows; i++) {
          std::vector<std::string> strs;
          boost::split(strs, param_strings[offset+i], boost::is_any_of(", "), boost::token_compress_on);
          EXPECT_EQ(num_cols, strs.size());
          for (size_t j = 0; j < num_cols; j++, ij++) {
            ASSERT_GT(fabs(param_vals[ij]-std::stod(strs[j])), margin);
          }
        }
      }

    }
  }
}

#endif
