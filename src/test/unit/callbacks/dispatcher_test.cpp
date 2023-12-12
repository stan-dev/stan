#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/info_type.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <string>

struct deleter_noop {
  template <typename T>
  constexpr void operator()(T* arg) const {}
};



class CallbacksDispatcher : public ::testing::Test {
 public:
  CallbacksDispatcher()
      : ss_json(), ss_csv(),
        writer_json(std::unique_ptr<std::stringstream, deleter_noop>(&ss_json)),
        writer_csv(std::unique_ptr<std::stringstream, deleter_noop>(&ss_csv)) {}

  void SetUp() {
    ss_json.str(std::string());
    ss_json.clear();
    ss_csv.str(std::string());
    ss_csv.clear();
  }

  void TearDown() {}

  std::stringstream ss_json;
  std::stringstream ss_csv;
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_json;
  // this should be csv, but for now, need structured csv writer JSON
  stan::callbacks::json_writer<std::stringstream, deleter_noop> writer_csv;
};

bool is_whitespace(char c) { return c == ' ' || c == '\n'; }

std::string output_sans_whitespace(std::stringstream& ss) {
  auto out = ss.str();
  out.erase(std::remove_if(out.begin(), out.end(), is_whitespace), out.end());
  return out;
}

TEST_F(CallbacksDispatcher, output_config) {
  stan::callbacks::dispatcher  dp;

  std::shared_ptr<stan::callbacks::structured_writer> writer_json_ptr =
      std::make_shared<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(std::move(writer_json));

  std::shared_ptr<stan::callbacks::structured_writer> writer_csv_ptr =
      std::make_shared<stan::callbacks::json_writer<std::stringstream, deleter_noop>>(std::move(writer_csv));

  dp.addWriter(stan::callbacks::info_type::METRIC, std::move(writer_json_ptr));
  dp.addWriter(stan::callbacks::info_type::DRAW_CONSTRAINED, std::move(writer_csv_ptr));

  dp.begin_record(stan::callbacks::info_type::METRIC);
  dp.write(stan::callbacks::info_type::METRIC, "metric_type", "unit_e");
  dp.end_record(stan::callbacks::info_type::METRIC);

  auto out = output_sans_whitespace(ss_json);
  EXPECT_EQ("{\"metric_type\":\"unit_e\"}", out);
}
