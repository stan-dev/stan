// Want:
// 1. check options
// 2. iterate over all possible ways to call the model
//    - check output
//    - verify that it gets set

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <sstream>
#include <test/models/utility.hpp>


class ModelCommand : public testing::Test {
public:
  static void SetUpTestCase() {
    separator = get_path_separator();
  }

  static void TearDownTestCase() {
  }

  void SetUp() {
    model_path = "models" + separator + "command";

  }
  std::string model_path;
  static std::string separator;
};
std::string  ModelCommand::separator = "";

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  std::string line;
  while (std::getline(stream, line)) {
    lines.push_back(line);
  }
  return lines;
}

std::vector<std::string> next_block(const std::vector<std::string>& lines,
                                    const size_t& start) {
  if (start > lines.size()) {
    ADD_FAILURE() << start 
                  << " is greater than the size of the lines: " 
                  << lines.size();
  }
  std::vector<std::string> block;
  for (size_t n = start; n < lines.size(); n++) {
    block.push_back(lines[n]);
    if (lines[n] == "") {
      return(block);
    }
  }
  return(block);
}

TEST_F(ModelCommand, check_help_options) {
  size_t line_number = 0;
  std::string help_command = model_path + " help";
  std::vector<std::string> output;
  std::vector<std::string> block;
  
  // run: "model/command help"
  run_command_output out = run_command(help_command);
  output = split_lines(out.output);
  
  // Usage
  block = next_block(output, line_number);
  ASSERT_EQ(2U, block.size());
  EXPECT_EQ(0U, block[0].find("Usage"))
    << "line " << line_number + 0 << ": "
    << "expecting 'Usage' line. Found: "
    << block[0];
  EXPECT_EQ("", block[1]);
  line_number += block.size();
  
  // Method
  block = next_block(output, line_number);
  ASSERT_EQ(5U, block.size());
  EXPECT_EQ(0U, block[0].find("Begin by selecting"))
    << "line " << line_number + 0 << ": "
    << "expecting 'Begin by selecting' line. Found: "
    << block[0];
  EXPECT_EQ("", block[4]);
  line_number += block.size();

  // Help
  block = next_block(output, line_number);
  ASSERT_EQ(4U, block.size());
  EXPECT_EQ(0U, block[0].find("Or see help"))
  << "line " << line_number + 0 << ": "
  << "expecting 'Or see help' line. Found: "
  << block[0];
  EXPECT_EQ("", block[3]);
  line_number += block.size();
  
  // Configuration
  block = next_block(output, line_number);
  ASSERT_EQ(7U, block.size());
  EXPECT_EQ(0U, block[0].find("Additional configuration"))
  << "line " << line_number + 0 << ": "
  << "expecting 'Additional configuration' line. Found: "
  << block[0];
  EXPECT_EQ("", block[6]);
  line_number += block.size();
  
  // Footer
  block = next_block(output, line_number);
  ASSERT_EQ(2U, block.size());
  EXPECT_EQ(0U, block[0].find("See"))
  << "line " << line_number + 0 << ": "
  << "expecting 'See' line. Found: "
  << block[0];
  EXPECT_EQ("", block[1]);
  line_number += block.size();
  
  // Residual
  EXPECT_EQ(line_number, output.size())
    << "there should be no more lines of output";
}

