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
    
    // help options: the options described when typing
    //   model/command help
    help_options.push_back("id=<>");
    help_options.push_back("data=<>");
    help_options.push_back("init=<>");
    help_options.push_back("test_gradient");
    help_options.push_back("random");
    help_options.push_back("output");
    help_options.push_back("method=<>");

  }
  std::string model_path;
  std::vector<std::string> help_options;
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
                                    const int& start) {
  if (start > lines.size()) {
    ADD_FAILURE() << start 
                  << " is greater than the size of the lines: " 
                  << lines.size();
  }
  std::vector<std::string> block;
  for (int n = start; n < lines.size(); n++) {
    block.push_back(lines[n]);
    if (lines[n] == "") {
      return(block);
    }
  }
  return(block);
}

TEST_F(ModelCommand, check_help_options) {
  int line_number = 0;
  std::string help_command = model_path + " help";
  std::vector<std::string> output;
  std::vector<std::string> block;
  
  // run: "model/command help"
  output = split_lines(run_command(help_command));
  
  // usage output
  block = next_block(output, line_number);
  ASSERT_EQ(2, block.size());
  EXPECT_EQ(0, block[0].find("Usage"))
    << "line " << line_number + 0 << ": "
    << "expecting 'Usage' line. Found: "
    << block[0];
  EXPECT_EQ("", block[1]);
  line_number += block.size();
  
  // valid arguments
  block = next_block(output, line_number);
  ASSERT_EQ(2, block.size());
  EXPECT_EQ(0, block[0].find("Valid arguments"))
    << "line " << line_number + 0 << ": "
    << "expecting 'Valid arguments' line. Found: "
    << block[0];
  EXPECT_EQ("", block[1]);
  line_number += block.size();

  block = next_block(output, line_number);
  ASSERT_EQ(help_options.size(), block.size()-1)
    << "the block should match the help options";
  for (int n = 0; n < help_options.size(); n++) {
    EXPECT_EQ(2, block[n].find(help_options[n]))
      << "line " << line_number + n << ": "
      << "expecting '" << help_options[n] << "' line. Found: "
      << block[n];
  }
  EXPECT_EQ("", block[block.size()-1]);
  line_number += block.size();

  block = next_block(output, line_number);
  ASSERT_TRUE(block.size() > 1)
    << "for \"See 'model'\": expecting at least 3 lines";
  EXPECT_EQ(0, block[0].find("See 'model'"))
    << "line " << line_number + 0 << ": "
    << "expecting \"See 'model'\" line. Found: "
    << block[0];
  line_number += block.size();
  
  EXPECT_EQ(line_number, output.size()) 
    << "there should be no more lines of output";
}

