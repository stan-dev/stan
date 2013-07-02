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

std::vector<std::string> next_argument(const std::vector<std::string>& arguments,
                                       const int& start) {
  if (start > arguments.size()) {
    ADD_FAILURE() << start 
                  << " is greater than the size of the arguments: " 
                  << arguments.size();
  }
  std::vector<std::string> argument;
  for (int n = start; n < arguments.size(); n++) {
    argument.push_back(arguments[n]);
    if (arguments[n] == "") {
      return(argument);
    }
  }
  return(argument);
}

TEST_F(ModelCommand, check_help_options) {
  int line_number = 0;
  std::string help_command = model_path + " help";
  std::vector<std::string> output;
  std::vector<std::string> argument;
  
  // run: "model/command help"
  output = split_lines(run_command(help_command));
  
  // usage output
  argument = next_argument(output, line_number);
  ASSERT_EQ(2, argument.size());
  EXPECT_EQ(0, argument[0].find("Usage"))
    << "line " << line_number + 0 << ": "
    << "expecting 'Usage' line. Found: "
    << argument[0];
  EXPECT_EQ("", argument[1]);
  line_number += argument.size();
  
  // valid arguments
  argument = next_argument(output, line_number);
  ASSERT_EQ(2, argument.size());
  EXPECT_EQ(0, argument[0].find("Valid arguments"))
    << "line " << line_number + 0 << ": "
    << "expecting 'Valid arguments' line. Found: "
    << argument[0];
  EXPECT_EQ("", argument[1]);
  line_number += argument.size();
  
  for (int n = 0; n < help_options.size(); n++) {
    argument = next_argument(output, line_number);
    ASSERT_TRUE(argument.size() > 2)
      << "for '" << help_options[n] << "': expecting at least 3 lines";
    EXPECT_EQ(0, argument[0].find(help_options[n]))
      << "line " << line_number + 0 << ": "
      << "expecting '" << help_options[n] << "' line. Found: "
      << argument[0];
    EXPECT_EQ("", argument[argument.size()-1]);
    line_number += argument.size();
  }

  argument = next_argument(output, line_number);
  ASSERT_TRUE(argument.size() > 1)
    << "for \"See 'model'\": expecting at least 3 lines";
  EXPECT_EQ(0, argument[0].find("See 'model'"))
    << "line " << line_number + 0 << ": "
    << "expecting \"See 'model'\" line. Found: "
    << argument[0];
  line_number += argument.size();
  
  EXPECT_EQ(line_number, output.size()) 
    << "there should be no more lines of output";
}

