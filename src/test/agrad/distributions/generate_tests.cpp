#include <test/agrad/distributions/utility.hpp>
#include <ostream>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <boost/algorithm/string.hpp>

template <class T>
std::ostream& operator<< (std::ostream& o, std::vector<T>& vec) {
  o << "vector size: " << vec.size() << std::endl;
  for (size_t n = 0; n < vec.size(); n++) {
    o << "  \'" << vec[n] << "\'" << std::endl;
  }
  return o;
}

std::vector<std::string> lookup_argument(const std::string& argument) {
  using boost::iequals;
  std::vector<std::string> args;
  if (iequals(argument, "int")) {
    args.push_back("int");
  } else if (iequals(argument, "ints")) {
    args.push_back("int");
    args.push_back("std::vector<int>");
    args.push_back("Eigen::Matrix<int, Eigen::Dynamic, -1>");
    args.push_back("Eigen::Matrix<int, -1, Eigen::Dynamic>");
  } else if (iequals(argument, "double")) {
    args.push_back("double");
  } else if (iequals(argument, "doubles")) {
    args.push_back("double");
    args.push_back("std::vector<double>");
    args.push_back("Eigen::Matrix<double, Eigen::Dynamic, -1>");
    args.push_back("Eigen::Matrix<double, -1, Eigen::Dynamic>");
  }
  return args;
}

void write_includes(std::ostream& out, const std::string& include) {
  out << "#include <gtest/gtest.h>" << std::endl;
  out << "#include <boost/mpl/vector.hpp>" << std::endl;
  out << "#include <stan/agrad/agrad.hpp>" << std::endl;
  out << "#include <test/agrad/distributions/new_distribution_test_fixture.hpp>" << std::endl;
  out << "#include <" << include.substr(include.find("src/")+4) << ">" << std::endl;  
  out << std::endl;
}

std::vector<std::string> tokenize_arguments(const std::string& arguments) {
  std::vector<std::string> tokens;
  std::string delimiters = ", ";
  std::string args_only_string = arguments.substr(arguments.find(":") + 1);
  boost::algorithm::trim(args_only_string);
  boost::algorithm::split(tokens, args_only_string, 
			  boost::is_any_of(delimiters), 
			  boost::token_compress_on);
  return tokens;
}

size_t size(const std::vector<std::vector<std::string> >& sequences) {
  if (sequences.size() == 0)
    return 0;
  size_t N = 1;
  for (size_t n = 0; n < sequences.size(); n++) 
    N *= sequences[n].size();
  return N;
}

bool is_argument_list(const std::string& line) {
  size_t comment = line.find("//");
  if (comment == std::string::npos)
    return false;
  size_t keyword = line.find("Arguments:", comment+1);
  if (keyword == std::string::npos)
    return false;
  return true;
}

std::string read_arguments_from_file(const std::string& in_name) {
  std::string arguments;
  std::ifstream in(in_name.c_str());
  if (in.is_open() && in.good()) {
    std::getline (in, arguments);
    while (in.good() && !is_argument_list(arguments)) {
      std::getline (in, arguments);
    }
    in.close();
  }
  if (!is_argument_list(arguments))
    arguments = "";
  return arguments;
}

std::string read_test_name_from_file(const std::string& in_name) {
  std::string test_name = "";
  std::ifstream in(in_name.c_str());
  
  std::string file;
  in.seekg(0, std::ios::end);
  file.resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&file[0], file.size());
  in.close();
  
  size_t pos = 0;
  std::string keyword = "class ";
  while (test_name == "" && pos < file.size()) {
    pos = file.find(keyword, pos) + keyword.size();
    test_name = file.substr(pos, file.find(":", pos)-pos);
    boost::algorithm::trim(test_name);
  }
  return test_name;
}

std::string read_fixture_name_from_file(const std::string& in_name) {
  std::string fixture_name = "";
  std::ifstream in(in_name.c_str());
  
  std::string file;
  in.seekg(0, std::ios::end);
  file.resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&file[0], file.size());
  in.close();
  
  size_t pos = 0;
  std::string keyword = "class ";
  std::string keyword2 = "public ";
  while (fixture_name == "" && pos < file.size()) {
    pos = file.find(keyword, pos) + keyword.size();
    pos = file.find(":", pos);
    pos = file.find(keyword2, pos) + keyword2.size();
    fixture_name = file.substr(pos, file.find("{", pos)-pos);
    boost::algorithm::trim(fixture_name);
    fixture_name += "Fixture";
  }
  return fixture_name;
}


std::vector<std::vector<std::string> > build_argument_sequence(const std::string& arguments) {
  std::vector<std::string> argument_list = tokenize_arguments(arguments);
  std::vector<std::vector<std::string> > argument_sequence;
  for (size_t n = 0; n < argument_list.size(); n++)
    argument_sequence.push_back(lookup_argument(argument_list[n]));
  return argument_sequence;
}

void write_typedef(std::ostream& out, std::string base, size_t& N, std::vector<std::vector<std::string> > argument_sequence, const size_t depth) {
  std::vector<std::string> args = argument_sequence.front();
  argument_sequence.erase(argument_sequence.begin());
  if (argument_sequence.size() > 0) {
    for (size_t n = 0; n < args.size(); n++)
      write_typedef(out, base + args[n] + ", ", N, argument_sequence, depth);
  } else {
    std::string extra_args;
    for (size_t n = depth; n < 10; n++) {
      extra_args += ", empty";
    }
    for (size_t n = 0; n < args.size(); n++) {
      out << "typedef boost::mpl::vector<" << base << args[n] << extra_args;
      if (extra_args.size() == 0)
	out << " ";
      out << "> type_" << N << ";" << std::endl;
      N++;
    }
  }
}

void write_types(std::ostream& out, const std::string& test_name, const std::vector<std::vector<std::string> >& argument_sequence) {
  size_t N = 0;
  write_typedef(out, test_name + ", ", N, argument_sequence, argument_sequence.size());
  out << std::endl;
}

void write_test(std::ostream& out, const std::string& test_name, const std::string& fixture_name, const size_t N) {
  for (size_t n = 0; n < N; n++)
    out << "INSTANTIATE_TYPED_TEST_CASE_P(" << test_name << "_" << n << ", " << fixture_name << ", " << "type_" << n << ");" << std::endl;
}

void write_test_cases(std::ostream& out, const std::string& in_name) {
  std::string arguments = read_arguments_from_file(in_name);
  std::string test_name = read_test_name_from_file(in_name);
  std::string fixture_name = read_fixture_name_from_file(in_name);
  std::vector<std::vector<std::string> > argument_sequence = build_argument_sequence(arguments);

  write_types(out, test_name, argument_sequence); 
  write_test(out, test_name, fixture_name, size(argument_sequence));
}

/** 
 * Generate test cases.
 * 
 * @param argc Number of arguments
 * @param argv Arguments. Should contain one argument with a filename.
 * 
 * @return 0 for success, negative number otherwise.
 */
int main(int argc, const char* argv[]) {
  if (argc != 2)
    return -1;
  int return_code = 0;
  std::string in_suffix = "_test.hpp";
  std::string out_suffix = "_generated_test.cpp";

  std::string in_name = argv[1];
  
  size_t last_in_suffix = in_name.find_last_of(in_suffix) + 1 - in_suffix.length();
  std::string out_name = in_name.substr(0, last_in_suffix) + out_suffix;
  
  std::ofstream out(out_name.c_str());
  write_includes(out, in_name);
  write_test_cases(out, in_name);

  out.close();
  
  return 0;
}
