#include <stan/io/program_reader.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

std::vector<std::string> create_search_path() {
  std::vector<std::string> search_path;
  search_path.push_back("foo");
  search_path.push_back("src/test/test-models/included/");
  search_path.push_back("bar/baz");
  return search_path;
}

void expect_eq_traces(const std::vector<std::pair<std::string, int> >& e,
                         const std::vector<std::pair<std::string, int> >& f) {
  EXPECT_EQ(e.size(), f.size());
  for (size_t i = 0; i < e.size(); ++i) {
    EXPECT_EQ(e[i].first, f[i].first);
    EXPECT_EQ(e[i].second, f[i].second);
  }
}

void expect_trace(stan::io::program_reader& reader, int pos,
                  const std::string& path1, int pos1) {
  using std::pair;
  using std::string;
  using std::vector;
  vector<pair<string, int> > expected;
  expected.push_back(pair<string, int>(path1, pos1));
  vector<pair<string, int> > found = reader.trace(pos);
  expect_eq_traces(expected, found);
}
void expect_trace(stan::io::program_reader& reader, int pos,
                  const std::string& path1, int pos1,
                  const std::string& path2, int pos2) {
  using std::pair;
  using std::string;
  using std::vector;
  vector<pair<string, int> > expected;
  expected.push_back(pair<string, int>(path1, pos1));
  expected.push_back(pair<string, int>(path2, pos2));
  vector<pair<string, int> > found = reader.trace(pos);
  expect_eq_traces(expected, found);
}
void expect_trace(stan::io::program_reader& reader, int pos,
                  const std::string& path1, int pos1,
                  const std::string& path2, int pos2,
                  const std::string& path3, int pos3) {
  using std::pair;
  using std::string;
  using std::vector;
  vector<pair<string, int> > expected;
  expected.push_back(pair<string, int>(path1, pos1));
  expected.push_back(pair<string, int>(path2, pos2));
  expected.push_back(pair<string, int>(path3, pos3));
  vector<pair<string, int> > found = reader.trace(pos);
  expect_eq_traces(expected, found);
}

TEST(progr_reader, trimComment) {
  using stan::io::program_reader;
  EXPECT_EQ("", program_reader::trim_comment(""));
  EXPECT_EQ(" ", program_reader::trim_comment(" "));
  EXPECT_EQ("", program_reader::trim_comment("//"));
  EXPECT_EQ(" ", program_reader::trim_comment(" //"));
  EXPECT_EQ("  #include foo.stan",
            program_reader::trim_comment("  #include foo.stan//"));
  EXPECT_EQ("  #include foo.stan ",
            program_reader::trim_comment("  #include foo.stan //"));
  EXPECT_EQ("  #include foo.stan ",
            program_reader::trim_comment("  #include foo.stan //"));
  EXPECT_EQ("  #include foo.stan ",
            program_reader::trim_comment("  #include foo.stan //blah"));
  EXPECT_EQ("  #include foo.stan ",
            program_reader::trim_comment("  #include foo.stan // blah blah"));
  EXPECT_EQ("  #include foo.stan",
            program_reader::trim_comment("  #include foo.stan// blah blah"));

}

TEST(prog_reader, one) {
  using std::pair;
  using std::string;
  using stan::io::program_reader;

  std::stringstream ss;
  ss << "parameters {\n"            // 1
     << "  real y;\n"               // 2
     << "}\n"                       // 3
     << "model {\n"                 // 4
     << "  y ~ normal(0, 1);\n"     // 5
     << "}\n"                       // 6
     << "";                         // 7 (nothing on line)

  std::vector<std::string> search_path = create_search_path();
  stan::io::program_reader reader(ss, "foo", search_path);

  EXPECT_EQ("parameters {\n"
            "  real y;\n"
            "}\n"
            "model {\n"
            "  y ~ normal(0, 1);\n"
            "}\n",
            reader.program());
  // program is 6 lines, but spirit qi line_pos_iterator will go further
  for (int i = 1; i < 9; ++i)
    expect_trace(reader, i, "foo", i);

  EXPECT_THROW(reader.trace(0), std::runtime_error);
  EXPECT_THROW(reader.trace(9), std::runtime_error);
}



TEST(prog_reader, two) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"                // 1
     << "#include incl_fun.stan\n"     // 2
     << "}\n"                          // 3
     << "#include incl_params.stan\n"  // 4
     << "model {\n"                    // 5
     << "}\n";                         // 6

  vector<string> search_path = create_search_path();

  stan::io::program_reader reader(ss, "foo", search_path);

  EXPECT_EQ("functions {\n"            // 1
            "  int foo() {\n"          // 2 foo, 1 include
            "    return 1;\n"          // 3, 2 foo, 2 include
            "  }\n"                    // 4, 2 foo, 3 include
            "}\n"                      // 5, 3 foo
            "parameters {\n"           // 6, 4 foo, 1 include
            "  real y;\n"              // 7, 4 foo, 2 include
            "}\n"                      // 8, 4 foo, 3 indluce
            "model {\n"                // 9, 5 foo
            "}\n",                     // 10, 6 foo
            reader.program());

  expect_trace(reader, 1, "foo", 1);
  expect_trace(reader, 2, "foo", 2, "incl_fun.stan", 1);
  expect_trace(reader, 3, "foo", 2, "incl_fun.stan", 2);
  expect_trace(reader, 4, "foo", 2, "incl_fun.stan", 3);
  expect_trace(reader, 5, "foo", 3);
  expect_trace(reader, 6, "foo", 4, "incl_params.stan", 1);
  expect_trace(reader, 7, "foo", 4, "incl_params.stan", 2);
  expect_trace(reader, 8, "foo", 4, "incl_params.stan", 3);
  expect_trace(reader, 9, "foo", 5);
  expect_trace(reader, 10, "foo", 6);
  expect_trace(reader, 11, "foo", 7);  // padding
  expect_trace(reader, 12, "foo", 8);  // padding

  EXPECT_THROW(reader.trace(0), std::runtime_error);
  EXPECT_THROW(reader.trace(13), std::runtime_error);
}


TEST(prog_reader, three) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"              // 1
     << "#include incl_rec.stan\n"   // 2
     << "}\n"                        // 3
     << "model { }\n";               // 4

  vector<string> search_path = create_search_path();

  stan::io::program_reader reader(ss, "foo", search_path);

  EXPECT_EQ("functions {\n"          // 1, foo 1
            "parameters {\n"         // 2, foo 2, incl 1
            "real y;\n"              // 3, foo 2, incl 2
            "real z;\n"              // 4, foo 2, incl 3
            "}\n"                    // 5, foo 2, incl 4
            "transformed parameters {\n"  // 6
            "  real w = y + z;\n"         // 7
            "}\n"                         // 8, foo 2
            "}\n"                         // 9, foo 3
            "model { }\n",                // 10, foo 4
            reader.program());

  expect_trace(reader, 1, "foo", 1);
  expect_trace(reader, 2, "foo", 2, "incl_rec.stan", 1);
  expect_trace(reader, 3, "foo", 2, "incl_rec.stan", 2, "incl_nested.stan", 1);
  expect_trace(reader, 4, "foo", 2, "incl_rec.stan", 2, "incl_nested.stan", 2);
  expect_trace(reader, 5, "foo", 2, "incl_rec.stan", 3);
  expect_trace(reader, 6, "foo", 2, "incl_rec.stan", 4);
  expect_trace(reader, 7, "foo", 2, "incl_rec.stan", 5);
  expect_trace(reader, 8, "foo", 2, "incl_rec.stan", 6);
  expect_trace(reader, 9, "foo", 3);
  expect_trace(reader, 10, "foo", 4);
  expect_trace(reader, 11, "foo", 5);  // padding
  expect_trace(reader, 12, "foo", 6);  // padding

  EXPECT_THROW(reader.trace(0), std::runtime_error);
  EXPECT_THROW(reader.trace(13), std::runtime_error);
}

TEST(prog_reader, four) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"                  // 1
     << "#include incl_fun.stan\n"  // 2
     << "}\n"                            // 3
     << "#include incl_params.stan// comment should be OK\n"    // 4
     << "model {\n"                      // 5
     << "}\n";                           // 6

  vector<string> search_path = create_search_path();

  stan::io::program_reader reader(ss, "foo", search_path);

  EXPECT_EQ("functions {\n"            // 1
            "  int foo() {\n"          // 2 foo, 1 include
            "    return 1;\n"          // 3, 2 foo, 2 include
            "  }\n"                    // 4, 2 foo, 3 include
            "}\n"                      // 5, 3 foo
            "parameters {\n"           // 6, 4 foo, 1 include
            "  real y;\n"              // 7, 4 foo, 2 include
            "}\n"                      // 8, 4 foo, 3 indluce
            "model {\n"                // 9, 5 foo
            "}\n",                     // 10, 6 foo
            reader.program());

  expect_trace(reader, 1, "foo", 1);
  expect_trace(reader, 2, "foo", 2, "incl_fun.stan", 1);
  expect_trace(reader, 3, "foo", 2, "incl_fun.stan", 2);
  expect_trace(reader, 4, "foo", 2, "incl_fun.stan", 3);
  expect_trace(reader, 5, "foo", 3);
  expect_trace(reader, 6, "foo", 4, "incl_params.stan", 1);
  expect_trace(reader, 7, "foo", 4, "incl_params.stan", 2);
  expect_trace(reader, 8, "foo", 4, "incl_params.stan", 3);
  expect_trace(reader, 9, "foo", 5);
  expect_trace(reader, 10, "foo", 6);
  expect_trace(reader, 11, "foo", 7);  // padding
  expect_trace(reader, 12, "foo", 8);  // padding

  EXPECT_THROW(reader.trace(0), std::runtime_error);
  EXPECT_THROW(reader.trace(13), std::runtime_error);
}


TEST(prog_reader, ignoreRecursive) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"
     << "#include badrecurse1.stan\n"
     << "}\n"
     << "model { }\n";
  vector<string> search_path = create_search_path();
  stan::io::program_reader reader(ss, "foo", search_path);
  EXPECT_EQ("functions {\n}\nmodel { }\n", reader.program());
}
TEST(prog_reader, ignoreRecursive2) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"
     << "#include badrecurse2.stan\n"
     << "}\n"
     << "model { }\n";
  vector<string> search_path = create_search_path();
  stan::io::program_reader reader(ss, "foo", search_path);
  EXPECT_EQ("functions {\n}\nmodel { }\n", reader.program());
}
TEST(prog_reader, allowSequential) {
  using std::vector;
  using std::string;
  std::stringstream ss;
  ss << "functions {\n"
     << "#include simple1.stan\n"
     << "#include simple1.stan\n"
     << "}\n"
     << "model { }\n";
  vector<string> search_path = create_search_path();
  stan::io::program_reader reader(ss, "foo", search_path);
  EXPECT_EQ("functions {\n// foo\n// foo\n}\nmodel { }\n", reader.program());
}
