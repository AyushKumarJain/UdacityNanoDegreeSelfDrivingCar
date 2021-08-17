/*
 Command-line option parser tests.

 Copyright (C) 2014 AMPL Optimization Inc

 Permission to use, copy, modify, and distribute this software and its
 documentation for any purpose and without fee is hereby granted,
 provided that the above copyright notice appear in all copies and that
 both that the copyright notice and this permission notice and warranty
 disclaimer appear in supporting documentation.

 The author and AMPL Optimization Inc disclaim all warranties with
 regard to this software, including all implied warranties of
 merchantability and fitness.  In no event shall the author be liable
 for any special, indirect or consequential damages or any damages
 whatsoever resulting from loss of use, data or profits, whether in an
 action of contract, negligence or other tortious action, arising out
 of or in connection with the use or performance of this software.

 Author: Victor Zverovich
 */

#include "gmock/gmock.h"
#include "gtest-extra.h"
#include "mp/option.h"

using testing::Return;

TEST(OptionTest, OptionError) {
  const char *message = "don't panic!";
  mp::OptionError error(message);
  EXPECT_STREQ(message, error.what());
}

TEST(OptionTest, EmptyOptionList) {
  mp::OptionList options;
  EXPECT_EQ(options.begin(), options.end());
  EXPECT_EQ(0, options.Find('c'));
  EXPECT_TRUE(options.sorted());
  options.Sort();
  EXPECT_TRUE(options.sorted());
}

struct MockHandler {
  MOCK_METHOD0(OnOption1, bool ());
  MOCK_METHOD0(OnOption2, bool ());
};

#define EXPECT_OPTION(option, name_, description_, handler_) \
  EXPECT_EQ(name_, (option).name); \
  EXPECT_STREQ(description_, (option).description); \
  EXPECT_EQ(static_cast<MockHandler*>(handler_), (option).handler)

TEST(OptionTest, BuildOptionList) {
  mp::OptionList options;
  testing::StrictMock<MockHandler> handler;
  mp::OptionList::Builder<MockHandler> builder(options, handler);
  builder.Add<&MockHandler::OnOption1>('a', "test option 1");
  builder.Add<&MockHandler::OnOption2>('b', "test option 2");
  mp::OptionList::iterator i = options.begin();
  EXPECT_OPTION(*i, 'a', "test option 1", &handler);
  EXPECT_CALL(handler, OnOption1());
  i->on_option(i->handler);
  ++i;
  EXPECT_OPTION(*i, 'b', "test option 2", &handler);
  EXPECT_CALL(handler, OnOption2());
  i->on_option(i->handler);
  ++i;
  EXPECT_EQ(i, options.end());
}

TEST(OptionTest, SortOptionList) {
  mp::OptionList options;
  testing::StrictMock<MockHandler> handler;
  mp::OptionList::Builder<MockHandler> builder(options, handler);
  EXPECT_TRUE(options.sorted());
  options.Sort();
  EXPECT_TRUE(options.sorted());
  builder.Add<&MockHandler::OnOption1>('b', "test option 1");
  EXPECT_FALSE(options.sorted());
  options.Sort();
  EXPECT_TRUE(options.sorted());
  builder.Add<&MockHandler::OnOption2>('a', "test option 2");
  EXPECT_FALSE(options.sorted());
  options.Sort();
  EXPECT_TRUE(options.sorted());
  mp::OptionList::iterator i = options.begin();
  EXPECT_OPTION(*i, 'a', "test option 2", &handler);
  EXPECT_CALL(handler, OnOption2());
  i->on_option(i->handler);
  ++i;
  EXPECT_OPTION(*i, 'b', "test option 1", &handler);
  EXPECT_CALL(handler, OnOption1());
  i->on_option(i->handler);
}

TEST(OptionTest, FindInOptionList) {
  mp::OptionList options;
  MockHandler handler;
  EXPECT_EQ(0, options.Find('a'));
  mp::OptionList::Builder<MockHandler> builder(options, handler);
  builder.Add<&MockHandler::OnOption1>('a', "test option 1");
  EXPECT_DEBUG_DEATH(options.Find('a'), "Assertion");
  builder.Add<&MockHandler::OnOption1>('b', "test option 2");
  builder.Add<&MockHandler::OnOption1>('c', "test option 3");
  options.Sort();
  for (char c = 'a'; c <= 'c'; ++c)
    EXPECT_EQ(c, options.Find(c)->name);
  EXPECT_EQ(0, options.Find('d'));
}

TEST(OptionTest, ParseOptions) {
  mp::OptionList options;
  testing::StrictMock<MockHandler> handler;
  mp::OptionList::Builder<MockHandler> builder(options, handler);
  builder.Add<&MockHandler::OnOption1>('a', "test option 1");
  builder.Add<&MockHandler::OnOption2>('b', "test option 2");
  char arg0[] = "-b";
  char arg1[] = "-a";
  char *args[] = {arg0, arg1, 0}, **argp = args;
  testing::InSequence sequence;
  EXPECT_CALL(handler, OnOption2()).WillOnce(Return(true));
  EXPECT_CALL(handler, OnOption1()).WillOnce(Return(true));
  EXPECT_EQ(0, ParseOptions(argp, options));
}

TEST(OptionTest, StopParseOptions) {
  mp::OptionList options;
  testing::StrictMock<MockHandler> handler;
  mp::OptionList::Builder<MockHandler> builder(options, handler);
  builder.Add<&MockHandler::OnOption1>('a', "test option 1");
  builder.Add<&MockHandler::OnOption2>('b', "test option 2");
  char arg0[] = "-b";
  char arg1[] = "-a";
  char *args[] = {arg0, arg1, 0}, **argp = args;
  EXPECT_CALL(handler, OnOption2()).WillOnce(Return(false));
  EXPECT_EQ('b', ParseOptions(argp, options));
}

TEST(OptionTest, InvalidOption) {
  mp::OptionList options;
  testing::StrictMock<MockHandler> handler;
  mp::OptionList::Builder<MockHandler> builder(options, handler);
  builder.Add<&MockHandler::OnOption1>('a', "test option 1");
  char arg0[] = "-bad";
  char *args[] = {arg0, 0}, **argp = args;
  EXPECT_THROW_MSG(ParseOptions(argp, options), mp::OptionError,
                   "invalid option '-bad'");
}
