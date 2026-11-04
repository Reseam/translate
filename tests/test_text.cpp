// Sentence splitting and mention repair (no models needed).
#include <cstdio>
#include <string>
#include <vector>

#include "text.h"

using namespace reseam::mt;

static int failures = 0;

static std::string show(const std::vector<Segment> &segs) {
  std::string s;
  for (const auto &g : segs) s += (g.translate ? "[" : "{") + std::string(g.text) + (g.translate ? "]" : "}");
  return s;
}

static void expect(const std::string &got, const std::string &want, const char *what) {
  if (got != want) {
    std::printf("FAIL %s\n  got:  %s\n  want: %s\n", what, got.c_str(), want.c_str());
    ++failures;
  }
}

int main() {
  // [translated] {verbatim}
  expect(show(splitSentences("Hola. Adiós.")), "[Hola.]{ }[Adiós.]", "two sentences");
  expect(show(splitSentences("e.g. this stays one")), "[e.g. this stays one]", "abbreviation before lowercase");
  expect(show(splitSentences("  Hola  \n\nAdiós")), "{  }[Hola]{  }{\n}{\n}[Adiós]", "spacing and line breaks");
  expect(show(splitSentences("Line one\r\nLine two")), "[Line one]{\r}{\n}[Line two]", "CRLF");
  expect(show(splitSentences("今日は暑い。明日は？")), "[今日は暑い。][明日は？]", "full-width terminators");
  expect(show(splitSentences("Wow!!! Really?")), "[Wow!!!]{ }[Really?]", "repeated terminators");
  expect(show(splitSentences("Mira esto https://t.co/abc123 increíble")), "[Mira esto]{ }{https://t.co/abc123}{ }[increíble]",
         "URL kept verbatim");
  expect(show(splitSentences("see https://x.y/z.")), "[see]{ }{https://x.y/z}[.]", "trailing period is not the URL's");
  expect(show(splitSentences("https://only.example")), "{https://only.example}", "URL only");
  expect(show(splitSentences("")), "", "empty");

  expect(restoreMentions("Gracias @usuario y @otro_1", "Thanks @user and @other_1"), "Thanks @usuario and @otro_1",
         "mentions restored in order");
  expect(restoreMentions("Hola @ana", "Hi @ana @bob"), "Hi @ana @bob", "count mismatch leaves translation alone");
  expect(restoreMentions("mail a@b.com", "mail a@b.com"), "mail a@b.com", "email is not a mention");
  expect(restoreMentions("no mentions", "@x"), "@x", "no source mentions");

  std::printf(failures ? "%d failures\n" : "text handling ok\n", failures);
  return failures != 0;
}
