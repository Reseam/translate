// Translates stdin line by line: translate <models> <from> <to> [workers]
#include <iostream>

#include "engine.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: translate <models> <from> <to> [workers]\n";
    return 2;
  }
  reseam::mt::EngineOptions opt;
  opt.modelsDir = argv[1];
  opt.workers = argc > 4 ? std::stoi(argv[4]) : 1;
  reseam::mt::Engine engine(opt);
  for (std::string line; std::getline(std::cin, line);) std::cout << engine.translate(argv[2], argv[3], {line})[0] << "\n";
}
