cmake_minimum_required(VERSION 3.28)
project(reseam-translate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Every file targets x86-64-v3 (AVX2 + FMA, Haswell / Zen 1 and later). The faster kernels are
# compiled separately with their own instruction sets and picked at runtime, so one binary runs
# at its best on AVX2, AVX-VNNI, AVX-512 VNNI (Ice Lake, Zen 4) and AMX (Sapphire Rapids) CPUs.
set(RESEAM_BASE_ARCH "x86-64-v3" CACHE STRING "Baseline -march for all code")
add_compile_options(-march=${RESEAM_BASE_ARCH} -mtune=generic -fno-math-errno)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# SentencePiece (Apache-2.0): tokenization, as in Marian.
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG v0.2.0
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)

# cpp-httplib (MIT) and nlohmann/json (MIT): the HTTP server, header-only.
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.18.1
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(sentencepiece httplib json)

set(AVX512_FLAGS -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni)

add_library(reseam_mt STATIC
  src/engine.cpp src/batch.cpp src/team.cpp src/model.cpp src/text.cpp src/ops.cpp src/ops_avx512.cpp
  src/gemm.cpp src/gemm_avx2.cpp src/gemm_avxvnni.cpp src/gemm_avx512.cpp src/gemm_amx.cpp)
target_include_directories(reseam_mt PUBLIC src PRIVATE ${sentencepiece_SOURCE_DIR}/src)
target_compile_options(reseam_mt PRIVATE -Wall -Wextra)
set_source_files_properties(src/gemm_avxvnni.cpp PROPERTIES COMPILE_OPTIONS "-mavxvnni")
set_source_files_properties(src/gemm_avx512.cpp src/ops_avx512.cpp PROPERTIES COMPILE_OPTIONS "${AVX512_FLAGS}")
set_source_files_properties(src/gemm_amx.cpp PROPERTIES COMPILE_OPTIONS "${AVX512_FLAGS};-mamx-tile;-mamx-int8")
find_package(Threads REQUIRED)
target_link_libraries(reseam_mt PRIVATE sentencepiece-static PUBLIC Threads::Threads)

add_executable(reseam-translate src/server.cpp)
target_link_libraries(reseam-translate PRIVATE reseam_mt httplib::httplib nlohmann_json::nlohmann_json)
install(TARGETS reseam-translate RUNTIME DESTINATION bin)

# Tests and tools.
option(RESEAM_BUILD_TESTS "Build tests and benchmarks" ON)
if(RESEAM_BUILD_TESTS)
  enable_testing()
  foreach(name test_gemm test_ops test_text bench bench_gemm translate)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE reseam_mt)
  endforeach()
  add_test(NAME kernels COMMAND test_gemm)
  add_test(NAME attention COMMAND test_ops)
  add_test(NAME text COMMAND test_text)
endif()
