#pragma once
#include <filesystem>
#include <string_view>
namespace splash::encoder {
int serveModernBert(int argc, char **argv, const std::filesystem::path &metallib, std::string_view buildId);
int serveD3(int argc, char **argv, const std::filesystem::path &metallib, std::string_view buildId);
}
