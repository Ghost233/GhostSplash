#pragma once
#include <filesystem>
#include <string_view>
namespace splash::encoder {
int serveDecision(std::string_view command, int argc, char **argv,
                  const std::filesystem::path &metallib, std::string_view buildId);
}
