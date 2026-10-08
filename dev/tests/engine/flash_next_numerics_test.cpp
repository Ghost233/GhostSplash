#include "model/FlashNextNumerics.hpp"
#include "TestChecks.hpp"
#include <array>
#include <iostream>
#include <limits>
using namespace splash;
int main() {
  try {
    std::array<ModelStepResult,3> lanes;
    lanes[0]=model::flashNextLaneResult(10,[] {
      model::requireFiniteFlashNextLogit(std::numeric_limits<float>::quiet_NaN(),"generation logit non-finite");
      ModelStepResult out;out.outputTokens={9};return out;
    });
    lanes[1]=model::flashNextLaneResult(11,[] {
      model::requireFiniteFlashNextLogit(1.25f,"score logit non-finite");
      ModelStepResult out;out.requestId=11;out.scoreLogits={1.25f};out.finished=true;return out;
    });
    lanes[2]=model::flashNextLaneResult(12,[] {
      model::requireFiniteFlashNextLogit(std::numeric_limits<float>::infinity(),"score logit non-finite");
      ModelStepResult out;out.scoreLogits={0};return out;
    });
    test::require(lanes[0].requestId==10 && !lanes[0].failure.empty() && lanes[0].outputTokens.empty(),
                  "invalid generation logits must fail only their lane before output");
    test::require(lanes[1].failure.empty() && lanes[1].scoreLogits==std::vector<float>{1.25f},
                  "a numeric failure must not stop another lane's score result");
    test::require(lanes[2].requestId==12 && !lanes[2].failure.empty() && lanes[2].scoreLogits.empty(),
                  "invalid score logits must not return a partial score row");
    test::rejects([] {static_cast<void>(model::flashNextLaneResult(13,[]() -> ModelStepResult {
      throw std::runtime_error("SSD read failed");
    }));},"SSD read failed","IO failure was converted to a request result");
    test::rejects([] {static_cast<void>(model::flashNextLaneResult(14,[]() -> ModelStepResult {
      throw std::logic_error("invalid KV position");
    }));},"invalid KV position","state contract failure was swallowed");
    std::cout<<"Flash Next finite generation/score checks isolate numeric failures; IO and invariants propagate\n";
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
