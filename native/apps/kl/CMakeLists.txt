# ingine-kl (RTX 5080 port): KL / top-1 against a reference dump, and expert routing calibration.
if(TARGET ingine_model)
  add_executable(ingine-kl main.cu)
  ingine_cuda_target(ingine-kl)
  ingine_internal_includes(ingine-kl)
  target_link_libraries(ingine-kl PRIVATE ingine_model ingine_text nlohmann_json::nlohmann_json CUDA::cudart)
endif()
