nanobind_add_module(_core NB_STATIC bindings.cpp)
target_link_libraries(_core PRIVATE kansai_core)

# Drop the compiled extension straight into the kansai/ package directory
# so `import kansai` works from the source tree with no install step.
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/kansai"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/kansai"
)
