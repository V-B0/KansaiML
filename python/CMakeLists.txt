nanobind_add_module(_core NB_STATIC bindings.cpp)
target_link_libraries(_core PRIVATE kansai_core)

# Drop the compiled extension straight into the kansai/ package directory
# so `import kansai` works from the source tree with no install step.
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/kansai"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/kansai"
)

# Separate from the LIBRARY_OUTPUT_DIRECTORY above (that one's for the
# no-install source-tree workflow the README's Quickstart uses) --
# this install() rule is what scikit-build-core's wheel build actually
# runs (via `cmake --install`) to pull the compiled extension into the
# wheel, at pyproject.toml's `wheel.packages = ["python/kansai"]` root.
# Harmless no-op for a plain `cmake --build` with no matching
# `cmake --install` step, so it doesn't change the existing local-dev
# workflow at all.
install(TARGETS _core LIBRARY DESTINATION kansai)
