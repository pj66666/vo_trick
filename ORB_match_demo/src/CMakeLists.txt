add_library(demo_match SHARED
ORBextractor.cpp
)

target_link_libraries(demo_match ${THIRD_PARTY_LIBS})
