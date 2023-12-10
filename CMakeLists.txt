cmake_minimum_required(VERSION 3.7)

project(NVAR_test)
include(CTest)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	set(CMAKE_CXX_COMPILER /opt/homebrew/bin/g++-13)
else()
	set(CMAKE_CXX_COMPILER g++)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wconversion -pedantic -O0 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-error=deprecated-enum-enum-conversion -Wno-error=class-memaccess -Werror -DNDEBUG -march=native -Wall -Wextra -pedantic -O3")

option(JSON_ImplicitConversions OFF)

include_directories(./include)
include_directories(~/include)
include_directories(nlohmann_json/include)
add_subdirectory(src)

# Forecast target for predicting the exact next difference
# Main target stuff
add_executable(main src/main.cpp)
target_compile_definitions(main PUBLIC FORECAST)
add_custom_command(
	TARGET main POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
	COMMENT "Transferring compile_commands.json"
)
add_custom_target(
	forecast
	COMMAND main
	DEPENDS main
	COMMENT "Running forecast."
)
add_custom_command(
	TARGET forecast POST_BUILD
	COMMAND python3.11 ./plot_forecast.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
	COMMENT "Plotting forecast data."
)

# Forecast target for predicting the change for the next parameter
add_executable(main_difference src/main.cpp)
target_compile_definitions(main_difference PUBLIC FORECAST TARGET_DIFFERENCE)
add_custom_target(
	forecast_difference
	COMMAND main_difference
	DEPENDS main_difference
	COMMENT "Running forecast with the TARGET_DIFFERENCE"
)
add_custom_command(
	TARGET main_difference POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
	COMMENT "Transferring compile_commands.json"
)
add_custom_command(
	TARGET forecast_difference POST_BUILD
	COMMAND python3.11 ./plot_forecast.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
	COMMENT "Plotting forecast data."
)

# Doublescroll target stuff
add_executable(double_scroll src/main.cpp)
target_compile_definitions(double_scroll PUBLIC DOUBLESCROLL)
add_custom_target(
	doublescroll
	COMMAND double_scroll
	DEPENDS double_scroll
	COMMENT "Running doublescroll."
)
add_custom_command(
	TARGET double_scroll POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
	COMMENT "Transferring compile_commands.json"
)
add_custom_command(
	TARGET double_scroll PRE_BUILD
	COMMAND python3.11 ./create_test_data.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
	COMMENT "Generating doublescroll data."
)
add_custom_command(
	TARGET doublescroll POST_BUILD
	COMMAND python3.11 ./plot_doublescroll.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
	COMMENT "Plotting doublescroll."
)

# HH model target
add_executable(hhmodel src/main.cpp)
target_compile_definitions(hhmodel PUBLIC HH_MODEL)
add_custom_target(
	hh_model
	COMMAND hhmodel
	DEPENDS hhmodel
	COMMENT "Fitting Hodgkin-Huxley model."
)
add_custom_command(
	TARGET hhmodel POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
	COMMENT "Transferring compile_commands.json"
)
add_custom_command(
	TARGET hh_model POST_BUILD
	COMMAND python3.11 ./plot_hh_model.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
	COMMENT "Plotting Hodgkin-Huxley model."
)

# Custom features target
add_executable(customfeat src/main.cpp)
target_compile_definitions(customfeat PUBLIC CUSTOM_FEATURES)
add_custom_target(
	custom_feat
	COMMAND customfeat
	DEPENDS customfeat
	COMMENT "Fitting custom features."
)
add_custom_command(
	TARGET customfeat POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json
	COMMENT "Transferring compile_commands.json"
)
add_custom_command(
	TARGET custom_feat POST_BUILD
	COMMAND python3.11 ./plot_custom.py
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data
	COMMENT "Plotting custom features."
)

# Testing target stuff
add_executable(unit_test src/unit_test.cpp)
add_test(test_build
	"${CMAKE_COMMAND}"
	--build .
	--target unit_test
)
set_tests_properties(test_build PROPERTIES FIXTURES_SETUP test_fixture)
add_test(test unit_test)
set_tests_properties(test PROPERTIES FIXTURES_REQUIRED test_fixture)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	target_link_libraries(main d_classic)
	target_link_libraries(main_difference d_classic)
	target_link_libraries(double_scroll d_classic)
	target_link_libraries(hhmodel d_classic)
	target_link_libraries(customfeat d_classic)
	target_link_libraries(unit_test d_classic)
endif()
