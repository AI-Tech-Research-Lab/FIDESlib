//
// Created by carlosad on 16/2/26.
//
#include <charconv>
#include <cstring>
#include <iostream>
#include <string_view>

#include "ParametrizedTest.cuh"

class GlobalEnv : public ::testing::Environment {
  public:
	void SetUp() override {
		// global initialization before any test runs
	}

	void TearDown() override {
		FIDESlib::Testing::cached_cc.clear();
		FIDESlib::CKKS::DeregisterAllContexts();
	}
};

namespace {

/// The suite ran on device 0 only, because `devices` (ParametrizedTest.cuh) was hardcoded and
/// nothing overrode it -- which is why the multi-GPU paths went untested. --devices=0,2,3 runs
/// the whole suite over that device set instead; the argument is consumed here, after
/// InitGoogleTest has taken its own flags, so gtest never sees it.
bool parseDevices(std::string_view spec, std::vector<int>& out) {
	std::vector<int> parsed;
	while (!spec.empty()) {
		const size_t comma	  = spec.find(',');
		std::string_view field = spec.substr(0, comma);
		int value			  = 0;
		const auto [end, ec]  = std::from_chars(field.data(), field.data() + field.size(), value);
		if (ec != std::errc{} || end != field.data() + field.size() || value < 0)
			return false;
		parsed.push_back(value);
		if (comma == std::string_view::npos)
			break;
		spec.remove_prefix(comma + 1);
	}
	if (parsed.empty())
		return false;
	out = std::move(parsed);
	return true;
}

} // namespace

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);

	constexpr std::string_view flag = "--devices=";
	for (int i = 1; i < argc; ++i) {
		const std::string_view arg{ argv[i] };
		if (!arg.starts_with(flag))
			continue;
		if (!parseDevices(arg.substr(flag.size()), devices)) {
			std::cerr << "Invalid " << flag << " value: " << arg.substr(flag.size()) << " (expected e.g. --devices=0,2,3)" << std::endl;
			return 2;
		}
		// Drop it so gtest's own unrecognised-flag handling does not see it.
		for (int j = i; j + 1 < argc; ++j)
			argv[j] = argv[j + 1];
		--argc;
		--i;
	}

	std::cout << "Running on device(s):";
	for (int d : devices)
		std::cout << ' ' << d;
	std::cout << std::endl;

	::testing::AddGlobalTestEnvironment(new GlobalEnv());
	return RUN_ALL_TESTS();
}
