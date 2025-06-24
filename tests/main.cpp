#include "test_config.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_session.hpp>
#include <string>
#include <vector>

// initialize the global device (set in test_config.hpp)
DeviceSelector GLOBAL_DEVICE = DeviceSelector::AUTO;

int main(int argc, char* argv[])
{

    // gather the kokkos and catch args separately
    std::vector<char*> kokkos_args;
    std::vector<char*> catch_args = { argv[0] };

    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--device=HOST" || std::string(argv[i]) == "--device=DEVICE" ||
            std::string(argv[i]) == "--device=AUTO")
        {
            std::string val = argv[i];
            val             = val.substr(val.find('=') + 1);
            if (val == "HOST")
                GLOBAL_DEVICE = DeviceSelector::HOST;
            else if (val == "DEVICE")
                GLOBAL_DEVICE = DeviceSelector::DEVICE;
            else
                GLOBAL_DEVICE = DeviceSelector::AUTO;
        }
        else
        {
            catch_args.push_back(argv[i]);
        }
    }

    // this makes sure that kokkos is only run **once** across all files, as catch creates a session
    // and runs them *all* in a single binary, attempting to initialize them in each test is a bad
    // idea.
    int kokkos_argc = 1;
    Kokkos::initialize(kokkos_argc, argv);

    // run the catch2 tests that are discovered
    Catch::Session session;
    int result = session.run((int)catch_args.size(), catch_args.data());

    // then finalize kokkos after everything completes!
    Kokkos::finalize();

    return result;
}
