#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"

#include <stdexcept>


template<typename ViewCollection>
void transfer_data_host_to_device(size_t tuple_idx, ViewCollection& view_collection)
{
    // start by getting the various views
    auto views_h  = std::get<0>(view_collection);
    auto views_d  = std::get<1>(view_collection);
    auto views_sc = std::get<2>(view_collection);

    std::cout << "Host to Device Data tranfer initialized..." << std::endl;
    std::cout << "  Tuple index is: " << tuple_idx << std::endl;

    // now we basically want to just find the view and do some checks based on the rank
    find_tuple(views_h,
               tuple_idx,
               [&]<typename HostViewType>(HostViewType& view_h)
    {
        // BEGIN LOGIC FOR RANK 1
        if constexpr (HostViewType::rank == 1)
        {
            // Rank 1 consists of 1D vectors that just go from host to device
            std::cout << "  Host views are of rank 1" << std::endl;

            // iterate through the device views
            find_tuple(views_d,
                       tuple_idx,
                       [&]<typename DeviceViewType>(DeviceViewType& view_d)
            {
                // at this level, we need to make sure the complier only considers rank 1 in view D
                if constexpr (DeviceViewType::rank == 1)
                {
                    // do the transfer
                    std::cout << "  Rank 1 transferring to device!" << std::endl;
                    Kokkos::deep_copy(view_d, view_h);
                }
                else
                {
                    throw std::runtime_error("HOST-TO-DEVICE: Rank 1 tensor on host matched with "
                                             "an incorrect rank on device!");
                }
            });
        }
        // BEGIN LOGIC FOR RANK 2
        else if constexpr (HostViewType::rank == 2)
        {
            std::cout << "  Host views are of rank 2" << std::endl;
            find_tuple(views_sc,
                       tuple_idx,
                       [&]<typename ScratchViewType>(ScratchViewType& view_sc)
            {
                // make sure the compiler only considers matches of rank 2
                if constexpr (ScratchViewType::rank == 2)
                {
                    std::cout << "  Rank 2 transfering to scratch!" << std::endl;

                    Kokkos::deep_copy(view_sc, view_h);

                    // once it's on the scratch, we need to copy over to the device
                    find_tuple(views_d,
                               tuple_idx,
                               [&]<typename DeviceViewType>(DeviceViewType& view_d)
                    {
                        // once again, make sure the compiler only considers matches of rank 2
                        if constexpr (DeviceViewType::rank == 2)
                        {
                            std::cout << "  Rank 2 transferring to device!" << std::endl;

                            Kokkos::deep_copy(view_d, view_sc);
                        }
                        else
                        {
                            throw std::runtime_error("HOST-TO-DEVICE: Rank 2 view on scratch "
                                                     "matched with an incorrect rank on device!");
                        }
                    });
                }
                else
                {
                    throw std::runtime_error("HOST-TO-DEVICE: Rank 2 view on host matched with an "
                                             "incorrect rank on scratch!");
                }
            });
        } // END SPLIT FOR RANK 1 AND 2
    });
}

template<typename ViewCollection>
void transfer_data_device_to_host(size_t tuple_idx, ViewCollection& view_collection)
{
    // start by getting the various views
    auto views_h  = std::get<0>(view_collection);
    auto views_d  = std::get<1>(view_collection);
    auto views_sc = std::get<2>(view_collection);

    std::cout << "Device to Host data transfer initialized..." << std::endl;
    std::cout << "  Tuple index is: " << tuple_idx << std::endl;

    // just like the function above, we want to find the view and do some checks absed on the rank
    // but this time we go backward! Device to host or device -> scratch -> host
    find_tuple(views_d,
               tuple_idx,
               [&]<typename DeviceViewType>(DeviceViewType& view_d)
    {
        // BEGIN LOGIC FOR RANK 1
        if constexpr (DeviceViewType::rank == 1)
        {
            // Rank 1 consists of 1D vectors that just go from device to host
            std::cout << "  Device views are of rank 1" << std::endl;

            // iterate through the host views
            find_tuple(views_h,
                       tuple_idx,
                       [&]<typename HostViewType>(HostViewType& view_h)
            {
                // at this level, we need to make sure the compiler only considers rank 1 in view h
                if constexpr (HostViewType::rank == 1)
                {
                    // do the transfer
                    std::cout << "  Rank 1 transferring to host!" << std::endl;
                    Kokkos::deep_copy(view_h, view_d);
                }
                else
                {
                    throw std::runtime_error("DEVICE-TO-HOST - Rank 1 tensor on device matched "
                                             "with an incorrect rank on host!");
                }
            });
        }
        // BEGIN LOGIC FOR RANK 2
        else if constexpr (DeviceViewType::rank == 2)
        {
            std::cout << "  Device views are of rank 2" << std::endl;
            find_tuple(views_sc,
                       tuple_idx,
                       [&]<typename ScratchViewType>(ScratchViewType& view_sc)
            {
                // make sure the compiler only consders matches of rank 2
                if constexpr (ScratchViewType::rank == 2)
                {
                    std::cout << "  Rank 2 transferring to scratch!" << std::endl;

                    Kokkos::deep_copy(view_sc, view_d);

                    // once it's on scratch, we need to copy over to the host
                    find_tuple(views_h,
                               tuple_idx,
                               [&]<typename HostViewType>(HostViewType& view_h)
                    {
                        // once again, make sure the compiler only considres matches of rank 2
                        if constexpr (HostViewType::rank == 2)
                        {
                            std::cout << "  Rank 2 is transferring to host!" << std::endl;

                            Kokkos::deep_copy(view_h, view_sc);
                        }
                        else
                        {
                            throw std::runtime_error("DEVICE-TO-HOST - Rank 2 tensor on scratch "
                                                     "matched with an incorrect rank on host!");
                        }
                    });
                }
                else
                {

                    throw std::runtime_error("DEVICE-TO-HOST - Rank 2 tensor on device matched "
                                             "with an incorrect rank on scratch!");
                }
            });
        } // END SPLIT FOR RANK 1 AND 2
    });
}