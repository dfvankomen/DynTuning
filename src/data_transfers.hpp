#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"

#include <cstddef>
#include <stdexcept>

// just uncomment this if you want to enable it (or we can enable in cmake)
// #define DYNTUNE_DEBUG_DATA_TRANSFER

// TODO: add timers back in (could do a null pointer as default)

/**
 * @brief Function that transfers data from host to device
 *
 * @tparam ViewCollection Inferred type of views
 * @param tuple_idx Index of tuple that needs to be transferred
 * @param view_collection All views available
 */
template<typename ViewCollection>
void transfer_data_host_to_device(size_t tuple_idx, ViewCollection& view_collection)
{
    // start by getting the various views
    auto views_h  = std::get<0>(view_collection);
    auto views_d  = std::get<1>(view_collection);
    auto views_sc = std::get<2>(view_collection);

#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
    std::cout << "Host to Device Data tranfer initialized..." << std::endl;
    std::cout << "  Tuple index is: " << tuple_idx << std::endl;
#endif

    // now we basically want to just find the view and do some checks based on the rank
    find_tuple(views_h,
               tuple_idx,
               [&]<typename HostViewType>(HostViewType& view_h)
    {
        // BEGIN LOGIC FOR RANK 1
        if constexpr (HostViewType::rank == 1)
        {
// Rank 1 consists of 1D vectors that just go from host to device
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
            std::cout << "  Host views are of rank 1" << std::endl;
#endif

            // iterate through the device views
            find_tuple(views_d,
                       tuple_idx,
                       [&]<typename DeviceViewType>(DeviceViewType& view_d)
            {
                // at this level, we need to make sure the complier only considers rank 1 in view D
                if constexpr (DeviceViewType::rank == 1)
                {
                    // do the transfer
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                    std::cout << "  Rank 1 transferring to device!" << std::endl;
#endif
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
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
            std::cout << "  Host views are of rank 2" << std::endl;
#endif
            find_tuple(views_sc,
                       tuple_idx,
                       [&]<typename ScratchViewType>(ScratchViewType& view_sc)
            {
                // make sure the compiler only considers matches of rank 2
                if constexpr (ScratchViewType::rank == 2)
                {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                    std::cout << "  Rank 2 transfering to scratch!" << std::endl;
#endif

                    Kokkos::deep_copy(view_sc, view_h);

                    // once it's on the scratch, we need to copy over to the device
                    find_tuple(views_d,
                               tuple_idx,
                               [&]<typename DeviceViewType>(DeviceViewType& view_d)
                    {
                        // once again, make sure the compiler only considers matches of rank 2
                        if constexpr (DeviceViewType::rank == 2)
                        {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                            std::cout << "  Rank 2 transferring to device!" << std::endl;
#endif

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
        }
        // BEGIN LOGIC FOR RANK 3
        else if constexpr (HostViewType::rank == 3)
        {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
            std::cout << "  Host views are of rank 3" << std::endl;
#endif
            find_tuple(views_sc,
                       tuple_idx,
                       [&]<typename ScratchViewType>(ScratchViewType& view_sc)
            {
                // make sure the compiler only considers matches of rank 3
                if constexpr (ScratchViewType::rank == 3)
                {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                    std::cout << "  Rank 3 transfering to scratch!" << std::endl;
#endif

                    Kokkos::deep_copy(view_sc, view_h);

                    // once it's on the scratch, we need to copy over to the device
                    find_tuple(views_d,
                               tuple_idx,
                               [&]<typename DeviceViewType>(DeviceViewType& view_d)
                    {
                        // once again, make sure the compiler only considers matches of rank 3
                        if constexpr (DeviceViewType::rank == 3)
                        {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                            std::cout << "  Rank 3 transferring to device!" << std::endl;
#endif

                            Kokkos::deep_copy(view_d, view_sc);
                        }
                        else
                        {
                            throw std::runtime_error("HOST-TO-DEVICE: Rank 3 view on scratch "
                                                     "matched with an incorrect rank on device!");
                        }
                    });
                }
                else
                {
                    throw std::runtime_error("HOST-TO-DEVICE: Rank 3 view on host matched with an "
                                             "incorrect rank on scratch!");
                }
            });
        } // END SPLIT FOR RANK 1, 2, and 3
    });
}


/**
 * @brief Function that transfers data from host to device, includes timer access
 *
 * @tparam ViewCollection Inferred type of views
 * @tparam Timer Inferred type of timer object
 * @param tuple_idx Index of tuple that needs to be transferred
 * @param view_collection All views available
 * @param elapsed Current elapsed time
 * @param timer Timer object used
 */
template<typename ViewCollection, typename Timer>
void transfer_data_host_to_device(size_t tuple_idx,
                                  ViewCollection& view_collection,
                                  double& elapsed,
                                  Timer& timer)
{
    // reset the timer
    timer.reset();

    transfer_data_host_to_device(tuple_idx, view_collection);

    // add the amount of seconds to the timer
    elapsed += timer.seconds();
}

/**
 * @brief Function that transfers device from device to host
 *
 * @tparam ViewCollection Inferred type of views
 * @param tuple_idx Index of tuple that needs to be transferred
 * @param view_collection All views available
 */
template<typename ViewCollection>
void transfer_data_device_to_host(size_t tuple_idx, ViewCollection& view_collection)
{
    // start by getting the various views
    auto views_h  = std::get<0>(view_collection);
    auto views_d  = std::get<1>(view_collection);
    auto views_sc = std::get<2>(view_collection);

#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
    std::cout << "Device to Host data transfer initialized..." << std::endl;
    std::cout << "  Tuple index is: " << tuple_idx << std::endl;
#endif

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
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
            std::cout << "  Device views are of rank 1" << std::endl;
#endif

            // iterate through the host views
            find_tuple(views_h,
                       tuple_idx,
                       [&]<typename HostViewType>(HostViewType& view_h)
            {
                // at this level, we need to make sure the compiler only considers rank 1 in view h
                if constexpr (HostViewType::rank == 1)
                {
                    // do the transfer
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                    std::cout << "  Rank 1 transferring to host!" << std::endl;
#endif
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
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
            std::cout << "  Device views are of rank 2" << std::endl;
#endif
            find_tuple(views_sc,
                       tuple_idx,
                       [&]<typename ScratchViewType>(ScratchViewType& view_sc)
            {
                // make sure the compiler only consders matches of rank 2
                if constexpr (ScratchViewType::rank == 2)
                {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                    std::cout << "  Rank 2 transferring to scratch!" << std::endl;
#endif

                    Kokkos::deep_copy(view_sc, view_d);

                    // once it's on scratch, we need to copy over to the host
                    find_tuple(views_h,
                               tuple_idx,
                               [&]<typename HostViewType>(HostViewType& view_h)
                    {
                        // once again, make sure the compiler only considres matches of rank 2
                        if constexpr (HostViewType::rank == 2)
                        {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                            std::cout << "  Rank 2 is transferring to host!" << std::endl;
#endif

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
        }
        // BEGIN LOGIC FOR RANK 3
        else if constexpr (DeviceViewType::rank == 3)
        {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
            std::cout << "  Device views are of rank 3" << std::endl;
#endif
            find_tuple(views_sc,
                       tuple_idx,
                       [&]<typename ScratchViewType>(ScratchViewType& view_sc)
            {
                // make sure the compiler only consders matches of rank 3
                if constexpr (ScratchViewType::rank == 3)
                {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                    std::cout << "  Rank 3 transferring to scratch!" << std::endl;
#endif

                    Kokkos::deep_copy(view_sc, view_d);

                    // once it's on scratch, we need to copy over to the host
                    find_tuple(views_h,
                               tuple_idx,
                               [&]<typename HostViewType>(HostViewType& view_h)
                    {
                        // once again, make sure the compiler only considres matches of rank 3
                        if constexpr (HostViewType::rank == 3)
                        {
#ifdef DYNTUNE_DEBUG_DATA_TRANSFER
                            std::cout << "  Rank 3 is transferring to host!" << std::endl;
#endif

                            Kokkos::deep_copy(view_h, view_sc);
                        }
                        else
                        {
                            throw std::runtime_error("DEVICE-TO-HOST - Rank 3 tensor on scratch "
                                                     "matched with an incorrect rank on host!");
                        }
                    });
                }
                else
                {

                    throw std::runtime_error("DEVICE-TO-HOST - Rank 3 tensor on device matched "
                                             "with an incorrect rank on scratch!");
                }
            });
        } // END SPLIT FOR RANK 1, 2 and 3
    });
}


/**
 * @brief Function that transfers data from device to host, includes timer access
 *
 * @tparam ViewCollection Inferred type of views
 * @tparam Timer Inferred type of timer object
 * @param tuple_idx Index of tuple that needs to be transferred
 * @param view_collection All views available
 * @param elapsed Current elapsed time
 * @param timer Timer object used
 */
template<typename ViewCollection, typename Timer>
void transfer_data_device_to_host(size_t tuple_idx,
                                  ViewCollection& view_collection,
                                  double& elapsed,
                                  Timer& timer)
{
    // reset the timer
    timer.reset();

    transfer_data_device_to_host(tuple_idx, view_collection);

    // add the amount of seconds to the timer
    elapsed += timer.seconds();
}
