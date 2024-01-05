// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
// 

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include <Kokkos_Core.hpp>

class Algorithm
{

};


// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename T>
struct EquivalentView<ExecutionSpace, std::vector<T>> {
	using type = Kokkos::View<typename std::vector<T>::value_type*, typename ExecutionSpace::array_layout, typename ExecutionSpace::memory_space>;
};

template<typename ExecutionSpace, typename T>
struct EquivalentView<ExecutionSpace, const std::vector<T>> {
	using type = Kokkos::View<const typename std::vector<T>::value_type*, typename ExecutionSpace::array_layout, typename ExecutionSpace::memory_space>;
};

template<typename ExecutionSpace>
struct Views
{
	// Create a view for a given executation space and C++ data structure(each structure needs a specialization)
	template<typename T>
	static typename EquivalentView<ExecutionSpace, T>::type create_view(T&& arr)
	{
		static_assert(false, "Specialization for type not implemented yet.");
	}

	// Specialization for std::vector (default allocator)
	template<typename T>
	static typename EquivalentView<ExecutionSpace, std::vector<T>>::type create_view(std::vector<T>& arr)
	{
		return typename EquivalentView<ExecutionSpace, std::vector<T>>::type(arr.data(), arr.size());
	}

	// Specialization for const std::vector (default allocator)
	template<typename T>
	static typename EquivalentView<ExecutionSpace, const std::vector<T>>::type create_view(const std::vector<T>& arr)
	{
		return typename EquivalentView<ExecutionSpace, const std::vector<T>>::type(arr.data(), arr.size());
	}

	
	// Creates view for a given execution space for a variadic list of data structures
	// (each needs a create_view specialization)
	template<typename... ParameterTypes>
	static auto create_views(ParameterTypes&&... params)
	{
		return std::make_tuple(Views<ExecutionSpace>::create_view(params)...);
	}
	
	template<typename... ParameterTypes>
	static auto create_views2(std::tuple<ParameterTypes...>& params_tuple)
	{
		return std::make_tuple((Views<ExecutionSpace>::create_view(std::forward<decltype(params)>(params)), ...));
	}

	/*
	template<typename... ParameterTypes>
	static auto create_views2(ParameterTypes&&... params)
	{
		return std::make_tuple(Views<ExecutionSpace>::create_view(params)...);
	}
	*/
};

// Need method to get view from data type

//template<typename ExecutionSpace, typename... ParameterTypes>
template<typename... ParameterTypes>
class Kernel
{
public:
	using ExecutionSpace = Kokkos::Serial;

	Kernel(ParameterTypes&... params)
		: parameters(std::forward<ParameterTypes>(params)...)
	{
	}

	void call()
	{
		Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace::memory_space> x_view(x.data(), x.size());
		Kokkos::View<double*, ExecutionSpace::array_layout, ExecutionSpace::memory_space> y_view(y.data(), y.size());

		auto kernel = [=](const auto& i) {
			y_view[i] = x_view[i] * x_view[i];
			};


		auto views = Views<ExecutionSpace>::create_views2(parameters...);


		// User provided
		auto kernel = [](auto& views, const auto& i) {
				std::get<1>(views)[i] = std::get<0>(views)[i] * std::get<0>(views)[i];
			};

		// data movement needs to happen at the algorithm level
		// 
		// Inside Kernel for the wrapper
		auto views = Views<ExecutionSpace>::create_views2(parameters...);
		// TODO deep copies (somewhere)
		auto kernel_wrapper = [=](const auto& i)
			{
				kernel(views, i);
			};

		// call
		//
		auto range_policy = Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace(), 0, x.size());

		Kokkos::parallel_for(range_policy, kernel_wrapper);
	};

	// data_host_references
	// data_views
	// execution_parameters = # of threads, etc.
	std::tuple<ParameterTypes&...> parameters;
	//std::tuple<Kokkos::View<decltype(ParameterTypes)>...> parameter_views;
};

int main()
{
	std::vector<double> x(1000);
	std::iota(x.begin(), x.end(), 0.0);

	std::vector<double> y(1000);

	auto views = Views<Kokkos::Serial>::create_views(x, y);

	// Kernels never allocate data (global)

	Kernel k(std::as_const(x), y /*,
		auto kernel = [](auto& views, const auto& i) {
				std::get<1>(views)[i] = std::get<0>(views)[i] * std::get<0>(views)[i];
			};*/
			);

	std::vector<double> z(1000);
	Kernel k2(std::as_const(y), z /*,
		auto kernel = [](auto& views, const auto& i) {
				std::get<1>(views)[i] = std::get<0>(views)[i] * std::get<0>(views)[i];
			};*/
	);

	

	// At the end, the algorithm needs to know the "final" output that needs copied to the host
	// Data needs moved if 1) it is a kernel input or 2) algorithm output
	// Data view deallocation if 1) it is not a downstream input 2) and not algorithm output
	// - perhaps use counter for each view (+1 for algorithm output) to know when to deallocate it
	// Need algorithm to construct the counters, for example:
	//   k.parameters[1] is not const and hence output
	//   k2.parameters[0] is const and hence input
	assert(&std::get<1>(k.parameters) == &std::get<0>(k2.parameters));

	auto view = Views<Kokkos::Serial>::create_view(x);

	

	
	auto views2 = Views<Kokkos::Serial>::create_views2(k.parameters);
	std::apply([](auto &&... params) { (Views<Kokkos::Serial>::create_view(std::forward<decltype(params)>(params)), ...); }, k.parameters);
	//k.call();

	Kokkos::View<double*> x_view(x.data(), x.size());
	Kokkos::View<double*> y_view(y.data(), y.size());

	auto kernel = [=](const auto& i) {
		y_view[i] = x_view[i] * x_view[i];
		};

	auto range_policy = Kokkos::RangePolicy<Kokkos::Serial>(Kokkos::Serial(), 0, 1000);

	Kokkos::parallel_for(range_policy, kernel);

	for (auto i = 0; i < 1000; i++)
	{
		assert(x[i] * x[i] == y[i]);
	}
}