/**
 * @file optimizer_build_views.cpp
 * @brief This file is just pure C++ that is to be **included** for building optimizer views
 *
 */

// NOTE: THIS IS **NOT** VALID C++ TO BE COMPILED, THIS IS ONLY TO BE INCLUDED
// TO AVOID REPEATING THINGS IN THE TEST CASES

std::vector<double> a(N);
std::vector<double> b(N);
std::vector<double> c(N);
std::vector<double> d(N);
std::vector<double> e(N);
std::vector<double> f(N);
std::vector<double> g(N);
std::vector<double> h(N);

// a few matrices
DynMatrix2D A(N, M);
DynMatrix2D B(N, M);

// then initialize
std::iota(a.begin(), a.end(), 1.0);
std::iota(b.begin(), b.end(), 2.0);
std::iota(c.begin(), c.end(), 3.0);
std::iota(d.begin(), d.end(), 4.0);
std::iota(e.begin(), e.end(), 5.0);
std::iota(f.begin(), f.end(), 6.0);
std::iota(g.begin(), g.end(), 7.0);
std::iota(h.begin(), h.end(), 8.0);

int ij = 0;
for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < M; j++)
    {
        A(i, j) = static_cast<double>(ij++);
        B(i, j) = static_cast<double>(ij++);
    }

constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                            HashedName<hash("B")>(),
                                            HashedName<hash("a")>(),
                                            HashedName<hash("b")>(),
                                            HashedName<hash("c")>(),
                                            HashedName<hash("d")>(),
                                            HashedName<hash("e")>(),
                                            HashedName<hash("f")>(),
                                            HashedName<hash("g")>(),
                                            HashedName<hash("h")>());

printf("\nbuilding the views...\n");
auto data_views = create_views(pack(A, B, a, b, c, d, e, f, g, h));

printf("views built!\n");

auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
// vector views
auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
auto& c_views = std::get<find<hash("c")>(data_names)>(data_views);
auto& d_views = std::get<find<hash("d")>(data_names)>(data_views);
auto& e_views = std::get<find<hash("e")>(data_names)>(data_views);
auto& f_views = std::get<find<hash("f")>(data_names)>(data_views);
auto& g_views = std::get<find<hash("g")>(data_names)>(data_views);
auto& h_views = std::get<find<hash("h")>(data_names)>(data_views);


#if 0
else if constexpr (chain_use == 1)
{
    // this kernel is the "original" testing with k1, k3, and k4 from the old main.cpp
    auto k1 = KernelVectorDot(options, std::as_const(a_views), std::as_const(b_views), c_views);
    auto k2 = KernelVectorDot(options, std::as_const(d_views), std::as_const(e_views), f_views);
    auto k3 = KernelVectorDot(options, std::as_const(c_views), std::as_const(g_views), h_views);
    // so, k2 is independent and k3 is dependent

    auto kernels = pack(k1, k2, k3);
}
#endif

// then we can set up the optimizer
