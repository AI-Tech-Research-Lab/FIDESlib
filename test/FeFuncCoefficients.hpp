//==================================================================================
// Fourier-series coefficient tables for FE functional bootstrapping tests.
//
// The three tables below are copied verbatim from OpenFHE v1.6.1,
// src/pke/examples/FE-functional-bootstrapping-ckks.cpp, and are distributed
// under the OpenFHE license reproduced here:
//
// BSD 2-Clause License
//
// Copyright (c) 2014-2026, NJIT, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================
//
// Convention: a function f on [-B, B] is evaluated on a message m in [-1/2, 1/2) as
// g(m) = f(2 * B * m). exp uses B = 2; sigmoid and GELU use B = 8.

#ifndef FIDESLIB_TEST_FE_FUNC_COEFFICIENTS_HPP
#define FIDESLIB_TEST_FE_FUNC_COEFFICIENTS_HPP

#include <complex>
#include <vector>

namespace fe_func_coefficients {

// Fourier coefficients for the function std::exp(x) in [-2, 2] of degree 29
static const std::vector<std::complex<double>> coeff_exp_2_double_29{
        std::complex<double>(3.222093706013e+00, 0.000000000000e+00),
        std::complex<double>(-3.847497325992e+00, -3.036911315182e+00),
        std::complex<double>(1.613269565467e+00, 2.561562635536e+00),
        std::complex<double>(-7.142688329459e-01, -1.718044084361e+00),
        std::complex<double>(3.343375256712e-01, 1.087677253934e+00),
        std::complex<double>(-1.590083748525e-01, -6.591856314553e-01),
        std::complex<double>(7.444125399842e-02, 3.797240558554e-01),
        std::complex<double>(-3.347789249240e-02, -2.057416664390e-01),
        std::complex<double>(1.415639337190e-02, 1.036151378646e-01),
        std::complex<double>(-5.504078824289e-03, -4.782261609566e-02),
        std::complex<double>(1.912943397918e-03, 1.985252654576e-02),
        std::complex<double>(-5.690043709362e-04, -7.207032810727e-03),
        std::complex<double>(1.328669037822e-04, 2.177453328757e-03),
        std::complex<double>(-1.854630094550e-05, -4.890711403220e-04),
        std::complex<double>(-1.453596732859e-06, 5.053417952833e-05),
        std::complex<double>(1.572743922262e-06, 1.548627313317e-05),
        std::complex<double>(-2.838788709181e-07, -8.312159353832e-06),
        std::complex<double>(-7.562004956940e-08, 8.274522508133e-07),
        std::complex<double>(4.633880680387e-08, 6.598799631857e-07),
        std::complex<double>(-5.869467412101e-10, -2.528773346321e-07),
        std::complex<double>(-6.055031566421e-09, -3.204665662698e-08),
        std::complex<double>(1.190783888161e-09, 4.385916878681e-08),
        std::complex<double>(7.747582545133e-10, -3.358504294115e-09),
        std::complex<double>(-3.348892690935e-10, -7.088429287982e-09),
        std::complex<double>(-9.585399054352e-11, 1.863929871559e-09),
        std::complex<double>(8.204146390023e-11, 1.136804343802e-09),
        std::complex<double>(9.573941899682e-12, -5.803840255614e-10),
        std::complex<double>(-2.013818972690e-11, -1.764939479287e-10),
        std::complex<double>(1.533040187851e-13, 1.657178907988e-10),
        std::complex<double>(5.134323521894e-12, 2.323787581630e-11)};

// Fourier coefficients for the function 1 / (1 + std::exp(-x)) in [-8, 8] of degree 34
static const std::vector<std::complex<double>> coeff_sigmoid_8_double_34{
        std::complex<double>(2.500000000000e-01, 0.000000000000e+00),
        std::complex<double>(1.276756478319e-15, -2.986114677117e-01),
        std::complex<double>(-2.711798613355e-15, -1.305833657358e-03),
        std::complex<double>(2.012279232133e-15, -6.069910606535e-02),
        std::complex<double>(-3.874233833051e-17, -3.496972005070e-03),
        std::complex<double>(-5.065392549852e-16, -1.436915534313e-02),
        std::complex<double>(-4.025501720156e-16, -3.090757902275e-03),
        std::complex<double>(6.661338147751e-16, -3.011048528323e-03),
        std::complex<double>(9.549219054383e-17, -1.395169741000e-03),
        std::complex<double>(-4.406197628981e-16, -7.358936474771e-04),
        std::complex<double>(-2.137022113088e-16, -4.185575122752e-04),
        std::complex<double>(4.614364446098e-16, -2.239585471946e-04),
        std::complex<double>(4.897883314203e-19, -1.182545527493e-04),
        std::complex<double>(-4.024558464266e-16, -6.447113701542e-05),
        std::complex<double>(-1.695114860110e-16, -3.521198006050e-05),
        std::complex<double>(3.903127820948e-16, -1.881494613389e-05),
        std::complex<double>(1.909467050622e-16, -1.006822281208e-05),
        std::complex<double>(-9.540979117872e-17, -5.490602888772e-06),
        std::complex<double>(-6.585471100731e-17, -2.983401096265e-06),
        std::complex<double>(1.040834085586e-16, -1.590510122180e-06),
        std::complex<double>(1.641151407480e-16, -8.526983863427e-07),
        std::complex<double>(-6.938893903907e-17, -4.673955051855e-07),
        std::complex<double>(-1.931861246494e-16, -2.539297722240e-07),
        std::complex<double>(3.035766082959e-16, -1.341034123884e-07),
        std::complex<double>(7.532874064103e-17, -7.180768169489e-08),
        std::complex<double>(-1.049507702966e-16, -4.002071006315e-08),
        std::complex<double>(-8.495889538759e-17, -2.179193493192e-08),
        std::complex<double>(1.587271980519e-16, -1.117364778916e-08),
        std::complex<double>(5.298577332100e-17, -5.959189144748e-09),
        std::complex<double>(-3.009745230820e-16, -3.501207830137e-09),
        std::complex<double>(-9.186228167035e-17, -1.915294608676e-09),
        std::complex<double>(2.055647319033e-16, -8.879601871852e-10),
        std::complex<double>(4.123369939761e-16, -4.698662414010e-10),
        std::complex<double>(-3.139849491518e-16, -3.309931637583e-10),
        std::complex<double>(5.506066522859e-17, -1.816820400721e-10)};

// Fourier coefficients for the GELU approximation in [-8, 8] of degree 44
static const std::vector<std::complex<double>> coeff_gelu_8_double_44{
        std::complex<double>(1.970461314322e+00, 0.000000000000e+00),
        std::complex<double>(-1.593875111273e+00, -2.423646025475e+00),
        std::complex<double>(-5.332570094236e-02, 1.043627310063e+00),
        std::complex<double>(-1.570391974449e-01, -5.402873054677e-01),
        std::complex<double>(-3.940694771457e-02, 2.819289867863e-01),
        std::complex<double>(-4.716127625363e-02, -1.391268773786e-01),
        std::complex<double>(-2.418730432251e-02, 6.233923043025e-02),
        std::complex<double>(-1.941167643365e-02, -2.432937320475e-02),
        std::complex<double>(-1.260556521786e-02, 7.763676253796e-03),
        std::complex<double>(-8.664960605021e-03, -1.754420766855e-03),
        std::complex<double>(-5.704614002479e-03, 1.292565051534e-04),
        std::complex<double>(-3.651093482904e-03, 8.792842531399e-05),
        std::complex<double>(-2.270560959509e-03, -3.329062052015e-05),
        std::complex<double>(-1.367226398367e-03, -2.302378982580e-06),
        std::complex<double>(-7.979823621962e-04, 4.854603433774e-06),
        std::complex<double>(-4.528446723700e-04, -5.500080713671e-07),
        std::complex<double>(-2.508556888546e-04, -7.491453056646e-07),
        std::complex<double>(-1.363438350097e-04, 2.239517241233e-07),
        std::complex<double>(-7.312414261638e-05, 1.280298946860e-07),
        std::complex<double>(-3.885704183314e-05, -6.935818435338e-08),
        std::complex<double>(-2.045210744808e-05, -2.372215028579e-08),
        std::complex<double>(-1.059289000752e-05, 2.148573735963e-08),
        std::complex<double>(-5.323541994507e-06, 4.532466957308e-09),
        std::complex<double>(-2.534606159357e-06, -7.005649868379e-09),
        std::complex<double>(-1.095525020071e-06, -8.007947707722e-10),
        std::complex<double>(-3.890401084408e-07, 2.430954434496e-09),
        std::complex<double>(-7.123273537061e-08, 8.161443049159e-11),
        std::complex<double>(5.044242802621e-08, -8.975149646921e-10),
        std::complex<double>(8.120772961289e-08, 3.455961905541e-11),
        std::complex<double>(7.542928368715e-08, 3.512492108152e-10),
        std::complex<double>(5.884712616061e-08, -3.699939660207e-11),
        std::complex<double>(4.212943410103e-08, -1.450199807245e-10),
        std::complex<double>(2.870135674255e-08, 2.446543662105e-11),
        std::complex<double>(1.893829504329e-08, 6.286847323116e-11),
        std::complex<double>(1.220565836477e-08, -1.443831859627e-11),
        std::complex<double>(7.712072154498e-09, -2.849185102259e-11),
        std::complex<double>(4.784403550098e-09, 8.229923686986e-12),
        std::complex<double>(2.915358732203e-09, 1.344066297062e-11),
        std::complex<double>(1.744138401955e-09, -5.307266236734e-12),
        std::complex<double>(1.034329697769e-09, -6.648169766598e-12),
        std::complex<double>(6.098917724102e-10, 3.421664931631e-12),
        std::complex<double>(3.589237189208e-10, 2.944192267787e-12),
        std::complex<double>(2.213863333275e-10, -2.779015756058e-12),
        std::complex<double>(1.478126284614e-10, -1.420742295235e-12),
        std::complex<double>(1.014093012815e-10, 4.286355138519e-13)};

}  // namespace fe_func_coefficients

#endif  // FIDESLIB_TEST_FE_FUNC_COEFFICIENTS_HPP
