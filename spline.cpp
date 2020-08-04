#include <iostream>
#include <vector>
#include <algorithm> // std::fill
#include <limits>
#include <random>
#include <numeric>
#include <assert.h>     /* assert */
#include "cxxopts.hpp"
#include <chrono>

struct Ugrid
{
  double start, end;
  int num;
  double delta, delta_inv;
};

// Base type is 'multi_UBspline_3d_s'
// Removed unused argument. Will need to pad to the correct dimension for more realism
struct SplineType
{
  intptr_t x_stride, y_stride, z_stride;
  Ugrid x_grid, y_grid, z_grid;
};


template<typename T, typename TRESIDUAL>
inline void getSplineBound(T x, TRESIDUAL& dx, int& ind, int nmax)
{
  // lower bound
  if (x < 0)
  {
    ind = 0;
    dx  = T(0);
  }
  else
  {
    T ipart;
    dx  = std::modf(x, &ipart);
    ind = static_cast<int>(ipart);
    // upper bound
    if (ind > nmax)
    {
      ind = nmax;
      dx  = T(1) - std::numeric_limits<T>::epsilon();
    }
  }
}

template<typename T>
inline static void compute_prefactors(T a[4], T da[4], T d2a[4], T tx)
{
  //48 total
  static constexpr T A00 = -1.0/6.0, A01 =  3.0/6.0, A02 = -3.0/6.0, A03 = 1.0/6.0;
  static constexpr T A10 =  3.0/6.0, A11 = -6.0/6.0, A12 =  0.0/6.0, A13 = 4.0/6.0;
  static constexpr T A20 = -3.0/6.0, A21 =  3.0/6.0, A22 =  3.0/6.0, A23 = 1.0/6.0;
  static constexpr T A30 =  1.0/6.0, A31 =  0.0/6.0, A32 =  0.0/6.0, A33 = 0.0/6.0;
  static constexpr T dA01 = -0.5, dA02 =  1.0, dA03 = -0.5;
  static constexpr T dA11 =  1.5, dA12 = -2.0, dA13 =  0.0;
  static constexpr T dA21 = -1.5, dA22 =  1.0, dA23 =  0.5;
  static constexpr T dA31 =  0.5, dA32 =  0.0, dA33 =  0.0;
  static constexpr T d2A02 = -1.0, d2A03 =  1.0;
  static constexpr T d2A12 =  3.0, d2A13 = -2.0;
  static constexpr T d2A22 = -3.0, d2A23 =  1.0;
  static constexpr T d2A32 =  1.0, d2A33 =  0.0;

  a[0]   = ((A00 * tx + A01) * tx + A02) * tx + A03;
  a[1]   = ((A10 * tx + A11) * tx + A12) * tx + A13;
  a[2]   = ((A20 * tx + A21) * tx + A22) * tx + A23;
  a[3]   = ((A30 * tx + A31) * tx + A32) * tx + A33;
  da[0]  = (dA01 * tx + dA02) * tx + dA03;
  da[1]  = (dA11 * tx + dA12) * tx + dA13;
  da[2]  = (dA21 * tx + dA22) * tx + dA23;
  da[3]  = (dA31 * tx + dA32) * tx + dA33;
  d2a[0] = d2A02 * tx + d2A03;
  d2a[1] = d2A12 * tx + d2A13;
  d2a[2] = d2A22 * tx + d2A23;
  d2a[3] = d2A32 * tx + d2A33;
}

template<typename T>
inline void computeLocationAndFractional(
    const SplineType* __restrict__ spline_m, T x, T y, T z,
    int& ix, int& iy, int& iz, T a[4], T b[4], T c[4], T da[4], T db[4], T dc[4], T d2a[4],
    T d2b[4], T d2c[4])
{
  //150 total?
  //3?
  x -= spline_m->x_grid.start;
  y -= spline_m->y_grid.start;
  z -= spline_m->z_grid.start;

  T tx, ty, tz;

  //3?
  getSplineBound(x * spline_m->x_grid.delta_inv, tx, ix, spline_m->x_grid.num - 1);
  getSplineBound(y * spline_m->y_grid.delta_inv, ty, iy, spline_m->y_grid.num - 1);
  getSplineBound(z * spline_m->z_grid.delta_inv, tz, iz, spline_m->z_grid.num - 1);

  //144
  compute_prefactors(a, da, d2a, tx);
  compute_prefactors(b, db, d2b, ty);
  compute_prefactors(c, dc, d2c, tz);
}

inline void evaluate_vgh(const SplineType* __restrict__ spline_m,
                         float* __restrict__ coefs_m,
                         float x,
                         float y,
                         float z, 
                         float* __restrict__ vals,
                         float* __restrict__ grads,
                         float* __restrict__ hess,
                         size_t n_splines_local)
{
  int ix, iy, iz;
  float a[4], b[4], c[4], da[4], db[4], dc[4], d2a[4], d2b[4], d2c[4];

  // cLAndF(150?) + 102 + 665*n_splines

  //150?
  computeLocationAndFractional(spline_m, x, y, z, ix, iy, iz, a, b, c, da, db, dc, d2a, d2b, d2c);

  const intptr_t xs = spline_m->x_stride;
  const intptr_t ys = spline_m->y_stride;
  const intptr_t zs = spline_m->z_stride;


  // gx,gy,gz each size [n_splines_local]
  // grads = [gx,gy,gz]
  // 3(ptr)
  float* __restrict__ gx = grads;
  float* __restrict__ gy = grads + n_splines_local;
  float* __restrict__ gz = grads + 2 * n_splines_local;

  // 9(ptr)
  float* __restrict__ hxx = hess;
  float* __restrict__ hxy = hess + n_splines_local;
  float* __restrict__ hxz = hess + 2 * n_splines_local;
  float* __restrict__ hyy = hess + 3 * n_splines_local;
  float* __restrict__ hyz = hess + 4 * n_splines_local;
  float* __restrict__ hzz = hess + 5 * n_splines_local;

  // 16*(6+41*n_splines)
  // 96 + 656*n_splines
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
    {
      // 13(ptr)
      // coefs has dims [ nx+3, ny+3, nz+3, n_splines_local ]
      // explicit outer loops over x and y, unroll inner loop over z
      const float* __restrict__ coefs =  coefs_m + ((ix + i) * xs + (iy + j) * ys + iz * zs);
      const float* __restrict__ coefszs = coefs + zs;
      const float* __restrict__ coefs2zs = coefs + 2 * zs;
      const float* __restrict__ coefs3zs = coefs + 3 * zs;

      // 6
      const float pre20 = d2a[i] * b[j];
      const float pre10 = da[i] * b[j];
      const float pre00 = a[i] * b[j];
      const float pre11 = da[i] * db[j];
      const float pre01 = a[i] * db[j];
      const float pre02 = a[i] * d2b[j];

      //41*n_splines
      for (int n = 0; n < n_splines_local; n++)
      {
       float coefsv    = coefs[n];
       float coefsvzs  = coefszs[n];
       float coefsv2zs = coefs2zs[n];
       float coefsv3zs = coefs3zs[n];

       // 21
       float sum0 = c[0] * coefsv + c[1] * coefsvzs + c[2] * coefsv2zs + c[3] * coefsv3zs;
       float sum1 = dc[0] * coefsv + dc[1] * coefsvzs + dc[2] * coefsv2zs + dc[3] * coefsv3zs;
       float sum2 = d2c[0] * coefsv + d2c[1] * coefsvzs + d2c[2] * coefsv2zs + d2c[3] * coefsv3zs;

       // 20
       vals[n] += pre00 * sum0;
       gx[n] += pre10 * sum0;
       gy[n] += pre01 * sum0;
       gz[n] += pre00 * sum1;
       hxx[n] += pre20 * sum0;
       hxy[n] += pre11 * sum0;
       hxz[n] += pre10 * sum1;
       hyy[n] += pre02 * sum0;
       hyz[n] += pre01 * sum1;
       hzz[n] += pre00 * sum2;
      }
    }

  const float dxInv = spline_m->x_grid.delta_inv;
  const float dyInv = spline_m->y_grid.delta_inv;
  const float dzInv = spline_m->z_grid.delta_inv;
  // 6
  const float dxx   = dxInv * dxInv;
  const float dxy   = dxInv * dyInv;
  const float dxz   = dxInv * dzInv;
  const float dyy   = dyInv * dyInv;
  const float dyz   = dyInv * dzInv;
  const float dzz   = dzInv * dzInv;

  //9*n_splines
  for (size_t n = 0; n < n_splines_local; n++)
  {
    gx[n] *= dxInv;
    gy[n] *= dyInv;
    gz[n] *= dzInv;
    hxx[n] *= dxx;
    hxy[n] *= dxy;
    hxz[n] *= dxz;
    hyy[n] *= dyy;
    hyz[n] *= dyz;
    hzz[n] *= dzz;
  }

}

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  cxxopts::Options options("nanoqmc",
                           "miniapps of miniqmc");

  options.add_options()
          ("h,help", "Print help")
          ("x", "Size of spline x", cxxopts::value<int>()->default_value("37"))
          ("y", "Size of spline y", cxxopts::value<int>()->default_value("37"))
          ("z", "Size of spline z", cxxopts::value<int>()->default_value("37"))
          ("e,", "Number of electron", cxxopts::value<int>()->default_value("20"))
          ("b,nblock", "Number of block", cxxopts::value<int>()->default_value("1"))
          ("n,niter", "Number of dummy iterations", cxxopts::value<int>()->default_value("1"))
          ("v,verbose", "output level", cxxopts::value<int>()->default_value("2"))
  ;

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  const int verbose = result["v"].as<int>();
  const int nx = result["x"].as<int>();
  const int ny = result["y"].as<int>();
  const int nz = result["z"].as<int>();
  const int nelectrons =  result["e"].as<int>();
  const int niter = result["n"].as<int>();

  //Declaration
  const int norb = nelectrons  / 2;
  const int n_splines = norb;
  const int mx = nx+3;
  const int my = ny+3;
  const int mz = nz+3;
  const int ngridpts = mx*my*mz;
  const int n_coef = n_splines * ngridpts;

  const int nblock = result["nblock"].as<int>();
  assert ( n_splines%nblock == 0);
  
  const int n_splines_block = n_splines / nblock;
  const int n_coef_block = n_coef / nblock;
  
  std::vector<double> l_start{0.,0.,0.};   
  std::vector<double> l_end{1.,1.,1.};
  std::vector<int> l_num{nx,ny,nz};

  std::vector<double> l_delta(3);
  std::vector<double> l_delta_inv(3);

  for (int i=0; i<3; i++){
    l_delta[i] =  ( l_end[i] - l_start[i] ) / (double)(l_num[i]) ;
    l_delta_inv[i] = 1./l_delta[i];
  }

  // coefs is array with dims: [n_blocks, nx+3, ny+3, nz+3, n_splines_per_block] 
  std::vector<float> coefs(n_coef);
  std::vector<std::vector<float>> vals(nelectrons, std::vector<float>(n_splines));
  std::vector<std::vector<float>> grads(nelectrons, std::vector<float>(n_splines*3));
  std::vector<std::vector<float>> hess(nelectrons, std::vector<float>(n_splines*6));

  std::vector<float> electron_pos_x(nelectrons);
  std::vector<float> electron_pos_y(nelectrons);
  std::vector<float> electron_pos_z(nelectrons);

  // Put random values
  std::uniform_real_distribution<float> distribution(0.,1.);
  std::minstd_rand generator(0);

  // initialize by block to allow comparison with different number of blocks
  for (int ir=0; ir < ngridpts; ir++){
    for (int b=0; b < nblock; b++){
      int block_start = n_splines_block*(b*ngridpts + ir);
      std::generate_n(coefs.begin() + block_start,n_splines_block, [&distribution, &generator]() { return distribution(generator); });
    }
  }
  std::generate(electron_pos_x.begin(), electron_pos_x.end(), [&distribution, &generator]() { return distribution(generator); });
  std::generate(electron_pos_y.begin(), electron_pos_y.end(), [&distribution, &generator]() { return distribution(generator); });
  std::generate(electron_pos_z.begin(), electron_pos_z.end(), [&distribution, &generator]() { return distribution(generator); });



  // coefs layout per block is [mx, my, mz, n_splines_block], so in a flattened array:
  // coefs[ix,iy,iz,ispl] = coefs[ispl + nspl*(iz + mz*(iy + my*(ix)))]
  //                             [ispl + iz*nspl + iy*nspl*mz + iz*my*mz*nspl]
  auto s = SplineType {  n_splines_block*(nz+3)*(ny+3), n_splines_block*(nz+3), n_splines_block,
                         Ugrid{ l_start[0], l_end[0], l_num[0], l_delta[0], l_delta_inv[0] },
                         Ugrid{ l_start[1], l_end[1], l_num[1], l_delta[1], l_delta_inv[1] },
                         Ugrid{ l_start[2], l_end[2], l_num[2], l_delta[2], l_delta_inv[2] },
                      } ;

  auto start = std::chrono::system_clock::now();

  double nflop = 1.*niter*nelectrons*nblock*(252+665*n_splines_block);

  /*
   * sizes:
   * (float) coefs        : (nx+3) * (ny+3) * (nz+3) * nelectrons/2
   * (float) electron_pos : 3 * nelectrons
   * (float) vals         : nelectrons*
   */

  // Kernel
  // TODO: add walker loop
  for (int i=0; i< niter; i++) {
    for (int e=0 ; e < nelectrons ; e++){
      std::fill(vals[e].begin(), vals[e].end(), float());
      std::fill(grads[e].begin(), grads[e].end(), float());
      std::fill(hess[e].begin(), hess[e].end(), float());
    }
    //nelectrons*nblock*(cLAndF + 102 + 665*n_splines_block)
    //nelectrons*nblock*(252 + 665*n_splines_block)
    // loop over electrons
    #pragma omp parallel for collapse(2)
    for (int e=0 ; e < nelectrons ; e++){
      // loop over blocks
      for (int b=0; b < nblock; b++) {
        int block_idx_start = b*n_splines_block;
        //transfer: 1 spline, coef block, elec pos (3), vgh block
        //
        evaluate_vgh(&s, coefs.data()+block_idx_start*ngridpts,
            electron_pos_x[e], electron_pos_y[e], electron_pos_z[e],
            vals[e].data()+block_idx_start,
            grads[e].data()+3*block_idx_start,
            hess[e].data()+6*block_idx_start,
            n_splines_block);
      }
    }
  }
  auto end = std::chrono::system_clock::now();
  double walltime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "flop count: " << nflop << std::endl;
  std::cout << "time: " << walltime << " ns" << std::endl;
  std::cout << "GFLOPS: " << nflop/walltime << std::endl;
  if (verbose > 1){
    std::cout << "val:  " << std::accumulate(vals[0].begin(), vals[0].end(), 0.) << std::endl;
    std::cout << "grad: " << std::accumulate(grads[0].begin(), grads[0].end(), 0.) << std::endl;
    std::cout << "hess: " << std::accumulate(hess[0].begin(), hess[0].end(), 0.) << std::endl;
  }
  return 0;
}

