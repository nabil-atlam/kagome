# Author: Nabil 
VERSION >= v"1.0.0-alpha" && __precompile__()

module Kagome_Hamiltonian 
using Parameters, StaticArrays, LinearAlgebra
using Enzyme

export H, H3, evals3, evals3c, eigensystem3c, Velx, Vely, real_basis, recip_basis, Params

# MODEL PARAMETERS
@with_kw struct Params
    t1::Float64 = 1.0
    t2::Float64 = 0.0
    u1::Float64 = 0.0
    u2::Float64 = 0.0
end

# NBANDS
nbands = 3

# REAL BASIS 
global const lattice_constant::Float64 = 1.0
global const real_basis = SMatrix{2, 2}(1.0 * lattice_constant * Float64[1.0 0.0; 0.5 0.5*sqrt(3.0)])
# RECIPROCAL BASIS 
global const recip_basis::SMatrix{2, 2, Float64, 4} = inv(real_basis)'

# GELL-MANN MATRICES
global const λ0  = @SArray ComplexF64[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
global const λ1  = @SArray ComplexF64[0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
global const λ2  = @SArray ComplexF64[0.0 -im 0.0; im 0.0 0.0; 0.0 0.0 0.0]
global const λ3  = @SArray ComplexF64[1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0]
global const λ4  = @SArray ComplexF64[0.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0]
global const λ5  = @SArray ComplexF64[0.0 0.0 -im; 0.0 0.0 0.0; im 0.0 0.0]
global const λ6  = @SArray ComplexF64[0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
global const λ7  = @SArray ComplexF64[0.0 0.0 0.0; 0.0 0.0 -im; 0.0 im 0.0]
global const λ8  = (@SArray ComplexF64[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 -2.0]) / sqrt(3)


# COMPUTE A MORE CONVENIENT SET OF GENERATORS. THESE BEHAVE SOMEWHAT AS FERMIONIC Z_2 GRADED HARMONIC OSCILLARS WITHIN 2x2 MATRIX SUBSECTORS   
global const T1u_ = 0.5 * (λ1 + im * λ2)
global const T2u_ = 0.5 * (λ4 + im * λ5) 
global const T3u_ = 0.5 * (λ6 + im * λ7) 

global const T1u  = @SArray ComplexF64[0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
global const T2u  = @SArray ComplexF64[0.0 0.0 1.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
global const T3u  = @SArray ComplexF64[0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 0.0 0.0]


# SPIN 
sx = @SArray ComplexF64[0.0 1.0 ; 1.0 0.0]
sy = @SArray ComplexF64[0.0 -im ; im 0.0]
sz = @SArray ComplexF64[1.0 0.0 ; 0.0 -1.0]
sp = @SArray ComplexF64[0.0 1.0 ; 0.0 0.0]
sm = @SArray ComplexF64[0.0 0.0 ; 1.0 0.0]

# LAYER
τp = sp; τm = sm
τx = sx; τy = sy; τz = sz 



###################################################################################################
###################################################################################################
###################################################################################################
# Φ FUNCTIONS TAKING CARTESIAN 
# Remark:  k2  = -1/2 kx - 1 / 2 sqrt(3) ky          -0.5 * kx - 0.5 * sqrt(3.0) * ky
#       : -k3  =  1/2 kx - 1 / 2 sqrt(3) ky           0.5 * kx - 0.5 * sqrt(3.0) * ky

@inline function fdΦ1dkx_c(kx, ky, p)
    pp = 0.5 * kx + 0.5 * sqrt(3.0) * ky; pm = 0.5 * kx - 0.5 * sqrt(3.0) * ky
    nn = -sin(kx) + im*cos(kx)
    nnn = -0.5 * (sin(pp) + sin(pm)) + 0.5 * im * (cos(pp) + cos(pm))
    (p.t1 + im * p.u1) * nn + (p.t2 - im * p.u2) * nnn
end

@inline function fdΦ1dky_c(kx, ky, p)
    pp = 0.5 * kx + 0.5 * sqrt(3.0) * ky; pm = 0.5 * kx - 0.5 * sqrt(3.0) * ky
    nnn = -0.5 * sqrt(3.0) * sin(pp) + 0.5 * sqrt(3.0) * sin(pm) + im * 0.5 * sqrt(3.0) * (cos(pp) - cos(pm))
   (p.t2 - im * p.u2) * nnn
end



@inline function fdΦ2dkx_c(kx, ky, p)
    pp = 0.5 * kx + 0.5 * sqrt(3.0) * ky; pm = -0.5 * kx + 0.5 * sqrt(3.0) * ky
    nn = -0.5 * sin(pp) + 0.5 * im *cos(pp)
    nnn = 0.5 * sin(pm) - sin(kx) + im * (cos(kx) - 0.5 * cos(pm))
    (p.t1 - im * p.u1) * nn + (p.t2 + im * p.u2) * nnn
end

@inline function fdΦ2dky_c(kx, ky, p)
    pp = 0.5 * kx + 0.5 * sqrt(3.0) * ky; pm = -0.5 * kx + 0.5 * sqrt(3.0) * ky
    nn = -0.5 * sqrt(3.0) * sin(pp) + 0.5 * sqrt(3.0) * im * cos(pp)
    nnn = -0.5 * sqrt(3.0) * sin(pm) + 0.5 * sqrt(3.0) * im*cos(pm)
    (p.t1 - im * p.u1) * nn + (p.t2 + im * p.u2) * nnn



@inline function fdΦ3dkx_c(kx, ky, p)
    pp = 0.5 * kx + 0.5 * sqrt(3.0) * ky; pm = -0.5 * kx + 0.5 * sqrt(3.0) * ky
    nn = 0.5 * sin(pm) - 0.5 * im * cos(pm)
    nnn = sin(-kx) - 0.5 * sin(pp) + im * (-cos(kx) + 0.5 * cos(pp))
    (p.t1 + im * p.u1) * nn + (p.t2 - im * p.u2) * nnn
end

@inline function fdΦ3dky_c(kx, ky, p)
    pp = 0.5 * kx + 0.5 * sqrt(3.0) * ky; pm = -0.5 * kx + 0.5 * sqrt(3.0) * ky
    nn = -0.5 * sqrt(3.0) * sin(pm) + 0.5 * sqrt(3.0) * im*cos(pm)
    nnn = -0.5 * sqrt(3.0) * sin(pp) + 0.5 * sqrt(3.0) * im*cos(pp)
    (p.t1 + im * p.u1) * nn + (p.t2 - im * p.u2) * nnn
end
###################################################################################################
###################################################################################################
###################################################################################################



# RATIONAL POLYNOMIALS -- Here, the couplings are complexified due to spin-orbit effects 
@inline ϕ1(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t1 + im * p.u1) * (1 + z1) + (p.t2 - im * p.u2) * (z2 + z1 / z2)
@inline ϕ2(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t1 - im * p.u1) * (1 + z2) + (p.t2 + im * p.u2) * (z1 + z2 / z1)
@inline ϕ3(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t1 + im * p.u1) * (1 + z2 / z1) + (p.t2 - im * p.u2) * ((1 / z1) + z2)

# RELATION BETWEEN Z VARIABLES AND K VARIABLES 
@inline z1(k1::Float64) = exp(2 * pi * im * k1)
@inline z1(k2::Float64) = exp(2 * pi * im * k2)


@inline ϕ1(k1::Float64, k2::Float64, p::Params) = begin
    (p.t1 + im * p.u1) * (1 + exp(2 * pi * im * k1)) + (p.t2 - im * p.u2) * (exp(2 * pi * im * k2) + exp(2 * pi * im * (k1 - k2)))
end


@inline ϕ2(k1::Float64, k2::Float64, p::Params) = begin
    (p.t1 - im * p.u1) * (1 + exp(2 * pi * im * k2)) + (p.t2 + im * p.u2) * (exp(2 * pi * im * k1) + exp(-2 * pi * im * (k1 - k2)))
end

@inline ϕ3(k1::Float64, k2::Float64, p::Params) = begin
    (p.t1 + im * p.u1) * (1 + exp(-2 * pi * im * (k1 - k2))) + (p.t2 - im * p.u2) * (exp(-2 * pi * im * k1) + exp(2 * pi * im * k2))
end


# These take cartesian coordinates 

@inline ϕ1c(kx::Float64, ky::Float64, p::Params) = begin
    pp = 0.5 * kx + 0.8660254037844386 * ky; pm = 0.5 * kx - 0.8660254037844386 * ky
    
    (p.t1 + im * p.u1) * (1.0 + cos(kx) + im * sin(kx)) + (p.t2 - im * p.u2) * (cos(pp) + cos(pm) + im*(sin(pp) + sin(pm)))

end


@inline ϕ2c(kx::Float64, ky::Float64, p::Params) = begin
    pp = 0.5 * kx + 0.8660254037844386 * ky; pm = 0.5 * kx - 0.8660254037844386 * ky

    (p.t1 - im * p.u1) * (1.0 + cos(pp) + im * sin(pp)) + (p.t2 + im * p.u2) * (cos(kx) + cos(pm) + im * (-sin(pm) + sin(kx)))
end

@inline ϕ3c(kx::Float64, ky::Float64, p::Params) = begin
    pp = 0.5 * kx + 0.8660254037844386 * ky; pm = -0.5 * kx + 0.8660254037844386 * ky
    (p.t1 + im * p.u1) * (1.0 + cos(pm) + im*sin(pm)) + (p.t2 - im * p.u2) * (cos(-kx) + cos(pp) + im * (sin(-kx) + sin(pp)))
end


############################ HAMILTONIAN #######################################################
@inline function evals3(k::Vector{Float64}, p::Params)
    k1 = k[1]; k2 = k[2]
    H_ut::Matrix{ComplexF64} = ϕ1(k1, k2, p) * T1u + ϕ2(k1, k2, p) * T2u + ϕ3(k1, k2, p) * T3u
    eigvals(H_ut + H_ut')
end

@inline function evals3c(k::Vector{Float64}, p::Params)
    k1 = k[1]; k2 = k[2]
    H_ut::Matrix{ComplexF64} = ϕ1c(k1, k2, p) * T1u + ϕ2c(k1, k2, p) * T2u + ϕ3c(k1, k2, p) * T3u
    eigvals(H_ut + H_ut')
end

@inline function eigensystem3c(k::Vector{Float64}, p::Params)
    k1 = k[1]; k2 = k[2]
    H_ut::Matrix{ComplexF64} = ϕ1c(k1, k2, p) * T1u + ϕ2c(k1, k2, p) * T2u + ϕ3c(k1, k2, p) * T3u
    eigen(Hermitian(H_ut + H_ut'))
end



@inline function H3(k::Vector{Float64}, p::Params)
    k1 = k[1]; k2 = k[2]
    H_ut::Matrix{ComplexF64} = ϕ1(k1, k2, p) * T1u + ϕ2(k1, k2, p) * T2u + ϕ3(k1, k2, p) * T3u
    H_ut + H_ut'
end


############################ VELOCITIES ###################################################
@inline function Velx(kx::Float64, ky::Float64, p::Params)
    V::Matrix{ComplexF64} = fdΦ1dkx_c(kx, ky, p) * T1u + fdΦ2dkx_c(kx, ky, p) * T2u + fdΦ3dkx_c(kx, ky, p) * T3u
    V + V'
end

@inline function Vely(kx::Float64, ky::Float64, p::Params)
    V::Matrix{ComplexF64} = fdΦ1dky_c(kx, ky, p) * T1u + fdΦ2dky_c(kx, ky, p) * T2u + fdΦ3dky_c(kx, ky, p) * T3u
    V + V'
end


@inline function H(z1::ComplexF64, z2::ComplexF64, p::Params, T1u::Matrix{ComplexF64}, T2u::Matrix{ComplexF64}, T3u::Matrix{ComplexF64})
    H_ut::Matrix{ComplexF64} = ϕ1(z1, z2, p) * T1u + ϕ2(z1, z2, p) * T2u + ϕ3(z1, z2, p) * T3u
    H_ut + H_ut'
end

@inline function H(k1::Float64, k2::Float64, p::Params, T1u::Matrix{ComplexF64}, T2u::Matrix{ComplexF64}, T3u::Matrix{ComplexF64})
    H_ut::Matrix{ComplexF64} = ϕ1(k1, k2, p) * T1u + ϕ2(k1, k2, p) * T2u + ϕ3(k1, k2, p) * T3u
    H_ut + H_ut'
end
@inline function H(k1::Float64, k2::Float64, p::Params)
    H_ut::Matrix{ComplexF64} = ϕ1(k1, k2, p) * T1u + ϕ2(k1, k2, p) * T2u + ϕ3(k1, k2, p) * T3u
    H_ut + H_ut'
end


@inline function H(z1::ComplexF64, z2::ComplexF64, p::Params, T1u::SMatrix{ComplexF64}, T2u::SMatrix{ComplexF64}, T3u::SMatrix{ComplexF64})
    H_ut::SMatrix{3, 3, ComplexF64, 9} = ϕ1(z1, z2, p) * T1u + ϕ2(z1, z2, p) * T2u + ϕ3(z1, z2, p) * T3u
    H_ut + H_ut'
end

@inline function H(k1::Float64, k2::Float64, p::Params, T1u::SMatrix{ComplexF64}, T2u::SMatrix{ComplexF64}, T3u::SMatrix{ComplexF64})
    H_ut::SMatrix{3, 3, ComplexF64, 9} = ϕ1(k1, k2, p) * T1u + ϕ2(k1, k2, p) * T2u + ϕ3(k1, k2, p) * T3u
    H_ut + H_ut'
end


# PROCEED IN TWO WAYS : DIFFERENTIAL BY HAND (SIMPLE HERE) OR USE  AUTODIFF (TO LEARN THE TECH)

# Differentiation by Hand #  
@inline dϕ1dz1(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t1 + im * p.u1) + (p.t2 - im * p.u2) / z2
@inline dϕ1dz2(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t2 - im * p.u2) * (1 - (z1 / (z2^2)))

@inline dϕ2dz1(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t2 + im * p.u2) * (1 - (z2 / (z1^2)))
@inline dϕ2dz2(z1::ComplexF64, z2::ComplexF64, p::Params) = (p.t1 - im * p.u1) + (p.t2 + im * p.u2) / z1

@inline dϕ3dz1(z1::ComplexF64, z2::ComplexF64, p::Params) = -(p.t1 + im * p.u1) * z2 / (z1^2) - (p.t2 - im * p.u2) / (z1^2)
@inline dϕ3dz2(z1::ComplexF64, z2::ComplexF64, p::Params) =  (p.t1 + im * p.u1) / z1 + (p.t2 - im * p.u2)

@inline dϕ1dk1(z1::ComplexF64, z2::ComplexF64, p::Params) = 2 * pi * im * z1 * ((p.t1 + im * p.u1) + (p.t2 - im * p.u2) / z2)
@inline dϕ1dk2(z1::ComplexF64, z2::ComplexF64, p::Params) = 2 * pi * im * z2 * ((p.t2 - im * p.u2) * (1 - (z1 / (z2^2))))

@inline dϕ2dk1(z1::ComplexF64, z2::ComplexF64, p::Params) = 2 * pi * im * z1 * ((p.t2 + im * p.u2) * (1 - (z2 / (z1^2))))
@inline dϕ2dk2(z1::ComplexF64, z2::ComplexF64, p::Params) = 2 * pi * im * z2 * ((p.t1 - im * p.u1) + (p.t2 + im * p.u2) / z1)

@inline dϕ3dk1(z1::ComplexF64, z2::ComplexF64, p::Params) = 2 * pi * im * z1 * (-(p.t1 + im * p.u1) * z2 / (z1^2) - (p.t2 - im * p.u2) / (z1^2))
@inline dϕ3dk2(z1::ComplexF64, z2::ComplexF64, p::Params) = 2 * pi * im * z2 * ((p.t1 + im * p.u1) / z1 + (p.t2 - im * p.u2))

# partial_kxky in terms of partial_k1k2 
global Jacobian_Matrix::SMatrix{2, 2, Float64, 4} = [1.0 0.5; 0.0 sqrt(3)/2.0]


# USE THE JACOBIAN TO EXPRESS THE DIFFERENTIALS IN CARTESIAN COORDINATES 
@inline dϕ1dkx(z1::ComplexF64, z2::ComplexF64, p::Params) = dϕ1dk1(z1, z2, p) + 0.5 * dϕ1dk2(z1, z2, p)
@inline dϕ1dky(z1::ComplexF64, z2::ComplexF64, p::Params) = (sqrt(3) / 2.0) * dϕ1dk2(z1, z2, p)

@inline dϕ2dkx(z1::ComplexF64, z2::ComplexF64, p::Params) = dϕ2dk1(z1, z2, p) + 0.5 * dϕ2dk2(z1, z2, p)
@inline dϕ2dky(z1::ComplexF64, z2::ComplexF64, p::Params) = (sqrt(3) / 2.0) * dϕ2dk2(z1, z2, p)

@inline dϕ3dkx(z1::ComplexF64, z2::ComplexF64, p::Params) = dϕ3dk1(z1, z2, p) + 0.5 * dϕ3dk2(z1, z2, p)
@inline dϕ3dky(z1::ComplexF64, z2::ComplexF64, p::Params) = (sqrt(3) / 2.0) * dϕ3dk2(z1, z2, p)


# ORBITAL VELOCITY MATRICES #
# ------------------------- #
@inline function Vx(z1::ComplexF64, z2::ComplexF64, p::Params, T1u::Matrix{ComplexF64}, T2u::Matrix{ComplexF64}, T3u::Matrix{ComplexF64})
    H_ut::Matrix{ComplexF64} = dϕ1dkx(z1, z2, p) * T1u + dϕ2dkx(z1, z2, p) * T2u + dϕ3dkx(z1, z2, p) * T3u
    H_ut + H_ut'
end

@inline function Vy(z1::ComplexF64, z2::ComplexF64, p::Params, T1u::Matrix{ComplexF64}, T2u::Matrix{ComplexF64}, T3u::Matrix{ComplexF64})
    H_ut::Matrix{ComplexF64} = dϕ1dky(z1, z2, p) * T1u + dϕ2dky(z1, z2, p) * T2u + dϕ3dky(z1, z2, p) * T3u
    H_ut + H_ut'
end

# USING SMatrix 
@inline function Vx(z1::ComplexF64, z2::ComplexF64, p::Params, T1u::SMatrix{ComplexF64}, T2u::SMatrix{ComplexF64}, T3u::SMatrix{ComplexF64})
    H_ut::SMatrix{3, 3, ComplexF64, 9} = dϕ1dkx(z1, z2, p) * T1u + dϕ2dkx(z1, z2, p) * T2u + dϕ3dkx(z1, z2, p) * T3u
    H_ut + H_ut'
end

@inline function Vy(z1::ComplexF64, z2::ComplexF64, p::Params, T1u::SMatrix{ComplexF64}, T2u::SMatrix{ComplexF64}, T3u::SMatrix{ComplexF64})
    H_ut::SMatrix{3, 3, ComplexF64, 9} = dϕ1dky(z1, z2, p) * T1u + dϕ2dky(z1, z2, p) * T2u + dϕ3dky(z1, z2, p) * T3u
    H_ut + H_ut'
end

end 