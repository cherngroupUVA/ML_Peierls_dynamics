using DelimitedFiles
using LinearAlgebra
using Random

function computeForceLattice!(dim, dimSq, k0, sp, κ, ux, uy, fx, fy)
    for idx in 1:dimSq
        iIdx = div(idx  - 1, dim) + 1
        jIdx = mod(idx  - 1, dim) + 1
        ## compute index of the neighbors
        lIdx = (iIdx - 1) * dim + (mod(jIdx - 2, dim) + 1)
        rIdx = (iIdx - 1) * dim + (mod(jIdx    , dim) + 1)
        tIdx = mod(idx -  1 - dim, dimSq) + 1
        bIdx = mod(idx -  1 + dim, dimSq) + 1
        fx[idx] = -k0 * ux[idx] + sp * (ux[lIdx] + ux[rIdx] - Float64(2.0) * ux[idx]) - κ * (ux[lIdx] + ux[rIdx] + ux[tIdx] + ux[bIdx])
        fy[idx] = -k0 * uy[idx] + sp * (uy[tIdx] + uy[bIdx] - Float64(2.0) * uy[idx]) - κ * (uy[lIdx] + uy[rIdx] + uy[tIdx] + uy[bIdx])
    end
    return nothing
end

function thermalizeSystem!(rng,
                           dim   :: Int64,
                           dimSq :: Int64,
                           tp    :: Float64,             # temperature of the system
                           gm    :: Float64,             # damping factor for Langevin dynamics
                           k0    :: Float64,             # on-site harmonic potential
                           sp    :: Float64,
                           κ,             # spring constant
                           ux    :: Vector{Float64},
                           uy    :: Vector{Float64},
                           vx    :: Vector{Float64},
                           vy    :: Vector{Float64},
                           fx    :: Vector{Float64},
                           fy    :: Vector{Float64},
                           dt    :: Float64,
                           nS    :: Int64)
    ## velocity Verlet method is used to update dynamics
    ## compute a(0)
    computeForceLattice!(dim, dimSq, k0, sp, κ, ux, uy, fx, fy)
    ## compute 0.5 x a(t) x dt
    dvx = Float64(0.5) * dt * fx
    dvy = Float64(0.5) * dt * fy
    for step = 1 : nS
        ## u(t + dt) = u(t) + v(t + 0.5 x dt) x dt
        ux .+= (vx + dvx) * dt
        uy .+= (vy + dvy) * dt
        ## compute a(t + dt)
        computeForceLattice!(dim, dimSq, k0, sp, κ, ux, uy, fx, fy)
        ## v(t + dt) = v(t) + 0.5 x (a(t) + a(t + dt)) x dt
        dvxNew = Float64(0.5) * dt * fx
        dvyNew = Float64(0.5) * dt * fy
        vx .+=   dvx +  dvxNew
        vy .+=   dvy +  dvyNew
        ## update v(t + 0.5 x dt)
        dvx = dvxNew
        dvy = dvyNew
        ## generate damping and noise for Langevin dynamics
        etax = randn(rng, dimSq)
        etay = randn(rng, dimSq)
        alp  = Float64(exp(-gm * dt))
        sig  = Float64(sqrt((1-alp * alp) * tp))
        ## update velocity using Langevin dynamics
        vx .= vx * alp + etax * sig
        vy .= vy * alp + etay * sig
#         vx = vx * alp
#         vy = vy * alp
    end
    return nothing
end


############################################################
function initLookupTable(dim, dimSq)
    lIdxTbl = Array{Int64}(undef, dimSq)
    rIdxTbl = Array{Int64}(undef, dimSq)
    tIdxTbl = Array{Int64}(undef, dimSq)
    bIdxTbl = Array{Int64}(undef, dimSq)
    trIdxTbl= Array{Int64}(undef, dimSq)
    tlIdxTbl= Array{Int64}(undef, dimSq)
    brIdxTbl= Array{Int64}(undef, dimSq)
    blIdxTbl= Array{Int64}(undef, dimSq)
    for idx in 1:dimSq

        iIdx = div(idx  - 1, dim) + 1
        jIdx = mod(idx  - 1, dim) + 1
        lIdx = (iIdx - 1) * dim + (mod(jIdx - 2, dim) + 1)
        rIdx = (iIdx - 1) * dim + (mod(jIdx    , dim) + 1)
        tIdx = mod(idx -  1 - dim, dimSq) + 1
        bIdx = mod(idx -  1 + dim, dimSq) + 1

        tlIdx = mod(mod(idx - dim - 2, dimSq) + (div(mod(idx  - 2, dim) + 1, dim) * dim), dimSq) + 1
        trIdx = mod(mod(idx - dim, dimSq)     - (div(jIdx, dim) * dim), dimSq) + 1
        blIdx = mod(mod(idx + dim - 2, dimSq) + (div(mod(idx  - 2, dim) + 1, dim) * dim), dimSq) + 1
        brIdx = mod(mod(idx + dim, dimSq)     - (div(jIdx, dim) * dim), dimSq) + 1

        lIdxTbl[idx]  = lIdx
        rIdxTbl[idx]  = rIdx
        tIdxTbl[idx]  = tIdx
        bIdxTbl[idx]  = bIdx

        tlIdxTbl[idx] = tlIdx
        trIdxTbl[idx] = trIdx
        blIdxTbl[idx] = blIdx
        brIdxTbl[idx] = brIdx
    end
    return lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl, tlIdxTbl, trIdxTbl, blIdxTbl, brIdxTbl
end


function initializeNNNHmn!(dimSq, t2, Hmn, tlIdxTbl, trIdxTbl, blIdxTbl, brIdxTbl)
    for idx in 1:dimSq
        Hmn[idx, tlIdxTbl[idx]] = -t2
        Hmn[idx, trIdxTbl[idx]] = -t2
        Hmn[idx, blIdxTbl[idx]] = -t2
        Hmn[idx, brIdxTbl[idx]] = -t2
    end
    return nothing
end

function updateHoppingHmn!(dimSq, t0, e_ph, ux, uy, Hmn, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl)
    for idx in 1:dimSq
        lIdx = lIdxTbl[idx]
        rIdx = rIdxTbl[idx]
        tIdx = tIdxTbl[idx]
        bIdx = bIdxTbl[idx]
        Hmn[idx, lIdx] = -(t0 - e_ph * (ux[idx] - ux[lIdx]))
        Hmn[idx, rIdx] = -(t0 - e_ph * (ux[rIdx] - ux[idx]))
        Hmn[idx, tIdx] = -(t0 + e_ph * (uy[idx] - uy[tIdx]))
        Hmn[idx, bIdx] = -(t0 + e_ph * (uy[bIdx] - uy[idx]))
    end
    return nothing
end

function computeEnergy(dimSq, sp, k0, ux, uy, vx, vy, lIdxTbl, tIdxTbl)
    knt  = Float64(0.0)
    ptn1 = Float64(0.0)
    ptn2 = Float64(0.0)
    for idx in 1:dimSq
        lIdx = lIdxTbl[idx]
        tIdx = tIdxTbl[idx]
        knt   += Float64(0.50) * (vx[idx] * vx[idx] + vy[idx] * vy[idx])
        ptn1  += Float64(0.50) * ((ux[idx] - ux[lIdx]) * (ux[idx] - ux[lIdx])  +
                                  (uy[idx] - uy[tIdx]) * (uy[idx] - uy[tIdx])) * sp
        ptn2  += Float64(0.5) * k0 * (ux[idx] * ux[idx] + uy[idx] * uy[idx])
    end
    return knt/dimSq, (ptn1+ptn2)/dimSq
end

function computeForce!(dimSq, fermiF, k0, sp, κ, e_ph, ux, uy, fx, fy, rho, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl)
    for idx in 1:dimSq
        lIdx = lIdxTbl[idx]
        rIdx = rIdxTbl[idx]
        tIdx = tIdxTbl[idx]
        bIdx = bIdxTbl[idx]
        ## compute the expectation value
        ld = Float64(0.0)
        rd = Float64(0.0)
        td = Float64(0.0)
        bd = Float64(0.0)
        nn = Float64(0.0)
        for i = 1:dimSq
            ld += rho[idx, i] * rho[lIdx, i] * fermiF[i]
            rd += rho[idx, i] * rho[rIdx, i] * fermiF[i]
            td += rho[idx, i] * rho[tIdx, i] * fermiF[i]
            bd += rho[idx, i] * rho[bIdx, i] * fermiF[i]
        end
        fx[idx] = -k0 * ux[idx] + sp * (ux[lIdx] + ux[rIdx] - Float64(2.0) * ux[idx]) + Float64(2.0) * e_ph  * (rd - ld) - κ * (ux[lIdx] + ux[rIdx] + ux[tIdx] + ux[bIdx])
        fy[idx] = -k0 * uy[idx] + sp * (uy[tIdx] + uy[bIdx] - Float64(2.0) * uy[idx]) + Float64(2.0) * e_ph  * (td - bd) - κ * (uy[lIdx] + uy[rIdx] + uy[tIdx] + uy[bIdx])
    end
    return nothing
end

function evolveSystem(rng,
    dim    :: Int64,
    dimSq  :: Int64,
    T_quench,     
    T_thermalize,             # thermalize
    gm     :: Float64,             # damping factor for Langevin dynamics
    t0     :: Float64,             # hopping parameter
    t2     :: Float64,             # next-nearest neighbor hopping
    e_ph   :: Float64,             # electron-lattice coupling
    k0     :: Float64,             # on-site harmonic potential
    sp     :: Float64,             # spring constant
    κ      :: Float64,             # quadratic spring interaction
    dt     :: Float64,
    nS     :: Int64,
    lIdxTbl :: Vector{Int64},
    rIdxTbl :: Vector{Int64},
    tIdxTbl :: Vector{Int64},
    bIdxTbl :: Vector{Int64},
    tlIdxTbl:: Vector{Int64},
    trIdxTbl:: Vector{Int64},
    blIdxTbl:: Vector{Int64},
    brIdxTbl:: Vector{Int64},
    save_directory,         # New argument for directory              # New argument for quench temperature
    run::Int)                       # New argument for run number
    # Define the run directory
    run_directory = joinpath(save_directory, "T_$T_quench", "run_$run")
    if !isdir(run_directory)
        mkpath(run_directory)
    end
    tp = T_quench
# Initialize lattice variables (same as original)
    ux = 0.1 * randn(rng, Float64, dimSq)
    uy = 0.1 * randn(rng, Float64, dimSq)
    fx = zeros(Float64, dimSq)
    fy = zeros(Float64, dimSq)
    vx = 0.1 * randn(rng, Float64, dimSq)
    vy = 0.1 * randn(rng, Float64, dimSq)
    # Thermalize system (same as original)
    thermalizeSystem!(rng, dim, dimSq, T_thermalize, gm, k0, sp, κ, ux, uy, vx, vy, fx, fy, dt, 10000)

    # Hamiltonian and Fermi factors initialization (same as original)
    Hmn = zeros(Float64, (dimSq, dimSq))
    initializeNNNHmn!(dimSq, t2, Hmn, tlIdxTbl, trIdxTbl, blIdxTbl, brIdxTbl)
    updateHoppingHmn!(dimSq, t0, e_ph, ux, uy, Hmn, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl)
    vals, vecs = eigen(0.5 * (Hmn + Hmn'))
    fermiE = Float64(0.5) * (vals[Int64(dimSq/2)] + vals[Int64(dimSq/2)+1])
    fermiF = 1.0 ./ (exp.((vals .- fermiE) ./ tp) .+ 1.0)
    computeForce!(dimSq, fermiF, k0, sp, κ, e_ph, ux, uy, fx, fy, vecs, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl)

    # Energy arrays initialization
    eksum = Array{Float64}(undef, 0)
    epsum = Array{Float64}(undef, 0)
    eesum = Array{Float64}(undef, 0)
    ek, ep = computeEnergy(dimSq, sp, k0, ux, uy, vx, vy, lIdxTbl, tIdxTbl)
    push!(eksum, ek)
    push!(epsum, ep)
    push!(eesum, vals' * fermiF)

    # Save initial state in separate files
    open(joinpath(run_directory, "evolve_ux.txt"), "w") do io_ux
        writedlm(io_ux, Array(ux)')
    end
    open(joinpath(run_directory, "evolve_uy.txt"), "w") do io_uy
        writedlm(io_uy, Array(uy)')
    end
    open(joinpath(run_directory, "evolve_vx.txt"), "w") do io_vx
        writedlm(io_vx, Array(vx)')
    end
    open(joinpath(run_directory, "evolve_vy.txt"), "w") do io_vy
        writedlm(io_vy, Array(vy)')
    end
    open(joinpath(run_directory, "evolve_fx.txt"), "w") do io_fx
        writedlm(io_fx, Array(fx)')
    end
    open(joinpath(run_directory, "evolve_fy.txt"), "w") do io_fy
        writedlm(io_fy, Array(fy)')
    end
    println("initial energy: $(eksum[end] + epsum[end])")

# Main evolution loop with periodic saving
    open(joinpath(run_directory, "evolve_ux.txt"), "a") do io_ux
        open(joinpath(run_directory, "evolve_uy.txt"), "a") do io_uy
            open(joinpath(run_directory, "evolve_vx.txt"), "a") do io_vx
                open(joinpath(run_directory, "evolve_vy.txt"), "a") do io_vy
                  open(joinpath(run_directory, "evolve_fx.txt"), "a") do io_fx
                    open(joinpath(run_directory, "evolve_fy.txt"), "a") do io_fy
            # Calculate dvx, dvy for initial step
                        dvx = Float64(0.5) * dt * fx
                        dvy = Float64(0.5) * dt * fy

                        for step in 1:nS
                            ux .+= (vx + dvx) * dt
                            uy .+= (vy + dvy) * dt

                            # Force computation with updated Hamiltonian
                            updateHoppingHmn!(dimSq, t0, e_ph, ux, uy, Hmn, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl)
                            vals, vecs = eigen(0.5 * (Hmn + Hmn'))
                            fermiE = Float64(0.5) * (vals[Int64(dimSq/2)] + vals[Int64(dimSq/2)+1])
                            fermiF = 1.0 ./ (exp.((vals .- fermiE) ./ tp) .+ 1.0)
                            computeForce!(dimSq, fermiF, k0, sp, κ, e_ph, ux, uy, fx, fy, vecs, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl)

                            # Update velocities
                            dvxNew = Float64(0.5) * dt * fx
                            dvyNew = Float64(0.5) * dt * fy
                            vx .+= dvx + dvxNew
                            vy .+= dvy + dvyNew
                            dvx = dvxNew
                            dvy = dvyNew

                            # Langevin noise
                            etax = randn(rng, dimSq)
                            etay = randn(rng, dimSq)
                            alp  = Float64(exp(-gm * dt))
                            sig  = Float64(sqrt((1 - alp * alp) * tp))
                            vx .= vx * alp + etax * sig
                            vy .= vy * alp + etay * sig

                            # Save every 1000 steps (matching Code 2)
                            if step % 40 == 0
                                writedlm(io_ux, Array(ux)')
                                writedlm(io_uy, Array(uy)')
                                #writedlm(io_vx, Array(vx)')
                                #writedlm(io_vy, Array(vy)')
                                #writedlm(io_fx, Array(fx)')
                                #writedlm(io_fy, Array(fy)')
                            end
                        end
                    end
                end
            end
        end
    end
end
    return eksum, epsum, eesum
end





function main(T_quench, save_dir)
    dim    = Int64(50)
    dimSq  = dim * dim
    t0     = Float64(2.5)        ## default hopping
    t2     = Float64(0.5)        ## next-nearest neighbor hopping
    sp     = Float64(0.0)       ## spring force: -sp * (ui-uj)
    e_ph   = Float64(4.0)        ## electron-lattice coupling
    k0     = Float64(30.0)       ## onsite parabolic potential
    κ      = Float64(4.0)       ## quadratic coupling
    dt     = Float64(1e-2)
    nS     = Int64(3000)
    #rng    = Xoshiro(1)

    
    # tp : temperature is in terms of kT (8.61733e-5 eV / K)
    gm = Float64(1.0)
    T_thermalize = 0.025
    ## initialize the lookup table for nearest neighbors
    ## lIdxTbl : (i,j-1)
    ## rIdxTbl : (i,j+1)
    ## tIdxTbl : (i+1,j)
    ## bIdxTbl : (i-1,j)
    lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl, tlIdxTbl, trIdxTbl, blIdxTbl, brIdxTbl= initLookupTable(dim, dim*dim)

    for run in 1:20
    
        rng = Xoshiro(2024 + run)
        @time ek, ep, ee = evolveSystem(rng, dim, dimSq, T_quench, T_thermalize, gm, t0, t2, e_ph, k0, sp, κ, dt, nS, lIdxTbl, rIdxTbl, tIdxTbl, bIdxTbl, tlIdxTbl, trIdxTbl, blIdxTbl, brIdxTbl, save_dir, run)
    end
end

Twant = [1e-3]

save_directory = ""



for T_quench in Twant
    main(T_quench, save_directory)
end