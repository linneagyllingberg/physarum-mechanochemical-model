## make_simulation_videos.jl
##
## Produces publication-quality MP4 videos from a phase field simulation solution.
##
## Usage (from within a notebook or script, after a simulation has been solved):
##
##   include("make_simulation_videos.jl")
##
##   # Individual videos
##   make_calcium_video(sol_baseline,   x_dom, y_dom, "baseline")
##   make_thickness_video(sol_network,  x_dom, y_dom, "network")
##
##   # Combined side-by-side video
##   make_combined_video(sol_chemotaxis, x_dom, y_dom, "chemotaxis")
##
## All output goes to ANIMDIR (defined in the notebook as animations/).
## Requires: CairoMakie, ColorSchemes, Printf, CUDA (for Array() conversion)
##
## Field layout in sol.u[i]:
##   [:,:,1] = φ  (phase field)
##   [:,:,2] = c  (calcium, in M   → video shows nM)
##   [:,:,3] = ω  (IP₃ receptor state)
##   [:,:,4] = h  (thickness, in mm)

# ── Colormaps ──────────────────────────────────────────────────────────────────
# Re-use the colormaps built in the notebook (thickness_cmap, nutrient_cmap).
# For calcium we use a separate perceptually-ordered map that highlights waves.
using ColorSchemes
using Statistics
using Printf

"""
    build_calcium_cmap(n=256)

A perceptually-uniform, print-friendly colormap for calcium waves.
Runs from near-black (zero / outside cell) through deep blue → cyan → white-yellow,
emphasizing wave crests without harsh saturation jumps.
Uses the `matter` scheme from ColorSchemes, which is dark-to-light and print-safe.
"""
function build_calcium_cmap(n::Int = 256)
    return [get(ColorSchemes.matter, i / (n - 1)) for i in 0:(n-1)]
end

const calcium_cmap = build_calcium_cmap()

# ── Font sizes for screen/video (larger than PNAS print defaults) ─────────────
const _VID_TITLE_PT  = 44
const _VID_LABEL_PT  = 40
const _VID_TICK_PT   = 40
const _VID_CB_WIDTH  = 34   # colorbar width in px
const _VID_TICK_SIZE = 12   # tick mark length in px (default 6)

# ── Helper: global colorrange from a solution ─────────────────────────────────
"""
    _global_range(sol, field_idx; unit=1.0, frame_indices=nothing, hi_pct=1.0)

Compute (min, max) of `sol.u[i][:,:,field_idx]` over all (or selected) frames.
`unit` scales the raw values (e.g. 1e9 to convert M → nM).
`hi_pct` clips the upper bound to this percentile (e.g. 0.98 ignores top 2%),
which is useful when early frames are much thicker than the bulk dynamics.
"""
function _global_range(sol, field_idx::Int;
                       unit::Real = 1.0,
                       frame_indices = nothing,
                       hi_pct::Float64 = 1.0)
    frames = isnothing(frame_indices) ? eachindex(sol.t) : frame_indices

    if hi_pct >= 1.0
        # Fast path: no percentile clipping needed
        lo = Inf;  hi = -Inf
        for i in frames
            v = Array(sol.u[i][:, :, field_idx]) .* unit
            lo = min(lo, minimum(v))
            hi = max(hi, maximum(v))
        end
        return (Float64(lo), Float64(hi))
    else
        # Sample ~60 frames evenly to estimate percentile without loading everything
        all_frames = collect(frames)
        step = max(1, length(all_frames) ÷ 60)
        sample_frames = all_frames[1:step:end]
        all_vals = Float64[]
        for i in sample_frames
            append!(all_vals, vec(Array(sol.u[i][:, :, field_idx])) .* unit)
        end
        lo = minimum(all_vals)
        hi = quantile(all_vals, hi_pct)
        println("  ($(round(100*(1-hi_pct)))% of values above colorbar max are clipped)")
        return (Float64(lo), Float64(hi))
    end
end

# ── Single-field video ─────────────────────────────────────────────────────────
"""
    _make_field_video(sol, x_dom, y_dom, field_idx, outpath;
                      colormap, colorrange, colorbar_label,
                      field_title, unit, fps, figsize)

Core rendering loop used by `make_calcium_video` and `make_thickness_video`.
Records every frame in `sol.t` to an MP4 (or GIF if `outpath` ends in .gif).
"""
function _make_field_video(sol, x_dom, y_dom,
                           field_idx::Int,
                           outpath::AbstractString;
                           colormap,
                           colorrange::Tuple,
                           colorbar_label::AbstractString,
                           field_title::AbstractString,
                           unit::Real          = 1.0,
                           fps::Int            = 20,
                           figsize::Tuple      = (1100, 900),
                           frame_indices       = nothing)

    frames = isnothing(frame_indices) ? eachindex(sol.t) : frame_indices

    # Use 1:1 pixel mapping and override any global PNAS theme so video font
    # sizes are exactly what we specify (set_pnas_defaults! sets a global theme
    # with 7.5pt fonts that can otherwise win over per-element attributes).
    CairoMakie.activate!(; px_per_unit = 1)

    vid_theme = Theme(
        Axis     = (xticklabelsize = _VID_TICK_PT, yticklabelsize = _VID_TICK_PT,
                    xlabelsize = _VID_LABEL_PT, ylabelsize = _VID_LABEL_PT,
                    titlesize  = _VID_TITLE_PT, titlefont = :regular,
                    xticksize  = _VID_TICK_SIZE, yticksize = _VID_TICK_SIZE),
        Colorbar = (ticklabelsize = _VID_TICK_PT, labelsize = _VID_LABEL_PT,
                    ticksize = _VID_TICK_SIZE),
    )

    _xtick_vals = [x_dom[1], (x_dom[1]+x_dom[end])/2, x_dom[end]]
    _ytick_vals = [y_dom[1], y_dom[end]]
    _xticks = (_xtick_vals, [@sprintf("%.3g", v) for v in _xtick_vals])
    _yticks = (_ytick_vals, [@sprintf("%.3g", v) for v in _ytick_vals])

    with_theme(vid_theme) do
        fig = Figure(size = figsize, backgroundcolor = :white)

        ax = Axis(fig[1, 1];
                  aspect       = DataAspect(),
                  xlabel       = "x (mm)",
                  ylabel       = "y (mm)",
                  xticks       = _xticks,
                  yticks       = _yticks,
                  xgridvisible = false,
                  ygridvisible = false)

        # Plot first frame to anchor the colorbar
        first_frame = Array(sol.u[first(frames)][:, :, field_idx])' .* unit
        hm = CairoMakie.heatmap!(ax, x_dom, y_dom, first_frame;
                                  colormap   = colormap,
                                  colorrange = colorrange,
                                  rasterize  = 4)

        Colorbar(fig[1, 2], hm;
                 label = colorbar_label,
                 width = _VID_CB_WIDTH)

        colsize!(fig.layout, 1, Aspect(1, 1.0))
        colgap!(fig.layout, 1, 50)   # axis → colorbar

        # Record
        CairoMakie.record(fig, outpath, frames; framerate = fps) do idx
            empty!(ax)
            frame = Array(sol.u[idx][:, :, field_idx])' .* unit
            CairoMakie.heatmap!(ax, x_dom, y_dom, frame;
                                colormap   = colormap,
                                colorrange = colorrange,
                                rasterize  = 4)
            ax.title = @sprintf("%s    t = %.1f s", field_title, sol.t[idx])
        end
    end

    println("✓  Saved: $outpath")
    return outpath
end

# ── Public API: calcium video ──────────────────────────────────────────────────
"""
    make_calcium_video(sol, x_dom, y_dom, name;
                       fps=20, dir=ANIMDIR, colorrange=nothing,
                       frame_indices=nothing, ext=".mp4")

Render a video of the calcium concentration field (field 2 in sol).
Values are automatically converted from M to nM.

# Arguments
- `sol`           : DifferentialEquations solution object
- `x_dom`, `y_dom`: spatial coordinate ranges used in the simulation
- `name`          : base filename (e.g. "baseline" → "baseline_calcium.mp4")
- `fps`           : frames per second (default 20)
- `dir`           : output directory (default ANIMDIR from notebook)
- `colorrange`    : fixed (lo, hi) in nM; computed automatically if `nothing`
- `frame_indices` : subset of frame indices to render (default = all)
- `ext`           : ".mp4" (default) or ".gif"
"""
function make_calcium_video(sol, x_dom, y_dom, name::AbstractString;
                            fps::Int            = 20,
                            dir::AbstractString = ANIMDIR,
                            colorrange          = nothing,
                            figsize::Tuple      = (1100, 900),
                            frame_indices       = nothing,
                            ext::AbstractString = ".mp4")

    CUDA.allowscalar(true)
    unit = 1e9  # M → nM

    cr = isnothing(colorrange) ?
         _global_range(sol, 2; unit = unit, frame_indices = frame_indices) :
         colorrange

    println("Calcium colorrange: $(round(cr[1], sigdigits=3)) – $(round(cr[2], sigdigits=3)) nM")

    outpath = joinpath(dir, name * "_calcium" * ext)
    _make_field_video(sol, x_dom, y_dom, 2, outpath;
                      colormap       = calcium_cmap,
                      colorrange     = cr,
                      colorbar_label = "Ca²⁺ (nM)",
                      field_title    = "Calcium",
                      unit           = unit,
                      fps            = fps,
                      figsize        = figsize,
                      frame_indices  = frame_indices)
end

# ── Public API: thickness video ────────────────────────────────────────────────
"""
    make_thickness_video(sol, x_dom, y_dom, name;
                         fps=20, dir=ANIMDIR, colorrange=nothing,
                         frame_indices=nothing, ext=".mp4")

Render a video of the cell thickness field (field 4 in sol), in mm.

# Arguments same as `make_calcium_video` (colorrange in mm).
"""
function make_thickness_video(sol, x_dom, y_dom, name::AbstractString;
                              fps::Int            = 20,
                              dir::AbstractString = ANIMDIR,
                              colorrange          = (0.0, 0.09),
                              figsize::Tuple      = (1100, 900),
                              frame_indices       = nothing,
                              ext::AbstractString = ".mp4")

    CUDA.allowscalar(true)

    cr = isnothing(colorrange) ?
         _global_range(sol, 4; unit = 1.0, frame_indices = frame_indices, hi_pct = 0.98) :
         colorrange

    println("Thickness colorrange: $(round(cr[1], sigdigits=3)) – $(round(cr[2], sigdigits=3)) mm")

    outpath = joinpath(dir, name * "_thickness" * ext)
    _make_field_video(sol, x_dom, y_dom, 4, outpath;
                      colormap       = thickness_cmap,
                      colorrange     = cr,
                      colorbar_label = "h (mm)",
                      field_title    = "Thickness",
                      unit           = 1.0,
                      fps            = fps,
                      figsize        = figsize,
                      frame_indices  = frame_indices)
end

# ── Public API: combined side-by-side video ────────────────────────────────────
"""
    make_combined_video(sol, x_dom, y_dom, name;
                        fps=20, dir=ANIMDIR,
                        ca_colorrange=nothing, h_colorrange=nothing,
                        nutrient_field=nothing, nutrient_colorrange=nothing,
                        figsize=nothing, frame_indices=nothing, ext=".mp4")

Render a side-by-side video: [nutrient (optional)] | calcium | thickness.
Pass `nutrient_field = Q_baseline` to add a static Q panel on the left.

# Arguments
- `ca_colorrange`      : fixed (lo, hi) for calcium in nM
- `h_colorrange`       : fixed (lo, hi) for thickness in mm
- `nutrient_field`     : optional 2D array (or CuArray) for a static Q panel
- `nutrient_colorrange`: fixed (lo, hi) for Q; auto-computed if nothing
- `figsize`            : figure size in px; defaults to 1000px per panel × 900px tall
All other arguments same as `make_calcium_video`.
"""
function make_combined_video(sol, x_dom, y_dom, name::AbstractString;
                             fps::Int                        = 20,
                             dir::AbstractString             = ANIMDIR,
                             ca_colorrange                   = nothing,
                             h_colorrange                    = (0.0, 0.09),
                             nutrient_field                  = nothing,
                             nutrient_colorrange             = (0.299, 0.341),
                             figsize::Union{Nothing,Tuple}   = nothing,
                             frame_indices                   = nothing,
                             ext::AbstractString             = ".mp4")

    CUDA.allowscalar(true)
    unit_ca = 1e9  # M → nM

    frames = isnothing(frame_indices) ? eachindex(sol.t) : frame_indices

    # Compute global colorranges once
    cr_ca = isnothing(ca_colorrange) ?
            _global_range(sol, 2; unit = unit_ca, frame_indices = frame_indices) :
            ca_colorrange
    cr_h  = isnothing(h_colorrange) ?
            _global_range(sol, 4; unit = 1.0, frame_indices = frame_indices, hi_pct = 0.98) :
            h_colorrange

    println("Combined video colorranges:")
    println("  Ca²⁺  : $(round(cr_ca[1], sigdigits=3)) – $(round(cr_ca[2], sigdigits=3)) nM")
    println("  h     : $(round(cr_h[1],  sigdigits=3)) – $(round(cr_h[2],  sigdigits=3)) mm")

    # Layout: nutrient (optional, cols 1-2) | calcium (cols c_ca, c_ca+1) | thickness (cols c_h, c_h+1)
    has_Q   = !isnothing(nutrient_field)
    c_ca    = has_Q ? 3 : 1   # first column for calcium panel
    c_h     = has_Q ? 5 : 3   # first column for thickness panel
    n_cols  = has_Q ? 6 : 4

    actual_figsize = isnothing(figsize) ? ((has_Q ? 3 : 2) * 1000, 900) : figsize

    outpath = joinpath(dir, name * "_combined" * ext)

    CairoMakie.activate!(; px_per_unit = 1)

    vid_theme = Theme(
        Axis     = (xticklabelsize = _VID_TICK_PT, yticklabelsize = _VID_TICK_PT,
                    xlabelsize = _VID_LABEL_PT, ylabelsize = _VID_LABEL_PT,
                    titlesize  = _VID_TITLE_PT, titlefont = :regular,
                    xticksize  = _VID_TICK_SIZE, yticksize = _VID_TICK_SIZE),
        Colorbar = (ticklabelsize = _VID_TICK_PT, labelsize = _VID_LABEL_PT,
                    ticksize = _VID_TICK_SIZE),
    )

    _xtick_vals = [x_dom[1], (x_dom[1]+x_dom[end])/2, x_dom[end]]
    _ytick_vals = [y_dom[1], y_dom[end]]
    _xticks = (_xtick_vals, [@sprintf("%.3g", v) for v in _xtick_vals])
    _yticks = (_ytick_vals, [@sprintf("%.3g", v) for v in _ytick_vals])

    with_theme(vid_theme) do
        fig = Figure(size = actual_figsize, backgroundcolor = :white)

        # ── Optional static nutrient panel (leftmost) ─────────────────────────
        if has_Q
            Q_data = Array(nutrient_field)'
            _lo_Q = Float64(minimum(Q_data))
            _hi_Q = Float64(maximum(Q_data))
            if _lo_Q ≈ _hi_Q  # uniform field — pad ±5% so Makie can interpolate
                _pad = max(abs(_lo_Q) * 0.05, 1e-6)
                _lo_Q -= _pad;  _hi_Q += _pad
                println("  Q field is uniform ($(round(_lo_Q + _pad, sigdigits=4))) — padding colorrange ±5%")
            end
            cr_Q = isnothing(nutrient_colorrange) ? (_lo_Q, _hi_Q) : nutrient_colorrange
            println("  Q     : $(round(cr_Q[1], sigdigits=3)) – $(round(cr_Q[2], sigdigits=3))")
            ax_Q = Axis(fig[1, 1];
                        title        = "Nutrient field",
                        aspect       = DataAspect(),
                        xlabel       = "x (mm)",
                        ylabel       = "y (mm)",
                        xticks       = _xticks,
                        yticks       = _yticks,
                        xgridvisible = false,
                        ygridvisible = false)
            hm_Q = CairoMakie.heatmap!(ax_Q, x_dom, y_dom, Q_data;
                                        colormap   = nutrient_cmap,
                                        colorrange = cr_Q,
                                        rasterize  = 4)
            Colorbar(fig[1, 2], hm_Q;
                     label = "Q",
                     ticks = ([0.30, 0.32, 0.34], ["0.30", "0.32", "0.34"]),
                     width = _VID_CB_WIDTH)
            colsize!(fig.layout, 1, Aspect(1, 1.0))
        end

        # ── Calcium panel ─────────────────────────────────────────────────────
        ax_ca = Axis(fig[1, c_ca];
                     title        = "Calcium",
                     aspect       = DataAspect(),
                     xlabel       = "x (mm)",
                     ylabel       = has_Q ? "" : "y (mm)",
                     xticks       = _xticks,
                     yticks       = _yticks,
                     xgridvisible = false,
                     ygridvisible = false)
        has_Q && hideydecorations!(ax_ca, ticks = false)

        first_ca = Array(sol.u[first(frames)][:, :, 2])' .* unit_ca
        hm_ca = CairoMakie.heatmap!(ax_ca, x_dom, y_dom, first_ca;
                                     colormap   = calcium_cmap,
                                     colorrange = cr_ca,
                                     rasterize  = 4)
        Colorbar(fig[1, c_ca + 1], hm_ca;
                 label = "Ca²⁺ (nM)",
                 width = _VID_CB_WIDTH)

        # ── Thickness panel ───────────────────────────────────────────────────
        ax_h = Axis(fig[1, c_h];
                    title        = "Thickness",
                    aspect       = DataAspect(),
                    xlabel       = "x (mm)",
                    xticks       = _xticks,
                    yticks       = _yticks,
                    xgridvisible = false,
                    ygridvisible = false)
        hideydecorations!(ax_h, ticks = false)

        first_h = Array(sol.u[first(frames)][:, :, 4])'
        hm_h = CairoMakie.heatmap!(ax_h, x_dom, y_dom, first_h;
                                    colormap   = thickness_cmap,
                                    colorrange = cr_h,
                                    rasterize  = 4)
        Colorbar(fig[1, c_h + 1], hm_h;
                 label = "h (mm)",
                 ticks = ([0.00, 0.03, 0.06, 0.09], ["0.00", "0.03", "0.06", "0.09"]),
                 width = _VID_CB_WIDTH)

        # ── Shared time-stamp label at top ────────────────────────────────────
        time_label = Label(fig[0, 1:n_cols], @sprintf("t = %.1f s", sol.t[first(frames)]);
                           fontsize  = _VID_TITLE_PT + 2,
                           font      = :bold,
                           tellwidth = false)

        colsize!(fig.layout, c_ca, Aspect(1, 1.0))
        colsize!(fig.layout, c_h,  Aspect(1, 1.0))
        # Gaps: axis → its own colorbar (tight)
        has_Q && colgap!(fig.layout, 1, 50)        # ax_Q  → cb_Q
        colgap!(fig.layout, c_ca, 50)              # ax_ca → cb_ca
        colgap!(fig.layout, c_h,  50)              # ax_h  → cb_h
        # Gaps between panel groups (wider separation)
        has_Q && colgap!(fig.layout, 2, 100)       # cb_Q  → ax_ca
        colgap!(fig.layout, c_ca + 1, 100)         # cb_ca → ax_h

        # Record (nutrient panel is static — only ca and h update each frame)
        CairoMakie.record(fig, outpath, frames; framerate = fps) do idx
            empty!(ax_ca);  empty!(ax_h)

            c_frame = Array(sol.u[idx][:, :, 2])' .* unit_ca
            h_frame = Array(sol.u[idx][:, :, 4])'

            CairoMakie.heatmap!(ax_ca, x_dom, y_dom, c_frame;
                                colormap   = calcium_cmap,
                                colorrange = cr_ca,
                                rasterize  = 4)
            CairoMakie.heatmap!(ax_h, x_dom, y_dom, h_frame;
                                colormap   = thickness_cmap,
                                colorrange = cr_h,
                                rasterize  = 4)

            time_label.text[] = @sprintf("t = %.1f s", sol.t[idx])
        end
    end

    println("✓  Saved: $outpath")
    return outpath
end

println("make_simulation_videos.jl loaded.")
println("  make_calcium_video(sol, x_dom, y_dom, \"name\")")
println("  make_thickness_video(sol, x_dom, y_dom, \"name\")")
println("  make_combined_video(sol, x_dom, y_dom, \"name\")")
