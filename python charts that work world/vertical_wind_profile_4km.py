#!/usr/bin/env python3
"""
Time–height wind barbs at a point, using standard pressure levels.

- X axis: time
- Y axis 1 (left / PRIMARY): pressure (hPa), decreasing with height
- Y axis 2 (right / SECONDARY): height AGL (m and ft)
- Wind barbs are plotted at standard pressure levels so they line up
  with the primary pressure ticks. The secondary axis shows the
  corresponding heights with fixed "nice" height ticks.
"""

###############################################################################
# Imports (clean, ordered)
###############################################################################
import glob
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import wrf
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from metpy.units import units
from netCDF4 import Dataset
from PIL import Image
from scipy.ndimage import gaussian_filter
from wrf import ALL_TIMES, getvar, ll_to_xy, to_np

###############################################################################
# Warning suppression
###############################################################################
warnings.filterwarnings("ignore")

###############################################################################
# Canonical helper function block (v9 – contiguous, order-locked)
###############################################################################


def add_feature(
    ax, category, scale, facecolor, edgecolor, linewidth, name, zorder=None, alpha=None
):
    feature = cfeature.NaturalEarthFeature(
        category=category,
        scale=scale,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        name=name,
        zorder=zorder,
        alpha=alpha,
    )
    ax.add_feature(feature)


def parse_valid_time_from_wrf_name(path: str) -> datetime:
    base = os.path.basename(path)

    match = re.search(
        r"wrfout_.*?_(\d{4}-\d{2}-\d{2})_(\d{2}[:_]\d{2}[:_]\d{2})",
        base,
    )
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace("_", ":")
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H:%M:%S")
        except Exception:
            pass

    try:
        year = base[11:15]
        month = base[16:18]
        day = base[19:21]
        hour = base[22:24]
        minute = base[25:27]
        second = base[28:30]
        return datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(second)
        )
    except Exception:
        return datetime.utcfromtimestamp(os.path.getmtime(path))


def get_valid_time(ncfile: Dataset, ncfile_path: str, time_index: int) -> datetime:
    try:
        valid = wrf.extract_times(ncfile, timeidx=time_index)

        if isinstance(valid, np.ndarray):
            valid = valid.item()

        if isinstance(valid, np.datetime64):
            valid = valid.astype("datetime64[ms]").tolist()

        if isinstance(valid, datetime):
            return valid
    except Exception:
        pass

    return parse_valid_time_from_wrf_name(ncfile_path)


def compute_grid_and_spacing(lats, lons):
    lats_np = to_np(lats)
    lons_np = to_np(lons)

    dx, dy = mpcalc.lat_lon_grid_deltas(lons_np, lats_np)

    dx_km = dx.to(units.kilometer)
    dy_km = dy.to(units.kilometer)

    dx_km_rounded = np.round(dx_km.magnitude, 2)
    dy_km_rounded = np.round(dy_km.magnitude, 2)

    avg_dx_km = round(np.mean(dx_km_rounded), 2)
    avg_dy_km = round(np.mean(dy_km_rounded), 2)

    if avg_dx_km >= 9 or avg_dy_km >= 9:
        extent_adjustment = 0.50
        label_adjustment = 0.35
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        extent_adjustment = 0.25
        label_adjustment = 0.20
    else:
        extent_adjustment = 0.15
        label_adjustment = 0.15

    return lats_np, lons_np, avg_dx_km, avg_dy_km, extent_adjustment, label_adjustment


def add_latlon_gridlines(ax):
    gl = ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=True,
        linestyle="--",
        color="black",
        alpha=0.5,
    )

    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_right = False
    gl.ylabels_left = True

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.x_inline = False
    gl.top_labels = False
    gl.right_labels = False

    return gl


def plot_cities(ax, lons_np, lats_np, avg_dx_km, avg_dy_km):
    plot_extent = [
        lons_np.min(),
        lons_np.max(),
        lats_np.min(),
        lats_np.max(),
    ]

    cities_within_extent = cities.cx[
        plot_extent[0] : plot_extent[1],
        plot_extent[2] : plot_extent[3],
    ]

    sorted_cities = cities_within_extent.sort_values(
        by="POP_MAX", ascending=False
    ).head(150)

    if sorted_cities.empty:
        return

    if avg_dx_km >= 9 or avg_dy_km >= 9:
        min_distance = 1.0
    elif 3 < avg_dx_km < 9 or 3 < avg_dy_km < 9:
        min_distance = 0.75
    else:
        min_distance = 0.40

    gdf_sorted = gpd.GeoDataFrame(
        sorted_cities,
        geometry=gpd.points_from_xy(
            sorted_cities.LONGITUDE,
            sorted_cities.LATITUDE,
        ),
    )

    selected_rows = []
    selected_geoms = []

    for row in gdf_sorted.itertuples():
        geom = row.geometry
        if not selected_geoms:
            selected_geoms.append(geom)
            selected_rows.append(row)
        else:
            distances = [g.distance(geom) for g in selected_geoms]
            if min(distances) >= min_distance:
                selected_geoms.append(geom)
                selected_rows.append(row)

    if not selected_rows:
        return

    filtered_cities = gpd.GeoDataFrame(selected_rows).set_geometry("geometry")

    for city_name, loc in zip(filtered_cities.NAME, filtered_cities.geometry):
        ax.plot(
            loc.x,
            loc.y,
            marker="o",
            markersize=6,
            color="r",
            transform=crs.PlateCarree(),
            clip_on=True,
        )
        ax.text(
            loc.x,
            loc.y,
            city_name,
            transform=crs.PlateCarree(),
            ha="center",
            va="bottom",
            fontsize=11,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.08",
                facecolor="white",
                alpha=0.4,
            ),
            clip_on=True,
        )


def handle_domain_continuity_and_polar_mask(lats_np, lons_np, *fields):
    """
    Detect and correct dateline continuity and polar masking for WRF domains.

    Ensures proper handling of longitude wrapping across the 180° meridian
    and masking for domains including polar caps.

    This function is field-agnostic: pass any number of fields (or none).
    All provided fields are reordered/masked consistently with lats/lons.
    """
    lats_min = np.nanmin(lats_np)
    lats_max = np.nanmax(lats_np)
    lons_min = np.nanmin(lons_np)
    lons_max = np.nanmax(lons_np)

    lon_span = lons_max - lons_min
    dateline_crossing = lon_span > 180.0
    polar_domain = (abs(lats_min) > 70.0) or (abs(lats_max) > 70.0)

    fields_out = list(fields)

    if dateline_crossing:
        lons_wrapped = np.where(lons_np < 0.0, lons_np + 360.0, lons_np)
        sort_idx = np.argsort(lons_wrapped[0, :])

        lons_np = lons_wrapped[..., sort_idx]
        lats_np = lats_np[..., sort_idx]
        fields_out = [(f[..., sort_idx] if f is not None else None) for f in fields_out]

    if polar_domain and dateline_crossing:
        polar_cap_lat = 88.0
        polar_mask = (lats_np >= polar_cap_lat) | (lats_np <= -polar_cap_lat)

        fields_out = [
            (np.ma.masked_where(polar_mask, f) if f is not None else None)
            for f in fields_out
        ]

    return (lats_np, lons_np, *fields_out)


###############################################################################
# Natural Earth features (v9 canonical – verbatim, order-locked)
###############################################################################
# List of Natural Earth features to add (keep commented-out options intact)
features = [
    ("physical", "10m", cfeature.COLORS["land"], "black", 0.50, "minor_islands"),
    ("physical", "10m", "none", "black", 0.50, "coastline"),
    ("physical", "10m", cfeature.COLORS["water"], None, None, "ocean_scale_rank", -1),
    ("physical", "10m", cfeature.COLORS["water"], "lightgrey", 0.75, "lakes", 0),
    ("cultural", "10m", "none", "grey", 1.00, "admin_1_states_provinces", 2),
    ("cultural", "10m", "none", "black", 1.50, "admin_0_countries", 2),
    # ("cultural", "10m", "none", "black", 0.60, "admin_2_counties", 2, 0.6),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_lake_centerlines"),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_north_america", None), 0.75),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_australia", None), 0.75),
    # ("physical", "10m", "none", cfeature.COLORS["water"], None, "rivers_europe", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None,
    #  "lakes_north_america", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None,
    #  "lakes_australia", None), 0.75),
    # ("physical", "10m", cfeature.COLORS["water"], cfeature.COLORS["water"], None,
    #  "lakes_europe", None), 0.75),
]

###############################################################################
# Cities (module scope)
###############################################################################
cities = gpd.read_file(
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_populated_places.zip"
)

# =============================================================================
# IMPORTS AND GLOBAL CONFIGURATION
# =============================================================================

# (Retained verbatim as provided; imports are above in v9 order.)


# =============================================================================
# PER-FILE PROCESSING FUNCTION (RUNS IN EACH WORKER PROCESS)
# =============================================================================
def process_frame(args):
    """
    Worker function that interpolates U and V winds to standard pressure levels
    (limited to ≤ 4 km AGL subset as requested) for a single WRF output file.
    Designed to be called from a ProcessPoolExecutor worker process.
    """
    (
        wrf_file,
        time_index,
        domain,
        x_idx,
        y_idx,
        p_levels,
        mask_4km,
    ) = args

    print(f"Processing {wrf_file} (time_index={time_index})")
    with Dataset(wrf_file) as nc:
        # Get valid time as the timestamp authority (v9 canonical)
        dt = get_valid_time(nc, wrf_file, time_index)

        p_da = getvar(nc, "pres", timeidx=time_index, units="hPa")
        u_da = getvar(nc, "ua", timeidx=time_index, units="kt")
        v_da = getvar(nc, "va", timeidx=time_index, units="kt")

        p_prof = to_np(p_da[:, y_idx, x_idx]).astype(float)
        u_prof = to_np(u_da[:, y_idx, x_idx]).astype(float)
        v_prof = to_np(v_da[:, y_idx, x_idx]).astype(float)

        # Sort by pressure ascending for interpolation
        srt = np.argsort(p_prof)
        p_s = p_prof[srt]
        u_s = u_prof[srt]
        v_s = v_prof[srt]

        # Interpolate U/V to the full standard p_levels
        u_std_full = np.interp(p_levels, p_s, u_s, left=np.nan, right=np.nan)
        v_std_full = np.interp(p_levels, p_s, v_s, left=np.nan, right=np.nan)

        # Then restrict to the ≤ 4 km subset
        if mask_4km is not None and np.any(mask_4km):
            u_std = u_std_full[mask_4km]
            v_std = v_std_full[mask_4km]
        else:
            u_std = u_std_full
            v_std = v_std_full

    return dt, u_std, v_std


###############################################################################
# Frame Discovery (v9 canonical)
###############################################################################
def discover_frames(ncfile_paths):
    frames = []

    for path in ncfile_paths:
        with Dataset(path) as nc:
            if "Time" in nc.dimensions:
                n_times = len(nc.dimensions["Time"])
            elif "Times" in nc.variables:
                n_times = nc.variables["Times"].shape[0]
            else:
                n_times = 1

        for t in range(n_times):
            frames.append((path, t))

    return frames


# =============================================================================
# MAIN PROGRAM / COMMAND-LINE INTERFACE
# =============================================================================
def main():
    # -------------------------------------------------------------------
    # Command line args
    # -------------------------------------------------------------------
    if len(sys.argv) != 5:
        print("Usage: script.py <path_to_wrf_files> <domain> <latitude> <longitude>")
        sys.exit(1)

    path_wrf = sys.argv[1]
    domain = sys.argv[2]
    lat = float(sys.argv[3])
    lon = float(sys.argv[4])

    output_dir = "Vertical_Wind_Profiles_4km"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Find WRF output files
    # -------------------------------------------------------------------
    wrf_pattern = os.path.join(path_wrf, f"wrfout_{domain}_*")
    wrf_files = sorted(glob.glob(wrf_pattern))

    if not wrf_files:
        print(f"No WRF files found in {path_wrf} matching wrfout_{domain}_*")
        sys.exit(1)

    print(f"Found {len(wrf_files)} WRF files.")

    # -------------------------------------------------------------------
    # STEP 1: Define standard pressure levels and map them to height
    #         using the first file (reference sounding)
    # -------------------------------------------------------------------
    first_file = wrf_files[0]
    print(f"Using {first_file} to derive standard pressure→height mapping.")

    with Dataset(first_file) as nc:
        x_idx, y_idx = ll_to_xy(nc, lat, lon, meta=False)

        # Pressure and height AGL at the chosen grid point
        p_da = getvar(nc, "pres", timeidx=0, units="hPa")
        z_da = getvar(nc, "height_agl", timeidx=0, units="m")

        p_prof0 = to_np(p_da[:, y_idx, x_idx]).astype(float)
        z_prof0 = to_np(z_da[:, y_idx, x_idx]).astype(float)

        # Standard pressure levels of interest
        p_levels_all = np.array(
            [
                1013,
                1000,
                950,
                900,
                850,
                800,
                750,
                700,
                650,
                600,
                550,
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150,
                100,
            ],
            dtype=float,
        )

        # Keep only those within the model's pressure range
        pmin = min(p_prof0.min(), p_prof0.max())
        pmax = max(p_prof0.min(), p_prof0.max())
        mask = (p_levels_all >= pmin) & (p_levels_all <= pmax)
        p_levels = p_levels_all[mask]

        if p_levels.size == 0:
            raise RuntimeError(
                "None of the requested pressure levels are within the model range."
            )

        # Interpolate height at the standard pressure levels
        sort_p = np.argsort(p_prof0)
        p_sorted = p_prof0[sort_p]
        z_sorted = z_prof0[sort_p]

        z_levels = np.interp(p_levels, p_sorted, z_sorted)

        # Sort by height ascending, carry pressure along
        sort_z = np.argsort(z_levels)
        z_levels = z_levels[sort_z]
        p_levels = p_levels[sort_z]

    # Limit to heights ≤ 4000 m
    MAX_HEIGHT = 4000.0  # meters
    mask_4km = z_levels <= MAX_HEIGHT

    if not np.any(mask_4km):
        print("Warning: no levels at or below 4000 m; using full vertical range.")
        p_levels_plot = p_levels
        z_levels_plot = z_levels
    else:
        p_levels_plot = p_levels[mask_4km]
        z_levels_plot = z_levels[mask_4km]

    print("Standard pressure → height mapping used for plotting (≤ 4 km):")
    for p, z in zip(p_levels_plot, z_levels_plot):
        print(f"  {p:6.1f} hPa  ~  {z:7.1f} m")

    # =============================================================================
    # MULTIPROCESSING ANALYSIS AND SETUP
    # =============================================================================
    # Multiprocessing analysis:
    # - Interpolating U and V onto standard pressure levels is CPU-bound for
    #   each output time and independent between files.
    # - ProcessPoolExecutor with max_workers = 4 parallelizes this per-file work.
    # - Figure assembly and barb plotting are relatively lightweight and kept
    #   single-process.

    # -------------------------------------------------------------------
    # STEP 2: For each time, interpolate U and V onto these pressure levels
    # -------------------------------------------------------------------
    times = []
    u_list = []
    v_list = []

    frames = discover_frames(wrf_files)
    if not frames:
        print("No timesteps found in provided WRF files.")
        sys.exit(0)

    tasks = [
        (wrf_file, time_index, domain, x_idx, y_idx, p_levels, mask_4km)
        for (wrf_file, time_index) in frames
    ]

    max_workers = min(4, len(tasks)) if tasks else 1

    # Use a ProcessPoolExecutor with a fixed pool size of 4 worker processes
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for dt, u_std, v_std in executor.map(process_frame, tasks):
            times.append(dt)
            u_list.append(u_std)
            v_list.append(v_std)

    times = np.array(times)
    u_levels = np.array(u_list)
    v_levels = np.array(v_list)

    # Sort by time to ensure monotonic time axis
    order = np.argsort(times)
    times = times[order]
    u_levels = u_levels[order]
    v_levels = v_levels[order]

    nt, nlev_plot = u_levels.shape
    print(f"Interpolated winds (≤ 4 km): ntimes={nt}, nlevels={nlev_plot}")

    # -------------------------------------------------------------------
    # STEP 3: Build grids for barbs (time × pressure)
    # -------------------------------------------------------------------
    time_nums = mdates.date2num(times)

    MAX_TIME_COLUMNS = 40
    time_stride = max(1, nt // MAX_TIME_COLUMNS)

    # Base stride selection
    t_idx = np.arange(0, nt, time_stride)

    # Force-keep any exact 00:00 times (00Z) that exist
    keep_midnight = np.array([(t.hour == 0 and t.minute == 0) for t in times])
    t_idx = np.unique(np.concatenate([t_idx, np.where(keep_midnight)[0]]))

    time_sel = time_nums[t_idx]

    # 2D grids: y = p_levels_plot, x = selected times
    T, P = np.meshgrid(time_sel, p_levels_plot)
    U_plot = u_levels[t_idx, :].T
    V_plot = v_levels[t_idx, :].T

    # -------------------------------------------------------------------
    # STEP 4: Plot
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=150)

    # Wind barbs at each pressure level
    ax.barbs(T, P, U_plot, V_plot, length=6, clip_on=False)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(f"Time–Pressure Wind Barbs at {lat:.3f}, {lon:.3f} (≤ 4 km AGL)")

    # Pressure axis: high pressure at bottom, low pressure at top
    pad = 10.0
    ax.set_ylim(p_levels_plot.max() + pad, p_levels_plot.min() - pad)

    ax.grid(True, linestyle="--", alpha=0.3)

    # X-axis ticks & labels
    tick_times = mdates.num2date(time_sel)

    tick_labels = []
    for dt in tick_times:
        if dt.hour == 0 and dt.minute == 0:
            tick_labels.append(
                dt.strftime("%m-%d %H:%M")
            )  # date (no year) + time at 00Z
        else:
            tick_labels.append(dt.strftime("%H:%M"))  # time only otherwise

    ax.set_xticks(time_sel)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center")

    # Primary y-axis ticks at the plotted pressure levels
    ax.set_yticks(p_levels_plot)
    ax.set_yticklabels([str(int(p)) for p in p_levels_plot])

    # -------------------------------------------------------------------
    # Secondary y-axis (right): height AGL (m and ft)
    #   Primary data coord = pressure (hPa)
    #   Secondary coord = height (m)
    # -------------------------------------------------------------------

    # For mapping p -> z: need p ascending
    p_for_p_to_z = p_levels_plot[::-1]
    z_for_p_to_z = z_levels_plot[::-1]

    # For mapping z -> p: z_levels already ascending
    z_for_z_to_p = z_levels_plot
    p_for_z_to_p = p_levels_plot

    def p_to_z(p):
        """Map pressure (hPa) -> height (m) for secondary axis."""
        p = np.asarray(p)
        return np.interp(p, p_for_p_to_z, z_for_p_to_z)

    def z_to_p(z):
        """Map height (m) -> pressure (hPa) for secondary axis."""
        z = np.asarray(z)
        return np.interp(z, z_for_z_to_p, p_for_z_to_p)

    secax = ax.secondary_yaxis("right", functions=(p_to_z, z_to_p))
    secax.set_ylabel("Height AGL")

    # Desired height ticks in meters (up to 4 km)
    desired_heights = np.array(
        [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000], dtype=float
    )

    # Limit to heights that are within the plotted range
    zmin_map = z_levels_plot.min()
    zmax_map = z_levels_plot.max()
    mask_h = (desired_heights >= zmin_map) & (desired_heights <= zmax_map)
    z_ticks = desired_heights[mask_h]

    if z_ticks.size == 0:
        z_ticks = np.array([zmin_map, min(zmax_map, MAX_HEIGHT)])

    secax.set_yticks(z_ticks)

    height_labels = []
    for z in z_ticks:
        m_val = int(round(z))
        ft_val = int(round(z * 3.28084))
        height_labels.append(f"{m_val} m\n({ft_val} ft)")
    secax.set_yticklabels(height_labels)

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.25, top=0.90)

    # Original output name preserved here as a comment:
    # out_file = os.path.join(
    #     output_dir, "time_height_wind_barbs_pressure_primary_4km.png"
    # )

    fname_time = times[0].strftime("%Y%m%d%H%M%S") if len(times) else "00000000000000"
    out_file = os.path.join(
        output_dir, f"wrf_{domain}_time_pressure_wind_barbs_4km_{fname_time}.png"
    )

    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {out_file}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    ###############################################################################
    # Deviation Justification (required by v9)
    #
    # Deviation type: outputs
    # What changed (exactly):
    #   This script aggregates many (file, time_index) frames into ONE time–pressure PNG.
    # Why it is technically required:
    #   The diagnostic is a time–pressure cross-section; it is not meaningful as per-frame PNGs.
    # Why v9 canonical cannot satisfy this case:
    #   v9 frame semantics define one PNG per (file, time_index); this product is inherently multi-time.
    # Evidence:
    #   N/A (product definition)
    # Scope:
    #   This script only (time–pressure point wind barbs ≤ 4 km)
    # Rollback plan:
    #   Revert to per-frame PNGs only if product definition changes.
    ###############################################################################
    main()
