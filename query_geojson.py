import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import duckdb
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.wkb import loads
from shapely.wkt import dumps

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def create_partitioned_views(con: duckdb.DuckDBPyConnection, base_path: str):
    """
    Create views for partitioned tables.
    """
    # Ensure base_path ends with /
    if not base_path.endswith("/"):
        base_path += "/"

    views_sql = f"""
    CREATE OR REPLACE VIEW catchments_partitioned AS
    SELECT * FROM read_parquet('{base_path}catchments/*/*.parquet', hive_partitioning = 1);
    
    CREATE OR REPLACE VIEW hydrotables_partitioned AS
    SELECT * FROM read_parquet('{base_path}hydrotables/*/*.parquet', hive_partitioning = 1);
    
    CREATE OR REPLACE VIEW hand_rem_rasters_partitioned AS
    SELECT * FROM read_parquet('{base_path}hand_rem_rasters.parquet');
    
    CREATE OR REPLACE VIEW hand_catchment_rasters_partitioned AS
    SELECT * FROM read_parquet('{base_path}hand_catchment_rasters.parquet');
    
    """

    for stmt in views_sql.strip().split(";"):
        if stmt.strip():
            con.execute(stmt.strip() + ";")


def _partitioned_query_cte(wkt4326: str) -> str:
    """
    Returns the CTE for querying partitioned tables using direct spatial intersection.
    """
    return f"""
    WITH input_query AS (
      SELECT '{wkt4326}' AS wkt_string
    ),
    transformed_query AS (
      SELECT
        ST_Transform(
          ST_GeomFromText(wkt_string),
          'EPSG:4326', 'EPSG:5070', TRUE
        ) AS query_geom
      FROM input_query
    ),
    filtered_catchments AS (
      SELECT
        c.catchment_id,
        c.geometry,
        c.h3_index
      FROM catchments_partitioned c
      JOIN transformed_query tq ON ST_Intersects(ST_GeomFromWKB(c.geometry), tq.query_geom)
    )
    """


def get_catchment_data_for_geojson_poly_split_partitioned(
    geojson_fp: str, con: duckdb.DuckDBPyConnection
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Optional[Polygon]]:
    """
    Reads a GeoJSON polygon, runs two spatial queries against DuckDB
    (one for geometries, one for attributes), and returns:

      - geometries_gdf: GeoDataFrame[c   atchment_id, geometry in EPSG:5070]
      - attributes_df: DataFrame[*all attribute columns* including catchment_id]
      - query_polygon_5070: shapely Polygon for later overlap filtering
    """
    # 1) Load & reproject the query polygon
    gdf = gpd.read_file(geojson_fp)
    if gdf.empty:
        logger.error("GeoJSON is empty or invalid: %s", geojson_fp)
        return gpd.GeoDataFrame(), pd.DataFrame(), None

    # Ensure CRS is EPSG:4326 for injection into SQL
    if gdf.crs is None:
        logger.info("Assuming input GeoJSON CRS = EPSG:4326")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        logger.info("Reprojecting query polygon to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)

    src_poly_4326 = gdf.geometry.iloc[0]
    wkt4326 = dumps(src_poly_4326)

    # Also get the same polygon in EPSG:5070 for Python overlap checks
    query_poly_5070 = gdf.to_crs(epsg=5070).geometry.iloc[0]

    # 2) Build and run the geometry-only query using direct spatial intersection
    cte = _partitioned_query_cte(wkt4326)
    sql_geom = (
        cte
        + """
    SELECT
      fc.catchment_id,
      fc.geometry AS geom_wkb
    FROM filtered_catchments AS fc;
    """
    )
    geom_df = con.execute(sql_geom).fetch_df()
    if geom_df.empty:
        logger.info("No catchments intersect the query polygon.")
        empty_gdf = gpd.GeoDataFrame(
            columns=["catchment_id", "geometry"],
            geometry="geometry",
            crs="EPSG:5070",
        )
        return empty_gdf, pd.DataFrame(), query_poly_5070

    # Convert WKB data to Shapely geometry objects. Wrapping wkb in bytes because duckdb exports a bytearray but shapely wants bytes.
    geom_df["geometry"] = geom_df["geom_wkb"].apply(
        lambda wkb: loads(bytes(wkb)) if wkb is not None else None
    )
    geometries_gdf = gpd.GeoDataFrame(
        geom_df[["catchment_id", "geometry"]],
        geometry="geometry",
        crs="EPSG:5070",
    )

    # 3) Build and run the attribute query using partitioned tables
    sql_attr = (
        cte
        + """
    SELECT
      fc.catchment_id,
      h.* EXCLUDE (catchment_id, h3_index),
      hrr.raster_path AS rem_raster_path,
      hcr.raster_path AS catchment_raster_path
    FROM filtered_catchments AS fc
    LEFT JOIN hydrotables_partitioned AS h ON fc.catchment_id = h.catchment_id
    LEFT JOIN hand_rem_rasters_partitioned AS hrr ON fc.catchment_id = hrr.catchment_id
    LEFT JOIN hand_catchment_rasters_partitioned AS hcr ON fc.catchment_id = hcr.catchment_id;
    """
    )
    attributes_df = con.execute(sql_attr).fetch_df()

    return geometries_gdf, attributes_df, query_poly_5070


def filter_dataframes_by_overlap(
    geometries_gdf: gpd.GeoDataFrame,
    attributes_df: pd.DataFrame,
    query_polygon_5070: Polygon,
    overlap_threshold_percent: float = 10.0,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Filters geometries and attributes using contains/within/overlap logic.

    Selection criteria:
    - Catchments that completely contain the query polygon, OR
    - Catchments that are completely within the query polygon, OR
    - Catchments that overlap more than the threshold percentage of their own area

    Returns (filtered_geoms, filtered_attrs, summary_stats).
    """
    stats = {
        "initial_geoms": len(geometries_gdf),
        "initial_attrs": len(attributes_df),
    }
    if geometries_gdf.empty or query_polygon_5070 is None:
        stats.update(
            final_geoms=0, final_attrs=0, removed_geoms=0, removed_attrs=0
        )
        return geometries_gdf.copy(), attributes_df.copy(), stats

    geoms = geometries_gdf.copy()

    # Compute geometric relationships
    geoms["area"] = geoms.geometry.area
    geoms["inter"] = geoms.geometry.apply(
        lambda g: g.intersection(query_polygon_5070).area
        if not g.is_empty
        else 0.0
    )
    geoms["overlap_pct"] = (
        geoms["inter"] / geoms["area"].replace({0: pd.NA})
    ) * 100
    geoms["overlap_pct"] = geoms["overlap_pct"].fillna(0.0)

    # Check contains/within relationships
    geoms["contains_query"] = geoms.geometry.apply(
        lambda g: g.contains(query_polygon_5070)
    )
    geoms["within_query"] = geoms.geometry.apply(
        lambda g: g.within(query_polygon_5070)
    )

    # Debug: log the geometric relationships
    contains_count = geoms["contains_query"].sum()
    within_count = geoms["within_query"].sum()
    logger.info(
        f"Geometric relationships: {contains_count} contain query, {within_count} within query"
    )

    # Apply selection criteria: contains OR within OR overlap > threshold
    overlap_threshold_decimal = overlap_threshold_percent / 100.0
    selection_mask = (
        geoms["contains_query"]
        | geoms["within_query"]
        | (geoms["overlap_pct"] >= overlap_threshold_percent)
    )

    keep_ids = set(geoms.loc[selection_mask, "catchment_id"])
    filtered_geoms = geoms[geoms.catchment_id.isin(keep_ids)].drop(
        columns=[
            "area",
            "inter",
            "overlap_pct",
            "contains_query",
            "within_query",
        ]
    )
    filtered_attrs = attributes_df[
        attributes_df.catchment_id.isin(keep_ids)
    ].copy()

    # Enhanced statistics
    stats["final_geoms"] = len(filtered_geoms)
    stats["final_attrs"] = len(filtered_attrs)
    stats["removed_geoms"] = stats["initial_geoms"] - stats["final_geoms"]
    stats["removed_attrs"] = stats["initial_attrs"] - stats["final_attrs"]
    stats["contains_count"] = geoms["contains_query"].sum()
    stats["within_count"] = geoms["within_query"].sum()
    stats["overlap_only_count"] = (
        (geoms["overlap_pct"] >= overlap_threshold_percent)
        & ~geoms["contains_query"]
        & ~geoms["within_query"]
    ).sum()

    return filtered_geoms, filtered_attrs, stats


def main():
    """
    Fetch catchment data from DuckDB, filter by spatial overlap,
    and write per-catchment attribute tables to separate Parquet files.
    """

    p = argparse.ArgumentParser(
        description="Fetch/filter catchments and write per-ID Parquet files"
    )
    p.add_argument(
        "-g", "--geojson", required=True, help="Path to query GeoJSON"
    )
    p.add_argument(
        "-p",
        "--partitioned-path",
        required=True,
        help="Base path to partitioned parquet files (local path or s3://...)",
    )
    p.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=10.0,
        help="Min overlap % to keep a catchment",
    )
    p.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Directory to write <catchment_id>.parquet files",
    )
    args = p.parse_args()

    # Connect to DuckDB in-memory for partitioned mode
    con = duckdb.connect(":memory:")
    logger.info("Using partitioned mode with path: %s", args.partitioned_path)

    # Load required extensions
    try:
        con.execute("INSTALL spatial;")
        con.execute("LOAD spatial;")
        con.execute("INSTALL httpfs;")
        con.execute("LOAD httpfs;")
        con.execute("INSTALL aws;")
        con.execute("LOAD aws;")
    except duckdb.Error as e:
        logger.warning("Could not load DuckDB extensions: %s", e)

    # Setup views for partitioned mode
    logger.info("Creating views for partitioned tables...")
    create_partitioned_views(con, args.partitioned_path)

    # Fetch & filter
    geoms, attrs, query_poly_5070 = (
        get_catchment_data_for_geojson_poly_split_partitioned(args.geojson, con)
    )
    if geoms.empty:
        logger.info("No geometries found. Exiting.")
        return

    fg, fa, stats = filter_dataframes_by_overlap(
        geoms, attrs, query_poly_5070, args.threshold
    )
    logger.info("Overlap filter summary: %s", stats)

    # Prepare output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write one parquet per catchment_id (drop the catchment_id column inside each file)
    for catch_id, group in fa.groupby("catchment_id"):
        df = group.drop(columns=["catchment_id"]).copy()

        # Convert UUID columns to strings for Parquet compatibility
        uuid_columns = ["rem_raster_id", "catchment_raster_id"]
        for col in uuid_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)

        out_path = outdir / f"{catch_id}.parquet"
        df.to_parquet(str(out_path), index=False)
        logger.info(
            "Wrote %d rows for catchment '%s' → %s", len(df), catch_id, out_path
        )

    con.close()


if __name__ == "__main__":
    main()
