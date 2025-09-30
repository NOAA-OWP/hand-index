#!/usr/bin/env python3
"""
Interactive visualization test script for query-geojson.py results.
Plots:
1. Original test GeoJSON polygon (query ROI)
2. All catchments from partitioned data that intersect the ROI
3. Filtered catchments after overlap threshold is applied
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import duckdb
import folium
import branca
import geopandas as gpd
import pandas as pd
from shapely.wkb import loads
from shapely.wkt import dumps

# Import functions from our query script
from query_geojson import (
    create_partitioned_views,
    get_catchment_data_for_geojson_poly_split_partitioned,
    filter_dataframes_by_overlap,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_query_results(output_dir: str) -> Optional[gpd.GeoDataFrame]:
    """
    Load the query results from the output directory and combine them.
    Returns a GeoDataFrame with catchment_id and combined attributes.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning("Output directory does not exist: %s", output_dir)
        return None

    parquet_files = list(output_path.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No parquet files found in: %s", output_dir)
        return None

    # Load all parquet files and combine them
    dfs = []
    for file in parquet_files:
        try:
            catchment_id = file.stem  # filename without extension
            df = pd.read_parquet(file)
            df["catchment_id"] = catchment_id
            dfs.append(df)
        except Exception as e:
            logger.warning("Could not read %s: %s", file, e)

    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Loaded %d records from %d catchments", len(combined_df), len(parquet_files)
    )

    return combined_df


def get_all_intersecting_catchments(
    geojson_path: str, partitioned_path: str, h3_resolution: int = 1
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Get all catchments that intersect with the GeoJSON polygon (before overlap filtering).
    Returns both geometries and attributes.
    """
    # Connect to DuckDB and setup
    con = duckdb.connect(":memory:")

    # Load extensions
    con.execute("LOAD spatial;")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("INSTALL aws;")
    con.execute("LOAD aws;")
    con.execute("INSTALL h3 FROM community;")
    con.execute("LOAD h3;")

    # Create views
    create_partitioned_views(con, partitioned_path)

    # Get geometries and attributes (unfiltered)
    geoms, attrs, query_poly_5070 = (
        get_catchment_data_for_geojson_poly_split_partitioned(
            geojson_path, con
        )
    )

    con.close()
    return geoms, attrs


def get_all_catchments_in_region(
    geojson_path: str, partitioned_path: str, buffer_degrees: float = 0.5
) -> gpd.GeoDataFrame:
    """
    Get ALL catchments in the region (not just intersecting ones) for comparison.
    Uses a buffered bounding box around the query polygon.
    """
    # Load the test GeoJSON to get bounds
    roi_gdf = gpd.read_file(geojson_path)
    if roi_gdf.crs is None:
        roi_gdf.set_crs(epsg=4326, inplace=True)
    elif roi_gdf.crs.to_epsg() != 4326:
        roi_gdf = roi_gdf.to_crs(epsg=4326)

    # Get buffered bounds
    bounds = roi_gdf.total_bounds
    min_x, min_y, max_x, max_y = bounds
    buffered_bounds = [
        min_x - buffer_degrees,
        min_y - buffer_degrees,
        max_x + buffer_degrees,
        max_y + buffer_degrees,
    ]

    # Connect to DuckDB and setup
    con = duckdb.connect(":memory:")

    # Load extensions
    con.execute("LOAD spatial;")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute("INSTALL aws;")
    con.execute("LOAD aws;")
    con.execute("INSTALL h3 FROM community;")
    con.execute("LOAD h3;")

    # Create views
    create_partitioned_views(con, partitioned_path)

    # Query for all catchments in the buffered region
    sql = f"""
    SELECT 
        c.catchment_id,
        c.geometry AS geom_wkb
    FROM catchments_partitioned c
    WHERE ST_Intersects(
        ST_GeomFromWKB(c.geometry),
        ST_Transform(
            ST_MakeEnvelope({buffered_bounds[0]}, {buffered_bounds[1]}, {buffered_bounds[2]}, {buffered_bounds[3]}),
            'EPSG:4326', 'EPSG:5070', TRUE
        )
    )
    """

    result = con.execute(sql).fetch_df()
    con.close()

    if result.empty:
        return gpd.GeoDataFrame(
            columns=["catchment_id", "geometry"], geometry="geometry", crs="EPSG:5070"
        )

    # Convert WKB → shapely geometries
    result["geometry"] = result["geom_wkb"].apply(lambda wkb: loads(bytes(wkb)) if wkb is not None else None)
    all_catchments_gdf = gpd.GeoDataFrame(
        result[["catchment_id", "geometry"]],
        geometry="geometry",
        crs="EPSG:5070",
    )

    return all_catchments_gdf


def create_interactive_map(
    test_geojson_path: str,
    intersecting_catchments: gpd.GeoDataFrame,
    intersecting_attributes: pd.DataFrame,
    query_results: Optional[pd.DataFrame],
    all_catchments: Optional[gpd.GeoDataFrame] = None,
    output_html: str = "query_test_map.html",
):
    """
    Create an interactive Folium map showing:
    1. Test GeoJSON polygon (query ROI)
    2. All catchments in region (if provided) - light gray
    3. Query result catchments - green with hydrotable info
    """
    # Load the test GeoJSON
    roi_gdf = gpd.read_file(test_geojson_path)
    if roi_gdf.crs is None:
        roi_gdf.set_crs(epsg=4326, inplace=True)
    elif roi_gdf.crs.to_epsg() != 4326:
        roi_gdf = roi_gdf.to_crs(epsg=4326)

    # Calculate map center from ROI
    roi_bounds = roi_gdf.total_bounds
    center_lat = (roi_bounds[1] + roi_bounds[3]) / 2
    center_lon = (roi_bounds[0] + roi_bounds[2]) / 2

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=8, tiles="OpenStreetMap"
    )

    # Add all catchments in region first (if provided) - as background layer
    if all_catchments is not None:
        logger.info(f"Adding {len(all_catchments)} background catchments...")
        all_catchments_4326 = all_catchments.to_crs(epsg=4326)

        # Simplify geometries for performance
        all_catchments_4326["geometry"] = all_catchments_4326["geometry"].simplify(
            0.001
        )

        for idx, row in all_catchments_4326.iterrows():
            geom = row.geometry
            catchment_id = row.catchment_id

            # Skip invalid geometries
            if geom is None or geom.is_empty:
                continue

            try:
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda feature: {
                        "fillColor": "lightgray",
                        "color": "gray",
                        "weight": 1,
                        "fillOpacity": 0.2,
                    },
                    popup=folium.Popup(
                        f"Background Catchment ID: {catchment_id}", parse_html=True
                    ),
                    tooltip=f"Background catchment {catchment_id}",
                ).add_to(m)
            except Exception as e:
                logger.warning(
                    f"Could not add background catchment {catchment_id}: {e}"
                )
                continue

    # Add ROI polygon
    folium.GeoJson(
        roi_gdf.__geo_interface__,
        style_function=lambda feature: {
            "fillColor": "red",
            "color": "red",
            "weight": 3,
            "fillOpacity": 0.0,  # No fill, just outline
            "fill": False,  # Disable fill completely
        },
        popup=folium.Popup("Query ROI", parse_html=True),
        tooltip="Query ROI (Test GeoJSON)",
    ).add_to(m)

    # Reproject intersecting catchments to 4326 for mapping
    logger.info(f"Adding {len(intersecting_catchments)} intersecting catchments...")
    intersecting_4326 = intersecting_catchments.to_crs(epsg=4326)

    # Simplify geometries for performance
    intersecting_4326["geometry"] = intersecting_4326["geometry"].simplify(0.001)

    # Get query result catchment IDs (ensure they're strings for comparison)
    query_catchment_ids = set()
    if query_results is not None:
        query_catchment_ids = set(
            str(cid) for cid in query_results["catchment_id"].unique()
        )
        logger.info(
            f"Found {len(query_catchment_ids)} unique catchments in query results"
        )
        logger.info(f"Sample query result IDs: {list(query_catchment_ids)[:3]}")
    else:
        logger.info("No query results provided")

    # Add intersecting catchments with hydrotable info
    query_result_count = 0
    for idx, row in intersecting_4326.iterrows():
        geom = row.geometry
        catchment_id = row.catchment_id

        # Skip invalid geometries
        if geom is None or geom.is_empty:
            continue

        # Convert catchment_id to string for comparison
        catchment_id_str = str(catchment_id)

        # Check if this catchment has query results (compare as strings)
        is_in_results = catchment_id_str in query_catchment_ids
        if is_in_results:
            query_result_count += 1

        # Get hydrotable info for this catchment
        catchment_attrs = intersecting_attributes[
            intersecting_attributes["catchment_id"] == catchment_id
        ]

        # Build tooltip content (plain text - short)
        tooltip_lines = [
            f"Catchment: {catchment_id_str[-8:]}"
        ]  # Show last 8 chars of ID

        # Build popup content (proper HTML with DOCTYPE and full structure)
        if is_in_results:
            popup_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; padding: 0; }}
        .header {{ color: #2d5016; margin: 0 0 10px 0; font-size: 16px; font-weight: bold; }}
        .catchment-id {{ font-size: 9px; font-family: monospace; background: #f5f5f5; padding: 2px 4px; border: 1px solid #ddd; }}
        .info {{ margin: 5px 0; }}
        .label {{ font-weight: bold; }}
        .hydro-list {{ margin: 0; padding-left: 20px; }}
        .hydro-item {{ margin: 2px 0; font-size: 11px; }}
        .separator {{ margin: 10px 0; border-top: 1px solid #ddd; }}
        .raster-path {{ font-size: 8px; font-family: monospace; word-wrap: break-word; white-space: pre-wrap; background: #f9f9f9; padding: 4px; border: 1px solid #e0e0e0; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">✓ Query Result</div>
    <div class="info"><span class="label">Catchment ID:</span><br><span class="catchment-id">{catchment_id_str}</span></div>
"""
            tooltip_lines.insert(0, "✓ Query Result")
        else:
            popup_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; padding: 0; }}
        .header {{ color: #1f4788; margin: 0 0 10px 0; font-size: 16px; font-weight: bold; }}
        .catchment-id {{ font-size: 9px; font-family: monospace; background: #f5f5f5; padding: 2px 4px; border: 1px solid #ddd; }}
        .info {{ margin: 5px 0; }}
        .label {{ font-weight: bold; }}
        .hydro-list {{ margin: 0; padding-left: 20px; }}
        .hydro-item {{ margin: 2px 0; font-size: 11px; }}
        .separator {{ margin: 10px 0; border-top: 1px solid #ddd; }}
        .raster-path {{ font-size: 8px; font-family: monospace; word-break: break-all; background: #f9f9f9; padding: 2px; }}
    </style>
</head>
<body>
    <div class="header">• Intersecting Catchment</div>
    <div class="info"><span class="label">Catchment ID:</span><br><span class="catchment-id">{catchment_id_str}</span></div>
"""
            tooltip_lines.insert(0, "• Intersecting")

        if not catchment_attrs.empty:
            hydro_count = len(catchment_attrs)
            popup_html += f'<div class="info"><span class="label">Hydrotable records:</span> {hydro_count}</div>'
            tooltip_lines.append(f"Records: {hydro_count}")

            # Show sample hydrotable records
            display_count = min(3, hydro_count)
            sample_hydros = []
            popup_html += '<div class="info"><span class="label">Sample HydroIDs:</span></div><ul class="hydro-list">'
            for i, (_, attr_row) in enumerate(
                catchment_attrs.head(display_count).iterrows()
            ):
                hydro_id = attr_row.get("HydroID", "N/A")
                nwm_feature_id = attr_row.get("nwm_feature_id", "N/A")
                popup_html += f'<li class="hydro-item">{hydro_id} <em>(NWM: {nwm_feature_id})</em></li>'
                sample_hydros.append(f"{hydro_id}")
            if hydro_count > display_count:
                popup_html += f'<li class="hydro-item" style="font-style: italic; color: #666;">... and {hydro_count - display_count} more</li>'
            popup_html += "</ul>"

            if sample_hydros:
                tooltip_lines.append(f"HydroIDs: {', '.join(sample_hydros[:2])}")

            # Add raster info if available
            rem_path = catchment_attrs.iloc[0].get("rem_raster_path", "N/A")
            catch_path = catchment_attrs.iloc[0].get("catchment_raster_path", "N/A")

            # Show full paths with wrapping
            popup_html += f"""
            <div class="separator"></div>
            <div class="info"><span class="label">REM Raster:</span><br><div class="raster-path">{rem_path}</div></div>
            <div class="info"><span class="label">Catchment Raster:</span><br><div class="raster-path">{catch_path}</div></div>
            """
        else:
            popup_html += '<div class="info" style="color: #999; font-style: italic;">No hydrotable data found</div>'
            tooltip_lines.append("No hydrotable data")

        popup_html += "</body></html>"

        # Style based on whether it's in query results
        if is_in_results:
            style = {
                "fillColor": "green",
                "color": "darkgreen",
                "weight": 2,
                "fillOpacity": 0.7,
            }
        else:
            style = {
                "fillColor": "blue",
                "color": "blue",
                "weight": 2,
                "fillOpacity": 0.5,
            }

        # Create plain text tooltip
        tooltip_text = " | ".join(tooltip_lines)
        popup_content = popup_html

        try:
            # Use IFrame approach for better HTML rendering
            iframe = branca.element.IFrame(html=popup_content, width=420, height=350)
            popup = folium.Popup(iframe, max_width=450)

            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda feature, style=style: style,
                popup=popup,
                tooltip=tooltip_text,
            ).add_to(m)
        except Exception as e:
            logger.warning(f"Could not add catchment {catchment_id}: {e}")
            continue

    logger.info(f"Added {query_result_count} catchments as query results (green)")

    # Add legend
    legend_items = [
        '<p style="margin: 5px 0;"><span style="color:red">■</span> Query ROI</p>',
        '<p style="margin: 5px 0;"><span style="color:blue">■</span> Intersecting catchments</p>',
        '<p style="margin: 5px 0;"><span style="color:green">■</span> Query results</p>',
    ]

    if all_catchments is not None:
        legend_items.append(
            '<p style="margin: 5px 0;"><span style="color:lightgray">■</span> All catchments in region</p>'
        )

    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 280px; height: {140 + (20 if all_catchments is not None else 0)}px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4 style="margin: 0 0 10px 0;">Legend</h4>
    {''.join(legend_items)}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(output_html)
    logger.info("Interactive map saved to: %s", output_html)

    # Print summary
    intersecting_count = len(intersecting_catchments)
    query_result_count = len(query_catchment_ids) if query_results is not None else 0
    all_count = len(all_catchments) if all_catchments is not None else 0

    print(f"\n{'='*50}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*50}")
    print(f"Query ROI: {test_geojson_path}")
    if all_catchments is not None:
        print(f"All catchments in region: {all_count}")
    print(f"Intersecting catchments: {intersecting_count}")
    print(f"Query result catchments: {query_result_count}")
    print(f"Interactive map: {output_html}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize query-geojson.py results interactively"
    )
    parser.add_argument(
        "-g", "--geojson", required=True, help="Path to test GeoJSON file"
    )
    parser.add_argument(
        "-p",
        "--partitioned-path",
        required=True,
        help="Base path to partitioned parquet files",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        help="Directory containing query results (parquet files). If not provided, will show all intersecting catchments.",
    )
    parser.add_argument(
        "-o",
        "--output-html",
        default="query_test_map.html",
        help="Output HTML file for interactive map (default: query_test_map.html)",
    )
    parser.add_argument(
        "--h3-resolution",
        type=int,
        default=1,
        help="H3 resolution used for partitioning (default: 1)",
    )
    parser.add_argument(
        "--show-all-catchments",
        action="store_true",
        help="Show all catchments in the region as background (light gray)",
    )
    parser.add_argument(
        "--max-catchments",
        type=int,
        default=1000,
        help="Maximum number of background catchments to show (default: 1000)",
    )

    args = parser.parse_args()

    logger.info("Loading intersecting catchments...")
    intersecting_catchments, intersecting_attributes = get_all_intersecting_catchments(
        args.geojson, args.partitioned_path, args.h3_resolution
    )

    if intersecting_catchments.empty:
        logger.error("No catchments found intersecting the query polygon!")
        return

    logger.info("Found %d intersecting catchments", len(intersecting_catchments))

    # Load query results if provided
    query_results = None
    if args.results_dir:
        logger.info("Loading query results from: %s", args.results_dir)
        query_results = load_query_results(args.results_dir)

    # Load all catchments in region if requested
    all_catchments = None
    if args.show_all_catchments:
        logger.info("Loading all catchments in region...")
        all_catchments = get_all_catchments_in_region(
            args.geojson, args.partitioned_path
        )
        logger.info("Found %d total catchments in region", len(all_catchments))

        # Limit number of background catchments for performance
        if len(all_catchments) > args.max_catchments:
            logger.warning(
                f"Limiting to {args.max_catchments} catchments for performance"
            )
            all_catchments = all_catchments.head(args.max_catchments)

    # Create interactive map
    logger.info("Creating interactive map...")
    create_interactive_map(
        args.geojson,
        intersecting_catchments,
        intersecting_attributes,
        query_results,
        all_catchments,
        args.output_html,
    )


if __name__ == "__main__":
    main()
