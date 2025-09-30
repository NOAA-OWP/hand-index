import argparse
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import duckdb
import fsspec
import s3fs


def download_s3_file(s3_path: str, local_path: str, fs: s3fs.S3FileSystem) -> str:
    """Download a single file from S3 to local filesystem. This gets used to download many files in parallel in load_hand_suite when working with the catchment gpkg's. Downloading the gpkg data locally ended up being faster. Most likely because of high-latency seeks in GPKG files when trying to interact with them over S3."""
    try:
        fs.get(s3_path, local_path)
        return local_path
    except Exception as e:
        print(f"Failed to download {s3_path}: {e}")
        return None


def load_hand_suite(
    db_path: str,
    hand_dir: str,
    hand_version: str,
    h3_resolution: int,
    calb: bool,
    batch_size: int = 100,
):
    """
    Loads HAND data into DuckDB using parallel downloads and processing.
    """
    start_time = time.time()

    hand_dir = hand_dir.rstrip("/") + "/"
    # Just uncomment below and delete other base_glob_path when done with trials
    base_glob_path = f"{hand_dir}*/branches/*"
    # base_glob_path = f"{hand_dir}*12090301*/branches/*"
    gpkg_glob = f"{base_glob_path}/*gw_catchments*.gpkg"
    csv_glob_path = f"{hand_dir}*/" if calb else f"{base_glob_path}/"
    csv_glob = f"{csv_glob_path}hydroTable_*.csv"
    catch_glob = f"{base_glob_path}/*gw_catchments_reaches*.tif"
    rem_glob = f"{base_glob_path}/*rem_zeroed*.tif"

    conn = duckdb.connect(db_path)
    try:
        print("\nLoading extensions (httpfs, spatial, h3, aws)...")
        conn.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
        conn.execute("INSTALL h3 FROM community; LOAD h3;")
        conn.execute("INSTALL aws; LOAD aws;")

        # Configure memory management and spilling
        print("Configuring memory management...")
        conn.execute("SET memory_limit = '7GB';")
        conn.execute("SET temp_directory = '/tmp';")
        conn.execute("SET preserve_insertion_order = false;")

        print("Processing and inserting Catchment geometries...")

        # Create staging table for Catchments. This staging approach improves performance because it allows us to insert data without the database checking constraints on every insert
        print("Creating staging table for Catchments...")
        conn.execute("CREATE TEMP TABLE Catchments_Staging AS SELECT * FROM Catchments LIMIT 0;")

        # First get the list of files, then build a dynamic query
        file_list_query = f"SELECT file FROM glob('{gpkg_glob}')"
        files = conn.execute(file_list_query).fetchall()

        if not files:
            print("WARNING: No GeoPackage files found. Skipping catchment ingestion.")
            return

        print(f"Found {len(files)} GeoPackage files to process.")

        # Group files by branch directory and pick one file per branch (excluding gw_catchments_pixels_ files since they are different than the other gw_catchment files)
        print("Grouping files by branch directory...")
        branch_to_file = {}
        for (file_path,) in files:
            if "/branches/" in file_path and "_pixels_" not in file_path:
                # Extract branch directory
                parts = file_path.split("/branches/")
                if len(parts) > 1:
                    branch_dir = parts[0] + "/branches/" + parts[1].split("/")[0] + "/"
                    if branch_dir not in branch_to_file:
                        branch_to_file[branch_dir] = file_path  # Just take the first file found

        print(f"Found {len(branch_to_file)} unique branches from {len(files)} files.")

        # Process files in batches
        branch_files = list(branch_to_file.items())

        # Initialize S3 filesystem for parallel downloads if needed
        fs = s3fs.S3FileSystem() if hand_dir.startswith("s3://") else None

        for i in range(0, len(branch_files), batch_size):
            batch_branches = branch_files[i : i + batch_size]
            print(
                f"Processing batch {i//batch_size + 1}/{(len(branch_files) + batch_size - 1)//batch_size} ({len(batch_branches)} branches)..."
            )

            # Download files in parallel if using S3
            local_file_paths = []
            temp_dir = None

            if fs:
                temp_dir = tempfile.mkdtemp()

                with ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit download tasks
                    future_to_file = {}
                    for j, (branch_dir, file_path) in enumerate(batch_branches):
                        local_path = os.path.join(temp_dir, f"batch_{i}_{j}.gpkg")
                        future = executor.submit(download_s3_file, file_path, local_path, fs)
                        future_to_file[future] = (branch_dir, file_path, local_path)

                    # Collect results
                    for future in as_completed(future_to_file):
                        branch_dir, original_path, local_path = future_to_file[future]
                        result = future.result()
                        if result:
                            local_file_paths.append((branch_dir, original_path, local_path))
                        else:
                            print(f"Skipping failed download: {original_path}")

                print(f"Successfully downloaded {len(local_file_paths)} files.")
            else:
                # Use original paths for local files
                local_file_paths = [(branch_dir, file_path, file_path) for branch_dir, file_path in batch_branches]

            # Process each file individually - directly insert all geometries from GPKG
            print(f"Processing {len(local_file_paths)} files...")

            # Batch files to reduce round trips while avoiding single giant query
            for i in range(0, len(local_file_paths), batch_size):
                batch = local_file_paths[i : i + batch_size]

                # Build single query with all file reads, then process geometries
                file_reads = []
                for branch_dir, _, local_path in batch:
                    escaped_path = local_path.replace("'", "''")
                    file_reads.append(
                        f"SELECT geom, '{branch_dir}' AS branch_path FROM ST_Read('{escaped_path}') WHERE geom IS NOT NULL"
                    )

                batch_insert_sql = f"""
                INSERT INTO Catchments_Staging (catchment_id, hand_version_id, geometry, h3_index, branch_path)
                WITH all_geoms AS (
                    {' UNION ALL '.join(file_reads)}
                ),
                merged_geoms AS (
                    SELECT 
                        branch_path,
                        COUNT(*) AS geom_count,
                        -- 100 sets a 100 m tolerance in EPSG:5070
                        ST_Simplify(ST_Union_Agg(geom), 100) AS merged_geom
                    FROM all_geoms
                    GROUP BY branch_path
                )
                SELECT
                    uuid() AS catchment_id,
                    '{hand_version}' AS hand_version_id,
                    ST_AsWKB(ST_Force2D(merged_geom)) AS geometry,
                    h3_latlng_to_cell(
                        ST_Y(ST_Transform(ST_Centroid(merged_geom), 'EPSG:5070', 'EPSG:4326', true)),
                        ST_X(ST_Transform(ST_Centroid(merged_geom), 'EPSG:5070', 'EPSG:4326', true)),
                        {h3_resolution}
                    ) AS h3_index,
                    branch_path
                FROM merged_geoms
                WHERE merged_geom IS NOT NULL
                ;
                """

                try:
                    result = conn.execute(batch_insert_sql)
                    records_inserted = result.rowcount if hasattr(result, "rowcount") else len(batch)
                except Exception as e:
                    print(f"Error processing batch starting at index {i}: {e}")
                    continue

            print(f"-> Batch inserted catchment records.")

            # Clean up temporary files
            if temp_dir and os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir)

        # Perform bulk insert from staging to final table
        print("\nPerforming bulk insert from staging to Catchments table...")
        conn.execute("""
            INSERT INTO Catchments 
            SELECT * FROM Catchments_Staging 
            ON CONFLICT (catchment_id) DO NOTHING;
        """)
        final_count = conn.execute("SELECT COUNT(*) FROM Catchments").fetchone()[0]
        print(f"-> Total catchment records in table: {final_count}")

        # Clean up staging table
        print("Cleaning up Catchments staging table...")
        conn.execute("DROP TABLE Catchments_Staging;")

        # Loading entire catchments table first for data integrity. Catchments must exist before Hydrotables can reference them.

        print("\nProcessing and inserting HydroTable CSV paths...")

        csv_dir_extractor = f"regexp_extract(file, '(.*/{'branches/[^/]+/' if not calb else '[^/]+/'})')"

        hydrotable_insert_sql = f"""
        INSERT INTO Hydrotables (catchment_id, csv_path)
        SELECT DISTINCT
            c.catchment_id,
            csv_files.file AS csv_path
        FROM (SELECT file FROM glob('{csv_glob}')) AS csv_files
        JOIN Catchments c ON c.branch_path = {csv_dir_extractor};
        """

        conn.execute(hydrotable_insert_sql)
        hydrotable_count = conn.execute("SELECT COUNT(*) FROM Hydrotables").fetchone()[0]
        print(f"-> Total hydrotable CSV records in table: {hydrotable_count}")

        # Process REM rasters
        print("\nProcessing and inserting REM rasters...")

        rem_insert_sql = f"""
        INSERT INTO HAND_REM_Rasters (catchment_id, raster_path)
        SELECT
            c.catchment_id,
            rem_files.file AS raster_path
        FROM (SELECT file FROM glob('{rem_glob}')) AS rem_files
        JOIN Catchments c ON c.branch_path = regexp_extract(rem_files.file, '(.*/branches/[^/]+/)');
        """

        conn.execute(rem_insert_sql)
        rem_count = conn.execute("SELECT COUNT(*) FROM HAND_REM_Rasters").fetchone()[0]
        print(f"-> Total REM raster records in table: {rem_count}")

        # Process catchment rasters
        print("\nProcessing and inserting catchment rasters...")

        catch_insert_sql = f"""
        INSERT INTO HAND_Catchment_Rasters (catchment_id, raster_path)
        SELECT
            c.catchment_id,
            catch_files.file AS raster_path
        FROM (SELECT file FROM glob('{catch_glob}')) AS catch_files
        JOIN Catchments c ON c.branch_path = regexp_extract(catch_files.file, '(.*/branches/[^/]+/)');
        """

        conn.execute(catch_insert_sql)
        catch_count = conn.execute("SELECT COUNT(*) FROM HAND_Catchment_Rasters").fetchone()[0]
        print(f"-> Total catchment raster records in table: {catch_count}")

    except Exception as e:
        print(f"\nFATAL ERROR during data ingestion: {e}")
        raise
    finally:
        conn.close()
        print(f"--- Ingestion Finished in {time.time() - start_time:.2f} seconds ---")


def partition_tables_to_parquet(db_path: str, output_dir: str):
    """Exports tables from DuckDB to Parquet datasets."""
    print(f"\n--- Starting Parquet Export to {output_dir} ---")

    # Ensure output_dir ends with slash
    output_dir = output_dir.rstrip("/") + "/"

    with duckdb.connect(db_path, read_only=True) as conn:
        print("Loading extensions for S3 export (httpfs, aws)...")
        conn.execute("INSTALL httpfs; LOAD httpfs; INSTALL aws; LOAD aws;")

        # Export Catchments table with native partitioning
        start_time = time.time()
        print(f"\n  Exporting Catchments table partitioned by h3_index...")

        catchments_output = f"{output_dir}catchments/"
        conn.execute(f"""
            COPY (SELECT * FROM Catchments)
            TO '{catchments_output}'
            (FORMAT PARQUET, PARTITION_BY (h3_index), OVERWRITE_OR_IGNORE 1)
        """)

        elapsed = time.time() - start_time
        print(f"  -> Finished Catchments table in {elapsed:.2f} seconds.")

        # Export non-partitioned tables as single files
        single_file_tables = ["Hydrotables", "HAND_REM_Rasters", "HAND_Catchment_Rasters"]
        for table in single_file_tables:
            start_time = time.time()
            file_path = f"{output_dir}{table.lower()}.parquet"
            print(f"\n  Exporting '{table}' to '{file_path}' (single file)...")
            conn.execute(f"""
                COPY {table}
                TO '{file_path}'
                WITH (FORMAT PARQUET, OVERWRITE_OR_IGNORE 1);
            """)
            print(f"  -> Finished '{table}' in {time.time() - start_time:.2f} seconds.")

    print(f"\n--- Data successfully exported to: {output_dir} ---")


def main():
    p = argparse.ArgumentParser(description="HAND data loader and exporter for DuckDB.")
    p.add_argument("--db-path", required=True)
    p.add_argument(
        "--schema-path",
        default="./schema/hand-index-ver-fim100-uncalb.sql",
        help="Path to SQL schema file (default: ./schema/hand-index-ver-fim100.sql)",
    )
    p.add_argument("--hand-dir", required=True)
    p.add_argument("--hand-version", required=True)
    p.add_argument("--h3-resolution", type=int, default=1)
    p.add_argument("--calb", action="store_true")
    p.add_argument("--skip-load", action="store_true")
    p.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Batch size for processing files. The larger the tables that will be created the smaller this needs to be to avoid running out of memory while running the queries.",
    )
    p.add_argument("--output-dir", help="If provided, export tables to this S3 or local directory.")
    args = p.parse_args()

    # Validate output directory
    if args.output_dir:
        if args.output_dir.startswith("s3://"):
            try:
                fs, out_path = fsspec.core.url_to_fs(args.output_dir)
                if fs.exists(out_path):
                    print(f"Error: S3 output directory {args.output_dir} already exists.")
                    exit(1)
                print(f"S3 output directory {args.output_dir} validated")
            except Exception as e:
                print(f"Error: Could not validate S3 output directory: {e}")
                exit(1)
        else:
            out_path = Path(args.output_dir)
            if out_path.exists():
                print(f"Error: Output directory {args.output_dir} already exists.")
                exit(1)
            try:
                out_path.mkdir(parents=True, exist_ok=False)
                print(f"Created output directory: {args.output_dir}")
            except Exception as e:
                print(f"Error: Could not create output directory: {e}")
                exit(1)

    # Check if database already exists unless we're skipping load
    if not args.skip_load:
        if os.path.exists(args.db_path):
            print(
                f"Error: Database file already exists at {args.db_path}. Remove it or use --skip-load to work with existing database."
            )
            return
        with duckdb.connect(args.db_path) as conn:
            with open(args.schema_path, "r") as f:
                schema_sql = f.read()
                conn.execute(schema_sql)
        print(f"Database initialized at '{args.db_path}' using schema '{args.schema_path}'.")

    if not args.skip_load:
        load_hand_suite(
            db_path=args.db_path,
            hand_dir=args.hand_dir,
            hand_version=args.hand_version,
            h3_resolution=args.h3_resolution,
            calb=args.calb,
            batch_size=args.batch_size,
        )

    if args.output_dir:
        partition_tables_to_parquet(args.db_path, args.output_dir)

    print("\nWorkflow Complete.")


if __name__ == "__main__":
    main()
