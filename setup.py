from setuptools import setup, find_packages

setup(
    name="hand-metadata-db",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "geopandas", 
        "duckdb",
        "fsspec",
        "shapely",
        "pyogrio",
        "fiona",
        "folium",
        "branca",
        "h3",
    ],
    python_requires=">=3.8",
)