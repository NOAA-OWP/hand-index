-- Enable DuckDB spatial extension
INSTALL spatial;
LOAD spatial;

CREATE TABLE Catchments (
    catchment_id UUID PRIMARY KEY,
    hand_version_id TEXT, 
    geometry BLOB, -- storing geometry as WKB
    h3_index BIGINT,
    branch_path TEXT
);

CREATE TABLE Hydrotables (
    catchment_id UUID REFERENCES Catchments(catchment_id),
    csv_path TEXT
);

CREATE TABLE HAND_REM_Rasters (
    catchment_id UUID REFERENCES Catchments(catchment_id),
    raster_path TEXT,
);

CREATE TABLE HAND_Catchment_Rasters (
    catchment_id UUID REFERENCES Catchments(catchment_id),
    raster_path TEXT,
);

-- Create indexes
CREATE INDEX idx_hydrotables_catchment ON Hydrotables(catchment_id);
CREATE INDEX idx_catchments_branch_path ON Catchments(branch_path);
