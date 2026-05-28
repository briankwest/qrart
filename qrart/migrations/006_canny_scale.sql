-- Canny ControlNet scale. > 0 with an init image present stacks Canny
-- edge conditioning into the multi-controlnet — preserves logo/structure
-- while QR Monster paints the modules around it.
ALTER TABLE jobs ADD COLUMN canny_scale REAL NOT NULL DEFAULT 0.0;
