-- Per-scanner decode results for each candidate. NULL = not yet measured
-- (pre-migration rows; backfill via scripts/backfill_scanner_breakdown.py).
-- 0/1 once the scan_breakdown ran. The existing `scans` column stays as
-- the rolled-up "any decoder succeeded" for backwards compat.
ALTER TABLE candidates ADD COLUMN scans_cv2     INTEGER;
ALTER TABLE candidates ADD COLUMN scans_zxing   INTEGER;
ALTER TABLE candidates ADD COLUMN scans_qreader INTEGER;
