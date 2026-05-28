-- Fraction of the diffusion canvas the QR occupies. 1.0 = legacy
-- behavior (QR fills canvas). <1.0 centers a smaller QR with #808080
-- gray margin around it — the v2 community workflow trick that produces
-- "QR-as-feature, scene-around-it" outputs (cherry-tree-canopy /
-- temple-cluster / etc.).
ALTER TABLE jobs ADD COLUMN qr_coverage REAL NOT NULL DEFAULT 1.0;
