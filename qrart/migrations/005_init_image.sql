-- User-uploaded init image (content-addressed). When set, pass-1 runs as
-- ControlNet img2img seeded with this image at (1 - init_strength)
-- preservation. For non-standalone compositions, this image replaces the
-- generated scene that the QR art is composited into.
ALTER TABLE jobs ADD COLUMN init_image_path TEXT;
ALTER TABLE jobs ADD COLUMN init_strength REAL NOT NULL DEFAULT 0.65;
