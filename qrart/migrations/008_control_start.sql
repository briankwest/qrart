-- ControlNet window: fraction of total diffusion steps at which ControlNet
-- starts contributing. 0.30 default is the QR Monster v2 community sweet
-- spot — lets the first 30% of denoising paint the scene from the prompt
-- freely, then QR Monster shapes the result into a QR pattern.
ALTER TABLE jobs ADD COLUMN control_start REAL NOT NULL DEFAULT 0.30;
