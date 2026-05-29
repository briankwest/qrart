-- "Prefer scannable" toggle: when on, the server bumps controlnet_scale
-- +0.10 and candidates +2 and forces auto_escalate + require_scan ON.
-- Default ON so existing flow biases toward phone-readable outputs.
ALTER TABLE jobs ADD COLUMN prefer_scannable INTEGER NOT NULL DEFAULT 1;
