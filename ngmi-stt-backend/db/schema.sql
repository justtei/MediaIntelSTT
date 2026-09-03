-- Recognized-speech corpus schema. Applied automatically on first container
-- start via docker-entrypoint-initdb.d (see docker-compose.yml).
--
-- Design intent: `segments` is the actual fine-tuning corpus — every
-- finalized transcript segment from every channel, tagged with enough
-- metadata (language, timing, which backend/device produced it) to filter,
-- dedupe, and export training splits later without re-deriving anything.

CREATE TABLE IF NOT EXISTS channels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('youtube', 'mic')),
    url TEXT,
    language TEXT NOT NULL DEFAULT 'ur',
    status TEXT NOT NULL DEFAULT 'starting'
        CHECK (status IN ('starting', 'live', 'stopped', 'error')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    stopped_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS segments (
    id BIGSERIAL PRIMARY KEY,
    channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    language TEXT NOT NULL,
    t0_s DOUBLE PRECISION,
    t1_s DOUBLE PRECISION,
    infer_s DOUBLE PRECISION,
    rtf DOUBLE PRECISION,
    reason TEXT,           -- why the segment was cut: max | silence | stop | stream-end
    source TEXT,           -- mic | stream (matches Segmenter.source)
    backend TEXT,          -- openvino | faster-whisper
    device TEXT,           -- CPU | GPU | cuda | ...
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_segments_channel_created ON segments (channel_id, created_at);
CREATE INDEX IF NOT EXISTS idx_segments_language ON segments (language);
CREATE INDEX IF NOT EXISTS idx_channels_status ON channels (status);
