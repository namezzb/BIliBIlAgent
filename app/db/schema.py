SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        summary_text TEXT,
        recent_context_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        run_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        intent TEXT,
        route TEXT,
        status TEXT NOT NULL,
        requires_confirmation INTEGER NOT NULL DEFAULT 0,
        approval_status TEXT,
        latest_reply TEXT,
        pending_actions_json TEXT,
        execution_plan_json TEXT,
        approval_requested_at TEXT,
        approval_resolved_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        step_key TEXT NOT NULL,
        step_name TEXT NOT NULL,
        status TEXT NOT NULL,
        input_summary TEXT,
        output_summary TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(run_id, step_key)
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS import_run_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        favorite_folder_id TEXT NOT NULL,
        video_id TEXT NOT NULL,
        bvid TEXT,
        title TEXT NOT NULL,
        status TEXT NOT NULL,
        needs_asr INTEGER NOT NULL DEFAULT 0,
        failure_reason TEXT,
        retryable INTEGER NOT NULL DEFAULT 0,
        manifest_json TEXT,
        asr_job_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(run_id, video_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_import_run_items_run_id
    ON import_run_items (run_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS user_memory_profiles (
        user_id TEXT PRIMARY KEY,
        profile_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_favorite_folders (
        favorite_folder_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        intro TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_videos (
        video_id TEXT PRIMARY KEY,
        bvid TEXT,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_favorite_videos (
        favorite_folder_id TEXT NOT NULL,
        video_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (favorite_folder_id, video_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_video_pages (
        page_id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        page_number INTEGER NOT NULL,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_text_chunks (
        chunk_id TEXT PRIMARY KEY,
        video_id TEXT NOT NULL,
        source_type TEXT NOT NULL,
        source_language TEXT,
        block_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        start_ms INTEGER,
        end_ms INTEGER,
        embedding_model TEXT NOT NULL,
        embedding_version TEXT NOT NULL,
        index_status TEXT NOT NULL,
        vector_document_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_knowledge_video_pages_video_id
    ON knowledge_video_pages (video_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_knowledge_favorite_videos_video_id
    ON knowledge_favorite_videos (video_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_knowledge_text_chunks_video_id
    ON knowledge_text_chunks (video_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_chunk_pages (
        chunk_id TEXT NOT NULL,
        page_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (chunk_id, page_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_knowledge_chunk_pages_page_id
    ON knowledge_chunk_pages (page_id)
    """,
]
