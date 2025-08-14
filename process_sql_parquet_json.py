import os, re, json, httpx, tempfile, shutil, sqlite3
import uuid
from typing import List, Optional, Tuple
from fastapi import UploadFile
import duckdb
from sqlalchemy import create_engine, text
import pandas as pd
import asyncio
from starlette.datastructures import UploadFile
from typing import Literal, Union, Dict, Any, Tuple
SQLITE_EXTS = (".db", ".sqlite", ".sqlite3")
DUCKDB_EXTS = (".duckdb",)
SQL_EXTS    = (".sql",)
TABULAR_EXTS= (".parquet", ".json")   # NOTE: csv/tsv removed
DANGEROUS   = [r"\bATTACH\b", r"\bDETACH\b", r"\bLOAD\b", r"\.read\b", r"\.shell\b"]

def _block(sql_text: str):
    for pat in DANGEROUS:
        if re.search(pat, sql_text, flags=re.I):
            raise ValueError(f"Blocked SQL token: {pat}")

def _ext(name: str) -> str:
    return os.path.splitext(name.lower())[1]

async def _dl(urls: List[str]) -> List[Tuple[str, bytes]]:
    out = []
    for u in urls or []:
        try:
            r = httpx.get(u, timeout=30, follow_redirects=True)
            r.raise_for_status()
            out.append((u, r.content))
            print(f"[sql_agent] GET {u} -> {r.status_code}, ct={r.headers.get('content-type')}")
        except Exception as e:
            print(f"[sql_agent] URL error {u}: {e}")
    return out

def _safe_write(path: str, data: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

class SQLContextBuilder:
    """
    Session engine = DuckDB at session.duckdb
    - Attaches SQLite via sqlite_scanner
    - Imports DuckDB db tables
    - Imports .parquet / .json (local files or HTTP URLs)
    - Applies user .sql with safety blocklist
    """
    def __init__(self, base_dir: str):
        
        self.duckdb = duckdb
        self.base_dir = base_dir
        self._published: list[str] = []
        os.makedirs(self.base_dir, exist_ok=True)
        self.db_path = os.path.join(self.base_dir, "session.duckdb")
        if os.path.exists(self.db_path):
            try: os.remove(self.db_path)
            except: pass
        self.con = self.duckdb.connect(self.db_path)
        try: self.con.execute("INSTALL httpfs; LOAD httpfs;")   # for URL parquet/json
        except Exception: pass
        try: self.con.execute("INSTALL sqlite; LOAD sqlite;")    # sqlite_scanner
        except Exception: pass

    def close(self):
        try: self.con.close()
        except Exception: pass

    def register_sqlite_db(self, path: str, alias: str):
        # Ensure sqlite extension is available
        try:
            self.con.execute("INSTALL sqlite; LOAD sqlite;")
        except Exception:
            pass

        # Attach the SQLite file as a schema in DuckDB
        alias_sanitized = re.sub(r"[^A-Za-z0-9_]", "_", alias)
        quoted_alias = '"' + alias_sanitized.replace('"', '""') + '"'
        self.con.execute(f"ATTACH '{path}' AS {quoted_alias} (TYPE SQLITE);")

        # ---- Read ONLY TABLES from the SQLite file (skip views) ----
        tbls = []
        sconn = sqlite3.connect(path)
        try:
            rows = sconn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
            tbls = [r[0] for r in rows]
        finally:
            sconn.close()

        # ---- Publish each table as a view in DuckDB main schema ----
        for tname in tbls:
            safe_t = re.sub(r"[^A-Za-z0-9_]", "_", tname)
            quoted_t = '"' + tname.replace('"', '""') + '"'
            # Create a simple passthrough view to the attached SQLite table
            self.con.execute(
                f'CREATE OR REPLACE VIEW {safe_t} AS SELECT * FROM {quoted_alias}.{quoted_t};'
            )
            self._published.append(safe_t)
        print(f"[sql_agent] attached sqlite '{os.path.basename(path)}' -> {len(tbls)} tables exposed (views skipped)")
        

    def _qident(self, name: str) -> str:
    # safe double-quote for identifiers
        return '"' + str(name).replace('"', '""') + '"'
    # NEW: quote a SQL string literal safely
    def _qstring(self, s: str) -> str:
        return "'" + str(s).replace("'", "''") + "'"

    def register_duckdb_db(self, path: str):
        import os, uuid
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(abs_path)

        dbname = f"db_{uuid.uuid4().hex[:8]}"

        # 1) ATTACH **with a quoted string literal**, not parameter binding
        path_sql = self._qstring(abs_path.replace("\\", "/"))
        self.con.execute(f"ATTACH {path_sql} AS {self._qident(dbname)} (READ_ONLY)")

        # 2) List base tables robustly across DuckDB versions
        tables = []
        # Try information_schema first (most stable)
        try:
            rows = self.con.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_catalog = ? AND table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                [dbname],
            ).fetchall()
            tables = [r[0] for r in rows]
        except Exception:
            pass

        # Fall back to duckdb_tables() (some builds have fewer columns)
        if not tables:
            try:
                rows = self.con.execute(
                    """
                    SELECT DISTINCT table_name
                    FROM duckdb_tables()
                    WHERE database_name = ?
                    ORDER BY table_name
                    """,
                    [dbname],
                ).fetchall()
                tables = [r[0] for r in rows]
            except Exception:
                pass

        # Final fallback: SHOW TABLES FROM db.schema
        if not tables:
            try:
                rows = self.con.execute(
                    f"SHOW TABLES FROM {self._qident(dbname)}.main"
                ).fetchall()
                # SHOW returns a single column (name)
                tables = [r[0] for r in rows]
            except Exception:
                tables = []

        # Expose tables via simple views in the current (default) database
        if not hasattr(self, "_published"):
            self._published = []
        for t in tables:
            qt = self._qident(t)
            self.con.execute(
                f"CREATE OR REPLACE VIEW {qt} AS SELECT * FROM {self._qident(dbname)}.main.{qt}"
            )
            if t not in self._published:
                self._published.append(t)

        print(f"[sql_agent] attached duckdb '{os.path.basename(abs_path)}' as {dbname} -> {len(tables)} tables exposed (via views)")

    def register_tabular_file(self, path: str, table: Optional[str] = None):
        ext = _ext(path)
        table = table or re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(path))[0])
        if ext == ".parquet":
            self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{path}')")
        elif ext == ".json":
            self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_json_auto('{path}')")
        else:
            raise ValueError(f"Unsupported tabular ext: {ext}")
        self._published.append(table)


    def register_tabular_url(self, url: str, table: Optional[str] = None):
        # stream via httpfs
        ext = _ext(url.split("?",1)[0])
        table = table or re.sub(r"[^A-Za-z0-9_]", "_", os.path.basename(url).split("?",1)[0])
        if ext == ".parquet":
            self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{url}')")
        elif ext == ".json":
            self.con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_json_auto('{url}')")
        else:
            raise ValueError(f"Unsupported tabular URL ext: {ext}")
        self._published.append(table)


    def apply_user_sql(self, name: str, data: bytes):
        text = data.decode("utf-8", errors="replace")
        _block(text)
        print(f"[sql_agent] duckdb.executescript({name}, len={len(text)})")
        try:
            # Try to run directly in DuckDB (works for many .sql files)
            self.con.execute(text)
            return
        except Exception as e:
            print(f"[sql_agent] DuckDB parse failed: {e}; falling back to sqlite -> attach")

        # Fallback: execute SQL in a SQLite file, then attach to DuckDB
        base = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(name))[0])
        tmp_sqlite = os.path.join(self.base_dir, f"{base}.sqlite")

        sx = sqlite3.connect(tmp_sqlite)
        try:
            sx.executescript(text)   # SQLite understands [] identifiers etc.
            sx.commit()
        finally:
            sx.close()

        # Attach that SQLite DB into DuckDB (requires sqlite extension)
        self.register_sqlite_db(tmp_sqlite, alias=base)

    def _sanitize_preview(self, v, max_chars=120):
        try:
            max_chars = int(max_chars)
        except Exception:
            max_chars = 120

        if isinstance(v, memoryview):
            v = v.tobytes()

        # BLOBs -> placeholder
        if isinstance(v, (bytes, bytearray)):
            return f"<{type(v).__name__} {len(v)} bytes>"

        # Dates -> ISO
        from datetime import date, datetime
        if isinstance(v, (date, datetime)):
            return v.isoformat()

        # Decimal -> float (or int if integral)
        try:
            import decimal
            if isinstance(v, decimal.Decimal):
                f = float(v)
                return int(f) if float(f).is_integer() else f
        except Exception:
            pass

        # Keep primitives as-is
        if isinstance(v, (int, float, bool)) or v is None:
            return v

        # Strings/others -> single-line, trimmed, ellipsized
        s = v if isinstance(v, str) else str(v)
        s = s.replace("\r", " ").replace("\n", " ").strip()
        if max_chars < 5:
            max_chars = 5
        return (s[: max_chars - 1] + "…") if len(s) > max_chars else s


    def summarize(self, task_hint: str = "") -> str:
        # 0) Table list: prefer what you "published"; else discover in duckdb
        if hasattr(self, "_published") and self._published:
            tables = [t for t in dict.fromkeys(self._published) if not str(t).startswith("sqlite_")]
        else:
            tables = [r[0] for r in self.con.execute(
                "SELECT table_name FROM duckdb_tables() "
                "WHERE schema_name='main' AND table_name NOT LIKE 'sqlite_%' "
                "ORDER BY table_name"
            ).fetchall()]

        parts = [
            "SQL_CONTEXT",
            "ENGINE: duckdb",
            f"SESSION_DB_PATH: {self.db_path}",
            f"TABLES ({len(tables)}): " + (", ".join(tables) if tables else ""),
        ]

        for t in tables:
            # 1) Columns (prefer duckdb_columns; fallback to PRAGMA)
            try:
                cols_raw = self.con.execute(
                    "SELECT column_name, data_type "
                    "FROM duckdb_columns() "
                    "WHERE schema_name='main' AND table_name=? "
                    "ORDER BY column_index",
                    [t]
                ).fetchall()
            except Exception:
                info = self.con.execute(f"PRAGMA table_info({t})").fetchall()
                # PRAGMA returns: (cid, name, type, notnull, dflt_value, pk)
                cols_raw = [(r[1], r[2]) for r in info]

            # de-duplicate while preserving order
            seen = set()
            cols = []
            for c, ty in cols_raw:
                if c not in seen:
                    cols.append((c, ty))
                    seen.add(c)

            # 2) Row count (best-effort)
            try:
                n = self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception as e:
                n = f"? ({e})"

            # 3) Sample (sanitize BLOBs / long strings)
            try:
                rows = self.con.execute(f"SELECT * FROM {t} LIMIT 5").fetchall()
                headers = [d[0] for d in self.con.description] if self.con.description else []
                sample = []
                for r in rows:
                    sample.append({h: self._sanitize_preview(v) for h, v in zip(headers, r)})
            except Exception as e:
                sample = [{"_error": str(e)}]

            parts.append(f"- {t}: rows={n}, cols=" + ", ".join([f"{c}:{ty}" for c, ty in cols]))
            parts.append(f"  sample: {json.dumps(sample, ensure_ascii=False)}")

        if task_hint.strip():
            parts.append(f"TASK_HINT: {task_hint.strip()}")
        return "\n".join(parts)

    def summarize_json(self, task_hint: str = "", sample_rows: int = 5) -> dict:
        import os, json

        session_path = os.path.abspath(self.db_path).replace("\\", "/")
        tables = list(dict.fromkeys(self._published))
        out = {
            "engine": "duckdb",
            "session_db_path": session_path,          # <- pass this to your SQL agent
            "tables": [],
        }
        if task_hint.strip():
            out["task_hint"] = task_hint.strip()

        for t in tables:
            # columns (same as summarize)
            try:
                cols_raw = self.con.execute(
                    "SELECT column_name, data_type "
                    "FROM duckdb_columns() "
                    "WHERE schema_name='main' AND table_name=? "
                    "ORDER BY column_index",
                    [t]
                ).fetchall()
            except Exception:
                info = self.con.execute(f"PRAGMA table_info({t})").fetchall()
                cols_raw = [(r[1], r[2]) for r in info]

            seen = set()
            cols = []
            for c, ty in cols_raw:
                if c not in seen:
                    cols.append({"name": c, "type": ty})
                    seen.add(c)

            # row count
            try:
                n = self.con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except Exception as e:
                n = {"_error": str(e)}

            # sample (use your _sanitize_preview)
            try:
                rows = self.con.execute(f"SELECT * FROM {t} LIMIT ?", [sample_rows]).fetchall()
                headers = [d[0] for d in self.con.description] if self.con.description else []
                sample = [{h: self._sanitize_preview(v) for h, v in zip(headers, r)} for r in rows]
            except Exception as e:
                sample = [{"_error": str(e)}]

            out["tables"].append({
                "name": t,
                "row_count": n,
                "columns": cols,
                "sample": sample,
            })

        return out

# --- PROCESSING FUNCTION ----------
async def process_sql_parquet_json(
    task: str = "",
    # uploads
    db_files: Optional[List[UploadFile]] = None,         # .db/.sqlite/.sqlite3/.duckdb
    sql_files: Optional[List[UploadFile]] = None,        # .sql
    parquet_json_files: Optional[List[UploadFile]] = None,  # .parquet/.json (no csv/tsv/zip)
    # urls by type
    db_urls: Optional[List[str]] = None,                 # URLs to sqlite/duckdb files (downloaded)
    sql_urls: Optional[List[str]] = None,                # URLs to .sql files (downloaded)
    parquet_json_urls: Optional[List[str]] = None,       # URLs to .parquet/.json (streamed)
    # optional external DB URIs (introspect & sample)
    external_uris: Optional[List[str]] = None,
    # where to persist session.duckdb so the master agent can open it
    persist_dir: Optional[str] = None,
    return_format: Literal["text","json","both"] = "text",
) -> Union[str, Dict[str, Any], Tuple[Dict[str, Any], str]]:
    db_files            = db_files or []
    sql_files           = sql_files or []
    parquet_json_files  = parquet_json_files or []
    db_urls             = db_urls or []
    sql_urls            = sql_urls or []
    parquet_json_urls   = parquet_json_urls or []
    external_uris       = external_uris or []

    print(f"[sql_agent] START db_files={len(db_files)} sql_files={len(sql_files)} pj_files={len(parquet_json_files)} db_urls={len(db_urls)} sql_urls={len(sql_urls)} pj_urls={len(parquet_json_urls)} external_uris={len(external_uris)}")

    # Download things that need local bytes (DBs & .sql)
    url_db_bytes  = await _dl(db_urls)
    url_sql_bytes = await _dl(sql_urls)

    base_dir = persist_dir or os.path.join(os.getcwd(), "_session_sql")
    os.makedirs(base_dir, exist_ok=True)
    work_dir = os.path.join(base_dir, "_work")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    builder = SQLContextBuilder(base_dir=base_dir)

    try:
        # 1) Handle DB uploads/URLs
        def handle_db_blob(name: str, data: bytes):
            e = _ext(name)
            path = os.path.join(work_dir, os.path.basename(name))
            _safe_write(path, data)
            if e in SQLITE_EXTS:
                alias = re.sub(r"[^A-Za-z0-9_]", "_", os.path.splitext(os.path.basename(name))[0])
                builder.register_sqlite_db(path, alias=alias)
            elif e in DUCKDB_EXTS:
                builder.register_duckdb_db(path)
            else:
                print(f"[sql_agent] skipped DB blob {name} (ext {e})")

        for uf in db_files:
            data = await uf.read()
            handle_db_blob(getattr(uf, "filename", "upload.db"), data)
        for name, data in url_db_bytes:
            handle_db_blob(name, data)

        # 2) Handle parquet/json uploads
        for f in parquet_json_files:
            fn = getattr(f, "filename", "upload")
            e  = _ext(fn)
            if e in TABULAR_EXTS:
                path = os.path.join(work_dir, os.path.basename(fn))
                _safe_write(path, await f.read())
                builder.register_tabular_file(path)

        # 2b) parquet/json URLs (stream via httpfs)
        for u in parquet_json_urls:
            builder.register_tabular_url(u)

        # 3) Apply user SQL scripts
        for sf in sql_files:
            builder.apply_user_sql(getattr(sf, "filename", "<upload.sql>"), await sf.read())
        for name, data in url_sql_bytes:
            builder.apply_user_sql(name, data)

        # 4) External URIs (optional; introspect + sample import)
        if external_uris:
            try:
                for uri in external_uris:
                    try:
                        eng = create_engine(uri)
                        with eng.connect() as cx:
                            rows = cx.execute(text("""
                                SELECT table_schema, table_name
                                FROM information_schema.tables
                                WHERE table_type IN ('BASE TABLE','VIEW')
                                ORDER BY 1,2
                            """)).fetchall()
                            max_tables = 10
                            for (schema, tname) in rows[:max_tables]:
                                fq = f"{schema}.{tname}" if schema and schema.lower() not in ("public","dbo") else tname
                                q  = f"SELECT * FROM {fq} LIMIT 200"
                                df = cx.execute(text(q)).mappings().all()
                                if not df:
                                    continue
                                pdf = pd.DataFrame(df)
                                safe_t = re.sub(r"[^A-Za-z0-9_]", "_", f"ext_{schema}_{tname}")
                                builder.con.register("df", pdf)
                                builder.con.execute(f"CREATE OR REPLACE TABLE {safe_t} AS SELECT * FROM df")
                                builder.con.unregister("df")
                    except Exception as e:
                        print(f"[sql_agent] external uri error {uri}: {e}")
            except Exception as e:
                print(f"[sql_agent] sqlalchemy not available or failed: {e}")

        # 5) Summarize for Master
        if return_format == "text":
            return builder.summarize(task_hint=task)          # existing behavior
        elif return_format == "json":
            return builder.summarize_json(task_hint=task)     # for SQL agent
        else:  # "both"
            return builder.summarize_json(task_hint=task), builder.summarize(task_hint=task)

    finally:
        builder.close()
