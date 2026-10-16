#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="$project_dir/build"
source_file="${TMPDIR:-/tmp}/embedding-explorer-vec1.c"
compat_header="${TMPDIR:-/tmp}/embedding-explorer-vec1-compat.h"
sqlite_archive="${TMPDIR:-/tmp}/sqlite-amalgamation-3510100.zip"
sqlite_headers_dir="${TMPDIR:-/tmp}/sqlite-amalgamation-3510100"

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v cc >/dev/null || { echo "a C compiler is required" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }

python_cmd="${PYTHON:-python}"
"$python_cmd" -c 'import pysqlite3' >/dev/null 2>&1 || {
    echo "pysqlite3-binary is not installed in this Python environment" >&2
    echo "Run: python -m pip install pysqlite3-binary" >&2
    exit 1
}

mkdir -p "$build_dir"
curl --fail --location --max-time 60 -sS \
    'https://sqlite.org/vec1/doc/tip/vec1.c' \
    | sed -n '/<blockquote><pre>/,/<\/pre><\/blockquote>/p' \
    | sed '1d;$d' \
    | sed -e 's/&quot;/"/g' -e 's/&amp;/\&/g' -e 's/&lt;/</g' \
          -e 's/&gt;/>/g' -e 's/&#39;/'"'"'/g' > "$source_file"

# Use headers from the same SQLite release as pysqlite3-binary. System headers
# are often older and can produce an ABI-mismatched virtual-table module.
curl --fail --location --max-time 60 -sS \
    'https://sqlite.org/2025/sqlite-amalgamation-3510100.zip' \
    -o "$sqlite_archive"
rm -rf "$sqlite_headers_dir"
unzip -q "$sqlite_archive" -d "${TMPDIR:-/tmp}"

# pysqlite3 bundles a newer SQLite runtime than some system sqlite3.h files.
# These declarations let the official Vec1 source compile against that runtime.
cat > "$compat_header" <<'EOF'
#include <sqlite3.h>
#ifndef SQLITE_INDEX_CONSTRAINT_LIMIT
#define SQLITE_INDEX_CONSTRAINT_LIMIT 73
#endif
int sqlite3_vtab_in(sqlite3_index_info*, int, int);
int sqlite3_vtab_in_first(sqlite3_value*, sqlite3_value**);
int sqlite3_vtab_in_next(sqlite3_value*, sqlite3_value**);
EOF

pysqlite_so=$(find "$(dirname "$(readlink -f "$(command -v "$python_cmd")")")/../lib" \
    -path '*/site-packages/pysqlite3/_sqlite3*.so' -type f -print -quit 2>/dev/null || true)
if [[ -z "$pysqlite_so" ]]; then
    pysqlite_so=$("$python_cmd" -c 'import pathlib, pysqlite3; print(next(pathlib.Path(pysqlite3.__file__).parent.glob("_sqlite3*.so")))')
fi

cc -O3 -DNDEBUG -mavx2 -mfma -I"$sqlite_headers_dir" -include "$compat_header" \
    "$source_file" -shared -fPIC \
    -Wl,-rpath,"$(dirname "$pysqlite_so")" \
    -Wl,--no-as-needed -L"$(dirname "$pysqlite_so")" \
    -Wl,-l:"$(basename "$pysqlite_so")" \
    -o "$build_dir/vec1.so"

VEC1_EXTENSION="$build_dir/vec1.so" "$python_cmd" - <<'PY'
import os
import pysqlite3 as sqlite3

connection = sqlite3.connect(':memory:')
connection.enable_load_extension(True)
connection.load_extension(os.environ['VEC1_EXTENSION'][:-3])
connection.enable_load_extension(False)
print(f"SQLite {sqlite3.sqlite_version}; Vec1 {connection.execute('SELECT vec1_info()').fetchone()[0]}")
PY
