/* ===== Recent Successful Runs View ===== */

const RECENT_SUCCESS_COLUMNS = [
  { key: "timestamp", label: "Timestamp", type: "time" },
  { key: "dataset", label: "Dataset", type: "text" },
  { key: "preset", label: "Preset", type: "text" },
  { key: "hostname", label: "Host", type: "text" },
  { key: "gpu_type", label: "GPU", type: "text" },
  { key: "num_gpus", label: "GPUs", type: "number", numeric: true },
  { key: "git_commit", label: "Commit", type: "text" },
  { key: "pages", label: "Pages", type: "number", numeric: true },
  { key: "files", label: "Files", type: "number", numeric: true },
  { key: "pages_per_sec", label: "Pages / sec", type: "number", numeric: true },
  { key: "ingest_secs", label: "Ingest (s)", type: "number", numeric: true },
  { key: "recall_1", label: "Recall@1", type: "number", numeric: true },
  { key: "recall_5", label: "Recall@5", type: "number", numeric: true },
  { key: "recall_10", label: "Recall@10", type: "number", numeric: true },
];

function recentTimestampValue(timestamp) {
  if (!timestamp) return null;
  if (/^\d{8}_\d{6}_UTC$/.test(timestamp)) {
    return Date.UTC(
      Number(timestamp.slice(0, 4)),
      Number(timestamp.slice(4, 6)) - 1,
      Number(timestamp.slice(6, 8)),
      Number(timestamp.slice(9, 11)),
      Number(timestamp.slice(11, 13)),
      Number(timestamp.slice(13, 15)),
    );
  }
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : parsed;
}

function compareRecentRuns(a, b, column, direction) {
  let left = a[column.key];
  let right = b[column.key];

  if (column.type === "time") {
    left = recentTimestampValue(left);
    right = recentTimestampValue(right);
  }

  const leftMissing = left === null || left === undefined || left === "";
  const rightMissing = right === null || right === undefined || right === "";
  if (leftMissing || rightMissing) {
    if (leftMissing && rightMissing) return 0;
    return leftMissing ? 1 : -1;
  }

  let result;
  if (column.type === "number" || column.type === "time") {
    result = Number(left) - Number(right);
  } else {
    result = String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
  }
  return direction === "asc" ? result : -result;
}

function RecentSuccessesView({ data, loading, error, onRefresh, onSelectRun, githubRepoUrl }) {
  const [sortKey, setSortKey] = useState("timestamp");
  const [sortDirection, setSortDirection] = useState("desc");
  const runs = data?.runs || [];

  const sortedRuns = useMemo(() => {
    const column = RECENT_SUCCESS_COLUMNS.find(c => c.key === sortKey) || RECENT_SUCCESS_COLUMNS[0];
    return [...runs].sort((a, b) => compareRecentRuns(a, b, column, sortDirection));
  }, [runs, sortKey, sortDirection]);

  const pg = usePagination(sortedRuns, 25);

  const changeSort = (key) => {
    if (key === sortKey) {
      setSortDirection(current => current === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDirection(key === "timestamp" ? "desc" : "asc");
    }
  };

  const windowLabel = data?.window_start && data?.window_end
    ? `${new Date(data.window_start).toLocaleString()} – ${new Date(data.window_end).toLocaleString()}`
    : "Rolling 24-hour UTC window";

  const renderCell = (run, column) => {
    const value = run[column.key];
    if (column.key === "timestamp") return fmtTs(value);
    if (column.key === "git_commit") return <CommitLink sha={value} repoUrl={githubRepoUrl} />;
    if (column.key === "pages" || column.key === "files" || column.key === "num_gpus") return fmt(value, 0);
    if (column.key.startsWith("recall_")) return fmt(value, 3);
    if (column.type === "number") return fmt(value);
    return value || "—";
  };

  return (
    <>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"16px",gap:"12px",flexWrap:"wrap"}}>
        <div>
          <div style={{fontSize:"14px",fontWeight:600,color:"#fff"}}>Successful runs from the last 24 hours</div>
          <div style={{fontSize:"12px",color:"var(--nv-text-dim)",marginTop:"4px"}}>{windowLabel}</div>
        </div>
        <button className="btn btn-secondary" onClick={onRefresh} disabled={loading}>
          <IconRefresh /> Refresh
        </button>
      </div>

      <div className="card">
        <div style={{overflowX:"auto"}}>
          <table className="runs-table">
            <thead>
              <tr>
                {RECENT_SUCCESS_COLUMNS.map(column => (
                  <th key={column.key}
                    onClick={() => changeSort(column.key)}
                    aria-sort={sortKey === column.key ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}
                    style={{textAlign:column.numeric?"right":"left",cursor:"pointer",whiteSpace:"nowrap",userSelect:"none"}}
                    title={`Sort by ${column.label}`}>
                    {column.label}
                    <span style={{display:"inline-block",width:"12px",marginLeft:"4px",color:sortKey===column.key?"var(--nv-green)":"var(--nv-text-dim)"}}>
                      {sortKey === column.key ? (sortDirection === "asc" ? "▲" : "▼") : ""}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading && runs.length === 0 ? (
                <tr><td colSpan={RECENT_SUCCESS_COLUMNS.length} style={{textAlign:"center",padding:"60px",color:"var(--nv-text-muted)"}}>
                  <div className="spinner spinner-lg" style={{margin:"0 auto 12px"}}></div>
                  <div>Loading recent successful runs…</div>
                </td></tr>
              ) : error ? (
                <tr><td colSpan={RECENT_SUCCESS_COLUMNS.length} style={{textAlign:"center",padding:"48px",color:"#ff5050"}}>
                  <div style={{fontSize:"14px",fontWeight:600,marginBottom:"8px"}}>Could not load recent successful runs</div>
                  <div style={{fontSize:"12px",color:"var(--nv-text-muted)",marginBottom:"16px"}}>{error}</div>
                  <button className="btn btn-secondary" onClick={onRefresh}><IconRefresh /> Retry</button>
                </td></tr>
              ) : sortedRuns.length === 0 ? (
                <tr><td colSpan={RECENT_SUCCESS_COLUMNS.length} style={{textAlign:"center",padding:"60px",color:"var(--nv-text-muted)"}}>
                  <div style={{fontSize:"15px",marginBottom:"8px"}}>No successful runs in the last 24 hours</div>
                  <div style={{fontSize:"12px",color:"var(--nv-text-dim)"}}>New successful runs will appear here automatically.</div>
                </td></tr>
              ) : pg.pageData.map(run => (
                <tr key={run.id} onClick={() => onSelectRun(run.id)}>
                  {RECENT_SUCCESS_COLUMNS.map(column => (
                    <td key={column.key} style={{
                      textAlign:column.numeric?"right":"left",
                      whiteSpace:["timestamp","hostname","gpu_type"].includes(column.key)?"nowrap":undefined,
                      color:column.key==="recall_5"?"var(--nv-green)":column.key==="dataset"?"#fff":undefined,
                      fontWeight:column.key==="dataset"||column.key==="pages_per_sec"||column.key==="recall_5"?600:undefined,
                    }}>
                      {renderCell(run, column)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!error && (
          <Pagination page={pg.page} totalPages={pg.totalPages} totalItems={pg.totalItems}
            pageSize={pg.pageSize} onPageChange={pg.setPage} onPageSizeChange={pg.setPageSize} />
        )}
      </div>
    </>
  );
}
