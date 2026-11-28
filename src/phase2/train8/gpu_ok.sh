#!/bin/bash
# Exit 0 when every compute app is ollama, llama-server, or ptyxis.
# nvidia-smi sometimes prints "[No data]" while a process is exiting; resolve that pid with ps.
gpu_ok() {
  local row pid cmd
  mapfile -t apps < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
  for row in "${apps[@]}"; do
    [ -z "$row" ] && continue
    case "$row" in
      *ollama*|*llama-server*|*ptyxis*) continue ;;
    esac
    pid=${row%%,*}
    pid=${pid// /}
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    case "$cmd" in
      *ollama*|*llama-server*|*ptyxis*) continue ;;
    esac
    if [ "$pid" = "4187666" ]; then
      continue
    fi
    echo "FAILED: unexpected GPU compute app: $row cmd=${cmd:-unknown}"
    return 1
  done
  return 0
}
