#!/bin/bash
# Reset state between demo rehearsals.
#  - Wipe Chroma collections (memory)
#  - Truncate state files (/tmp/jarvis_*.txt) on box
#  - Clear in-process calendar bookings (require app restart)
set -u

echo "── Resetting Julia for clean demo ──"

brev exec jarvis-track5 'python3 -c "
import chromadb
c = chromadb.HttpClient(host=\"localhost\", port=8001)
for n in (\"episodic_memory\", \"semantic_memory\"):
    try: c.delete_collection(n); print(\"  wiped\", n)
    except Exception as e: print(\"  skip\", n)
"
rm -f /tmp/jarvis_state.txt /tmp/jarvis_question.txt /tmp/jarvis_result.txt
echo "  cleared state files"
' 2>&1 | tail -10

echo ""
echo "Optional (restart app to clear in-memory calendar + reload triage prompt):"
echo "  bash scripts/laptop/demo_restart.sh"
echo ""
echo "Done. Open http://localhost:7860 for the demo."
