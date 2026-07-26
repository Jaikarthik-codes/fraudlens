#!/bin/bash
set -e

CSV="log.csv"
MD="learning-log.md"

# Start markdown file fresh with a header if it doesn't exist
if [ ! -f "$MD" ]; then
  echo "# Learning Log" > "$MD"
  echo "" >> "$MD"
fi

# Skip header row, read the rest
tail -n +2 "$CSV" | while IFS=',' read -r date course entry; do
  # Random hour between 18:00 and 22:59, random minute
  hour=$((18 + RANDOM % 5))
  minute=$((RANDOM % 60))
  time=$(printf "%02d:%02d:00" "$hour" "$minute")
  timestamp="${date}T${time}"

  echo "- **${date}** [${course}] ${entry}" >> "$MD"

  git add "$MD"
  GIT_AUTHOR_DATE="$timestamp" GIT_COMMITTER_DATE="$timestamp" \
    git commit -m "${date}: ${entry}" --quiet

  echo "Committed: $timestamp - $entry"
done

echo "Done. Review with 'git log', then run 'git push -u origin main'."
