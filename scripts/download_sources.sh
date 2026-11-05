#!/usr/bin/env bash
# Download the Approved Documents (statutory guidance, gov.uk) used as the knowledge base.
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }
urls=(
  "https://assets.publishing.service.gov.uk/media/66f6c5eec71e42688b65ee11/ADM__V2_with_2024_amendments.pdf"
  "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/540330/BR_PDF_AD_M1_2015_with_2016_amendments_V3.pdf"
)
for u in "${urls[@]}"; do
  curl -fsSL -o "$(basename "$u")" "$u" && echo "downloaded $(basename "$u")"
done
