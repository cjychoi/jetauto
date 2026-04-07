#!/usr/bin/env python3
import json
import re
import os
from logger_config import setup_logger

logger = setup_logger(__name__, "util_remove_candidate_fields.log")

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, 'static/face_db.json')
output_file = os.path.join(script_dir, 'static/face_db.json')
backup_file = os.path.join(script_dir, 'static/face_db.json.backup')

logger.info(f"Loading {input_file}...")
with open(input_file, 'r') as f:
    data = json.load(f)

logger.info(f"Creating backup at {backup_file}...")
with open(backup_file, 'w') as f:
    json.dump(data, f)

logger.info("Removing candidate_xxx fields...")
removed_count = 0

if isinstance(data, dict) and 'embeddings' in data:
    if isinstance(data['embeddings'], dict):
        keys_to_remove = [key for key in data['embeddings'].keys() if re.match(r'^candidate_\d+$', key)]
        for key in keys_to_remove:
            del data['embeddings'][key]
            removed_count += 1

logger.info(f"Removed {removed_count} candidate_xxx fields from embeddings")

logger.info(f"Writing cleaned data to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

logger.info("Done!")
