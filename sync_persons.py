#!/usr/bin/env python3
"""
Sync person names from face database to node_sets.json
This utility scans the static/face_data directory and updates the persons list in node_sets.json
"""

import os
import json
from logger_config import setup_logger

logger = setup_logger(__name__, "sync_persons.log")


def sync_persons_to_node_sets(
    face_data_dir: str = "static/face_data",
    node_sets_path: str = "node_sets.json"
):
    """
    Sync person names from face database directory to node_sets.json.
    
    Args:
        face_data_dir: Path to the face data directory containing person subdirectories
        node_sets_path: Path to the node_sets.json file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    face_data_path = os.path.join(script_dir, face_data_dir)
    node_sets_file = os.path.join(script_dir, node_sets_path)

    # Face crops live here; embeddings may exist only in face_db.json. Missing folder is normal.
    if not os.path.exists(face_data_path):
        try:
            os.makedirs(face_data_path, exist_ok=True)
            logger.info(
                "Created empty face data directory (no crops yet): %s",
                face_data_path,
            )
        except OSError as e:
            logger.warning("Could not create face data directory %s: %s", face_data_path, e)
            return False
        logger.info(
            "Skipping node_sets sync: no person subfolders under %s yet.",
            face_data_path,
        )
        return True

    if not os.path.exists(node_sets_file):
        logger.info(
            "node_sets.json not found at %s; skipping sync (face recognition still works).",
            node_sets_file,
        )
        return True
    
    try:
        person_names = []
        for item in sorted(os.listdir(face_data_path)):
            item_path = os.path.join(face_data_path, item)
            if os.path.isdir(item_path) and not item.startswith("candidate_"):
                person_names.append(item)
        
        logger.info(f"Found {len(person_names)} person directories: {person_names}")
        
        with open(node_sets_file, 'r') as f:
            node_sets = json.load(f)
        
        old_persons = node_sets.get('persons', [])
        node_sets['persons'] = person_names
        
        with open(node_sets_file, 'w') as f:
            json.dump(node_sets, f, indent=2)
            f.write('\n')
        
        logger.info(f"✅ Successfully synced persons to {node_sets_file}")
        logger.info(f"   Old: {len(old_persons)} persons")
        logger.info(f"   New: {len(person_names)} persons")
        
        if set(old_persons) != set(person_names):
            added = set(person_names) - set(old_persons)
            removed = set(old_persons) - set(person_names)
            if added:
                logger.info(f"   Added: {sorted(added)}")
            if removed:
                logger.info(f"   Removed: {sorted(removed)}")
        else:
            logger.info("   No changes detected")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to sync persons: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync person names from face database to node_sets.json")
    parser.add_argument(
        "--face-data-dir",
        default="static/face_data",
        help="Path to face data directory (default: static/face_data)"
    )
    parser.add_argument(
        "--node-sets",
        default="node_sets.json",
        help="Path to node_sets.json file (default: node_sets.json)"
    )
    
    args = parser.parse_args()
    
    success = sync_persons_to_node_sets(args.face_data_dir, args.node_sets)
    exit(0 if success else 1)
