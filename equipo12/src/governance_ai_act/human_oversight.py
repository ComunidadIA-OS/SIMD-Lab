import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any

class HumanInTheLoopManager:
    def apply_human_corrections(self, original_df: pd.DataFrame, edited_df: pd.DataFrame) -> Dict[str, Any]:
        diffs = (original_df["importance_score"] != edited_df["importance_score"]).sum()
        new_global_score = round(float(edited_df["importance_score"].mean()), 1)
        by_source = edited_df.groupby("source")["importance_score"].mean().round(1).to_dict()

        return {
            "human_validated": int(diffs) > 0,
            "modified_rows_count": int(diffs),
            "recalculated_global_score": new_global_score,
            "recalculated_by_source": by_source,
            "timestamp_utc": datetime.now(timezone.utc).isoformat()
        }