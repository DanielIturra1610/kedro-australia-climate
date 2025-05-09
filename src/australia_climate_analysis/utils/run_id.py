import uuid
from datetime import datetime
import pandas as pd

def new_run() -> tuple[str, pd.DataFrame]:
    """Genera un identificador único de ejecución y prepara el DF para metadata.runs."""
    run_id = str(uuid.uuid4())

    run_row = pd.DataFrame(
        [{
            "run_id": run_id,
            "started_at": datetime.utcnow(),
            "description": "Ejecución automática desde pipeline Kedro"
        }]
    )
    return run_id, run_row
