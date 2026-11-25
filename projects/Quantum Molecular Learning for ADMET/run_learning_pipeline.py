"""Offline preparation and post-API binary ADMET learning entrypoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"python_classes")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Clean/split/fit train only and prepare all molecules; no API")
    inputs = prepare.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bbb-file", type=Path)
    inputs.add_argument("--data-csv", type=Path)
    prepare.add_argument("--endpoint", default="BBB_Martins")
    prepare.add_argument("--top-k", type=int, default=4)
    prepare.add_argument("--output-dir", type=Path, default=ROOT/"artifacts"/"learning")
    train = commands.add_parser("train", help="Fit on validated normalized results after API execution")
    train.add_argument("--preparation", type=Path, required=True)
    train.add_argument("--results", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, default=ROOT/"artifacts"/"trained_readout")
    predict = commands.add_parser("predict", help="Predict from validated decoded correlation records")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--records", type=Path, required=True,
                         help="JSON object with feature_schema and records from the result decoder")
    predict.add_argument("--output-dir", type=Path, default=ROOT/"artifacts"/"predictions")
    args = parser.parse_args(argv)
    from artifact_paths import validate_output_dir
    output = validate_output_dir(args.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(output/"cache"/"matplotlib"))
    from offline_guard import install
    install()
    from admet_submission_entry import _strict_json
    from runtime_input_protocol import atomic_write_json
    if args.command == "prepare":
        import pandas as pd
        from admet_curation import CurationConfig
        from data_benchmark import verify_bbb_file
        from learning_workflow import prepare_learning_workflow
        if args.bbb_file:
            import io
            path = verify_bbb_file(args.bbb_file)
            raw = path.read_bytes()
            import hashlib
            from data_benchmark import BBB_SHA256
            if hashlib.sha256(raw).hexdigest() != BBB_SHA256:
                raise ValueError("BBB source changed before parsing")
            table = pd.read_csv(io.BytesIO(raw), sep="\t")
        else:
            table = pd.read_csv(args.data_csv)
        directory, summary = prepare_learning_workflow(table, args.endpoint, output,
            curation_config=CurationConfig(top_k=args.top_k))
        print({"preparation":str(directory), "status":summary["status"],
               "prepared_molecules":summary["prepared_molecules"], "quantum_model_fitted":False})
    elif args.command == "train":
        from learning_workflow import train_post_api
        _, report = train_post_api(args.preparation, _strict_json(args.results.read_text()), output)
        print({"status":report["status"], "model_id":report["model_id"],
               "measurement_source":report["measurement_source"], "predictions":report["prediction_count"]})
    else:
        from post_api_ml import read_model, predict_correlation_model
        document = _strict_json(args.records.read_text())
        if not isinstance(document,dict) or set(document)!={"feature_schema","records"}:
            raise ValueError("Prediction requires the explicit decoded feature schema and records")
        predictions = predict_correlation_model(read_model(args.model), document["records"],
                                                feature_schema=document["feature_schema"])
        atomic_write_json(output/"predictions.json", predictions)
        print({"predictions":len(predictions), "output":str(output/"predictions.json")})


if __name__ == "__main__":
    main()
