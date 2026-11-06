"""Run engine operations from composed configurations and write current artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import sys

from .config.checks import ConfigError, render_reference
from .config.loading import dump_yaml, parse_assignment, set_path
from .config.schema import ROOT_TABLE, compose_config, parse_config


def _positive_number(text):
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive finite number") from error
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _seed(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser(prog="hwoslaps", description=__doc__)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="check composed configuration and its digest")
    simulate = commands.add_parser("simulate", help="simulate an injection or a smooth control")
    forecast = commands.add_parser("forecast", help="forecast the supplied subhalo masses")
    batch = commands.add_parser("batch", help="plan, run or inspect a resumable batch")
    batch_commands = batch.add_subparsers(dest="batch_command", required=True)
    batch_plan = batch_commands.add_parser("plan", help="validate every member and print the job plan")
    batch_run = batch_commands.add_parser("run", help="run selected missing batch jobs")
    batch_status = batch_commands.add_parser("status", help="inspect completion markers and failures")
    batch_plan.add_argument("spec", type=Path)
    batch_run.add_argument("spec", type=Path)
    batch_run.add_argument("-o", "--output-dir", type=Path, required=True)
    batch_run.add_argument("--fresh", action="store_true")
    batch_run.add_argument("--devices")
    batch_run.add_argument("--workers-per-device", type=_positive_integer)
    batch_run.add_argument("--select", metavar="GLOB")
    batch_run.add_argument("--verify", action="store_true")
    batch_run.add_argument("--require-single-revision", action="store_true")
    batch_status.add_argument("output_dir", type=Path)
    reference = commands.add_parser("reference", help="print the configuration reference")
    reference.add_argument("section", nargs="?")
    for command in (validate, simulate, forecast):
        command.add_argument("configs", nargs="+", metavar="CONFIG")
        command.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    validate.add_argument("--print", action="store_true", dest="print_config")
    for command in (simulate, forecast):
        command.add_argument("-o", "--output-dir", type=Path, required=True)
    noise = simulate.add_mutually_exclusive_group(required=True)
    noise.add_argument("--noise-seed", type=_seed)
    noise.add_argument("--expected", action="store_true")
    simulate.add_argument("--smooth", action="store_true")
    forecast.add_argument("--masses", type=_positive_number, nargs="+", required=True)
    forecast.add_argument("--engine", choices=("reference", "jax"), default="reference")
    forecast.add_argument("--reference-workers", type=_positive_integer, default=1)
    forecast.add_argument("--batch-size", type=_positive_integer, default=16)
    forecast.add_argument("--progress", action="store_true")
    return parser


def _load(args):
    mapping = compose_config(args.configs)
    for assignment in args.set:
        path, value = parse_assignment(assignment)
        mapping = set_path(mapping, path, value, create=True)
    return parse_config(mapping, base_dir=Path.cwd())


def _new_output_dir(path):
    destination = path.expanduser().resolve()
    try:
        destination.mkdir(parents=True)
    except FileExistsError as error:
        raise ConfigError("output_dir", f"{destination} already exists") from error
    return destination


def _configure_logging(level, path=None):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = [logging.StreamHandler()]
    handlers[0].setLevel(level)
    if path is not None:
        handlers.append(logging.FileHandler(path))
        handlers[-1].setLevel(logging.DEBUG)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    logging.captureWarnings(True)
    return root, handlers


def main(argv=None):
    command = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(command)
    if args.command == "batch":
        return _batch_main(args, parser)
    try:
        if args.command == "reference":
            from .analysis.nonlinear import CLASSIFICATION_TABLE
            from .batch.spec import BATCH_TABLE
            from .inference.settings import FIT_TABLE, REFINE_TABLE, SAMPLER_TABLE
            from .population.sampling import POPULATION_TABLE

            documents = [("", ROOT_TABLE), ("population", POPULATION_TABLE), ("batch", BATCH_TABLE),
                         ("fit", FIT_TABLE), ("sampler", SAMPLER_TABLE), ("refine", REFINE_TABLE),
                         ("classification", CLASSIFICATION_TABLE)]
            print(render_reference(documents, section=args.section))
            return 0
        config = _load(args)
        if args.command == "validate":
            if args.print_config:
                print(dump_yaml(config.to_mapping()), end="")
            else:
                print(f"{config.run_name}: valid, digest {config.digest()}")
            return 0
        if args.command == "forecast" and config.forecast is None:
            raise ConfigError("forecast", "required for forecasting")
        if args.command == "simulate" and not args.smooth and config.scene.injection is None:
            raise ConfigError("scene.injection", "no injection configured; pass --smooth to simulate the smooth scene")
        output = _new_output_dir(args.output_dir)
    except ConfigError as error:
        parser.error(str(error))
    root, handlers = _configure_logging(args.log_level, output / "run.log")
    try:
        from .artifacts import save_forecast, save_observation, write_json, write_yaml
        from .provenance import capture_provenance
        run_record = {"operation": args.command, "command": command,
                      "started_utc": datetime.now(timezone.utc).isoformat(),
                      "environment": capture_provenance(command=command)}
        if args.command == "simulate":
            from .scene.cosmology import Cosmology
            from .scene.subhalo import configured_injection
            from .simulation import simulate
            halo = None if args.smooth else configured_injection(config.scene, Cosmology(config.cosmology), seed=config.seed)
            observation = simulate(config, subhalo=halo, noise_seed=args.noise_seed)
            write_yaml(output / "effective_config.yaml", config.to_mapping())
            save_observation(observation, output / "observation.npz")
            config_digest = observation.config_digest
        else:
            from .fisher.api import Execution, forecast, prepare_forecast
            execution = Execution(args.engine, args.reference_workers, args.batch_size, args.progress)
            with prepare_forecast(config, execution=execution) as prepared:
                write_yaml(output / "effective_config.yaml", prepared.config.to_mapping())
                result = forecast(prepared, masses_msun=args.masses)
                save_forecast(result, output / "forecast.npz")
                config_digest = result.provenance["config_digest"]
        write_json(output / "provenance.json", {**run_record, "config_digest": config_digest})
        print(f"artifacts: {output}")
        return 0
    except KeyboardInterrupt:
        root.exception("operation interrupted")
        return 130
    except Exception:
        root.exception("operation failed")
        raise
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        logging.captureWarnings(False)


def _batch_main(args, parser):
    from collections import Counter
    from dataclasses import replace
    import json
    from .batch import (BatchConflict, BatchIncomplete, BatchLocked, load_batch_spec, open_batch,
                        plan_batch, run_batch)
    from .population import PopulationError
    try:
        if args.batch_command == "status":
            results = open_batch(args.output_dir)
            counts = Counter((job.kind, job.status) for job in results.jobs)
            print(json.dumps({"counts": {f"{kind}:{status}": count for (kind, status), count in sorted(counts.items())},
                              "failed": [{"job_id": job.job_id, "path": str(job.path)}
                                         for job in results.jobs if job.status == "failed"]}, sort_keys=True))
            return 0
        spec = load_batch_spec(args.spec)
        if args.batch_command == "plan":
            print(json.dumps(plan_batch(spec).to_mapping(), sort_keys=True))
            return 0
        updates = {}
        if args.devices is not None:
            if args.devices == "cpu":
                updates["devices"] = "cpu"
            else:
                try:
                    updates["devices"] = tuple(int(value) for value in args.devices.split(","))
                except ValueError as error:
                    raise ConfigError("execution.devices", "must be cpu or comma-separated non-negative indices") from error
        if args.workers_per_device is not None:
            updates["workers_per_device"] = args.workers_per_device
        execution = replace(spec.execution, **updates)
        root, handlers = _configure_logging(args.log_level)
        try:
            report = run_batch(spec, args.output_dir, resume=not args.fresh, execution=execution,
                               select=args.select, verify=args.verify, require_single_revision=args.require_single_revision)
            print(json.dumps(report.to_mapping(), sort_keys=True))
            return 0
        except (BatchIncomplete, BatchConflict, BatchLocked, ConfigError, PopulationError, KeyboardInterrupt):
            raise
        except Exception:
            destination = args.output_dir.expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            log = logging.FileHandler(destination / "run.log")
            root.addHandler(log)
            try:
                root.exception("batch operation failed")
            finally:
                root.removeHandler(log)
                log.close()
            raise
        finally:
            for handler in handlers:
                root.removeHandler(handler)
                handler.close()
            logging.captureWarnings(False)
    except (ConfigError, PopulationError, BatchConflict, BatchLocked) as error:
        parser.error(str(error))
    except BatchIncomplete as error:
        print(json.dumps(error.report.to_mapping(), sort_keys=True))
        return 3
    except KeyboardInterrupt:
        return 130
