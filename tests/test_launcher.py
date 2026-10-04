from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "launch_coldmd.sh"
BASH = os.environ.get("COLDMD_TEST_BASH") or shutil.which("bash")


def shell_path(path: Path) -> str:
    resolved = path.resolve()
    return resolved.as_posix() if os.name == "nt" else str(resolved)


def shell_search_path(path: Path) -> str:
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    relative = resolved.relative_to(resolved.anchor).as_posix()
    return f"/{drive}/{relative}"


@unittest.skipUnless(BASH, "bash is required for launcher integration tests")
class LauncherIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT if os.name == "nt" else None
        )
        self.root = Path(self.temporary.name)
        self.bin_dir = self.root / "fake bin"
        self.bin_dir.mkdir()
        self.calls_path = self.root / "coldmd-calls.jsonl"
        self.calls_path.touch()

        self.conda_sh = self.root / "conda.sh"
        self.conda_sh.write_text(
            textwrap.dedent(
                """\
                conda() {
                  if [[ "${1:-}" == "activate" ]]; then
                    return 0
                  fi
                  echo "unexpected fake conda invocation: $*" >&2
                  return 2
                }
                """
            ),
            encoding="utf-8",
            newline="\n",
        )

        self.fake_coldmd = self.bin_dir / "coldmd"
        self.fake_coldmd.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python
                import json
                import os
                from pathlib import Path
                import sys

                args = sys.argv[1:]
                calls = Path(os.environ["FAKE_CALLS"])
                with calls.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(args) + "\\n")

                if args == ["--version"]:
                    print("coldmd 2.1.2")
                    raise SystemExit(0)

                command = args[0] if args else ""
                if command == "validate":
                    if os.environ.get("FAKE_VALIDATE_FAIL") == "1":
                        print("synthetic validation failure", file=sys.stderr)
                        raise SystemExit(2)
                    print(json.dumps({
                        "status": "valid",
                        "calculator_checked": True,
                        "protocol_design": "stage_explicit",
                        "total_steps": 10000,
                        "strict_resume_available": False,
                    }))
                    raise SystemExit(0)

                if command == "dryrun":
                    steps = int(args[args.index("--nvt-steps") + 1])
                    output_dir = Path(args[args.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "status": "failed" if os.environ.get("FAKE_DRYRUN_INVALID") == "1" else "passed",
                        "dryrun_nvt_steps": steps,
                        "dryrun_state": "input_cell_fixed_nvt",
                        "production_total_steps": 10000,
                    }
                    (output_dir / "dryrun.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                    print(json.dumps(payload))
                    raise SystemExit(0)

                if command == "run":
                    output_dir = Path(args[args.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "fake-run-called").write_text("ok\\n", encoding="utf-8")
                    raise SystemExit(int(os.environ.get("FAKE_RUN_EXIT", "0")))

                if command == "fork":
                    if "--check-only" in args:
                        print(json.dumps({
                            "status": "valid",
                            "fork_ready": True,
                            "child_origin": {
                                "global_step": 0,
                                "time_fs": 0.0,
                                "volume_ratio_from_engine_start": 1.0,
                            },
                            "fork_provenance": {
                                "source_state": args[args.index("--state") + 1],
                                "source_state_sha256": "a" * 64,
                            },
                        }))
                        raise SystemExit(0)
                    output_dir = Path(args[args.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "fake-fork-called").write_text("ok\\n", encoding="utf-8")
                    raise SystemExit(int(os.environ.get("FAKE_FORK_EXIT", "0")))

                if command == "resume":
                    raise SystemExit(int(os.environ.get("FAKE_RESUME_EXIT", "0")))

                print(f"unexpected fake coldmd arguments: {args!r}", file=sys.stderr)
                raise SystemExit(2)
                """
            ),
            encoding="utf-8",
            newline="\n",
        )
        self.fake_coldmd.chmod(0o755)

        if os.name == "nt":
            helpers = {
                "realpath": """#!/usr/bin/env python
import sys
print(sys.argv[1])
""",
                "sleep": """#!/usr/bin/env python
import sys
import time
time.sleep(float(sys.argv[1]))
""",
                "tee": """#!/usr/bin/env python
from pathlib import Path
import sys
append = sys.argv[1] == "-a"
path = Path(sys.argv[2] if append else sys.argv[1])
data = sys.stdin.read()
with path.open("a" if append else "w", encoding="utf-8") as stream:
    stream.write(data)
sys.stdout.write(data)
""",
                "nohup": """#!/usr/bin/env sh
exec "$@"
""",
            }
            for name, source in helpers.items():
                helper = self.bin_dir / name
                helper.write_text(source, encoding="utf-8", newline="\n")
                helper.chmod(0o755)

        self.config = self.root / "config with spaces.yaml"
        self.original_config = "config_version: 1\n# test input remains unchanged\n"
        self.config.write_text(self.original_config, encoding="utf-8")

        self.run_base = self.root / "runs"
        self.log_dir = self.root / "logs"
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "CONDA_SH": shell_search_path(self.conda_sh),
                "COLDMD_ENV": "coldmd-test",
                "CPU_THREADS": "3",
                "PROJECT_DIR": "." if os.name == "nt" else str(PROJECT_ROOT),
                "RUN_BASE": "runs" if os.name == "nt" else str(self.run_base),
                "LOG_DIR": "logs" if os.name == "nt" else str(self.log_dir),
                "STARTUP_CHECK_SECONDS": "0.05",
                "FAKE_CALLS": shell_path(self.calls_path),
                "PATH": (
                    ":".join(
                        [
                            shell_search_path(self.bin_dir),
                            shell_search_path(Path(sys.executable).parent),
                            os.environ.get("COLDMD_TEST_COREUTILS", "/usr/bin"),
                        ]
                    )
                    if os.name == "nt"
                    else os.pathsep.join(
                        [
                            shell_path(self.bin_dir),
                            shell_path(Path(sys.executable).parent),
                            self.environment.get("PATH", ""),
                        ]
                    )
                ),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_launcher(
        self, *arguments: str, extra_environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = self.environment.copy()
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            [BASH, shell_search_path(LAUNCHER), *arguments],
            cwd=self.root,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )

    def calls(self) -> list[list[str]]:
        return [
            json.loads(line)
            for line in self.calls_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_check_uses_configured_dryrun_steps_without_rewriting_yaml(self) -> None:
        result = self.run_launcher(
            "check",
            shell_search_path(self.config),
            extra_environment={"DRYRUN_NVT_STEPS": "37"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.config.read_text(encoding="utf-8"), self.original_config)
        calls = self.calls()
        validate_call = next(call for call in calls if call[0] == "validate")
        self.assertEqual(Path(validate_call[1]).resolve(), self.config.resolve())
        self.assertEqual(validate_call[2:], ["--json"])
        dryrun_call = next(call for call in calls if call[0] == "dryrun")
        self.assertEqual(dryrun_call[dryrun_call.index("--nvt-steps") + 1], "37")
        self.assertFalse(any(call[0] in {"run", "resume"} for call in calls))
        self.assertIn("Dry run passed: 37 fixed-cell NVT steps", result.stdout)
        self.assertIn("WARNING: strict resume is unavailable", result.stdout)

        preflight_dirs = list(self.log_dir.glob("preflight-config_with_spaces-*"))
        self.assertEqual(len(preflight_dirs), 1)
        self.assertTrue((preflight_dirs[0] / "validate.json").is_file())
        self.assertTrue((preflight_dirs[0] / "dryrun" / "dryrun.json").is_file())
        self.assertTrue((preflight_dirs[0] / "preflight.log").is_file())

    def test_config_file_environment_fallback_uses_default_steps(self) -> None:
        result = self.run_launcher(
            "check", extra_environment={"CONFIG_FILE": shell_search_path(self.config)}
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        dryrun_call = next(call for call in self.calls() if call[0] == "dryrun")
        self.assertEqual(dryrun_call[dryrun_call.index("--nvt-steps") + 1], "100")

    def test_new_blocks_production_when_dryrun_inspection_fails(self) -> None:
        result = self.run_launcher(
            "new",
            shell_search_path(self.config),
            "blocked-run",
            extra_environment={"FAKE_DRYRUN_INVALID": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == "run" for call in self.calls()))
        self.assertIn("Preflight failed while inspecting dry-run JSON", result.stderr)
        self.assertEqual(list(self.run_base.glob("blocked-run-*")), [])

    def test_validation_failure_prevents_dryrun_and_run(self) -> None:
        result = self.run_launcher(
            "new",
            shell_search_path(self.config),
            "invalid-run",
            extra_environment={"FAKE_VALIDATE_FAIL": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        commands = [call[0] for call in self.calls() if call != ["--version"]]
        self.assertEqual(commands, ["validate"])
        self.assertIn("Preflight failed during validation", result.stderr)
        preflight_dirs = list(self.log_dir.glob("preflight-invalid-run-*"))
        self.assertEqual(len(preflight_dirs), 1)
        self.assertIn(
            "synthetic validation failure",
            (preflight_dirs[0] / "preflight.log").read_text(encoding="utf-8"),
        )

    def test_new_runs_preflight_before_background_run(self) -> None:
        result = self.run_launcher("new", shell_search_path(self.config), "accepted-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [call[0] for call in self.calls() if call != ["--version"]]
        self.assertEqual(commands, ["validate", "dryrun", "run"])
        run_call = next(call for call in self.calls() if call[0] == "run")
        output_dir = Path(run_call[run_call.index("--output-dir") + 1])
        if not output_dir.is_absolute():
            output_dir = self.root / output_dir
        self.assertTrue((output_dir / "fake-run-called").is_file())
        self.assertIn("Preflight completed successfully", result.stdout)
        self.assertIn("ColdMD completed before the background startup check", result.stdout)

        terminal_logs = list(self.log_dir.glob("new-accepted-run-*.out"))
        pid_files = list(self.log_dir.glob("new-accepted-run-*.pid"))
        self.assertEqual(len(terminal_logs), 1)
        self.assertEqual(len(pid_files), 1)
        self.assertIn("preflight_directory=", terminal_logs[0].read_text(encoding="utf-8"))

    def test_default_output_directories_are_siblings_of_checkout(self) -> None:
        checkout = self.root / "checkout"
        checkout.mkdir()
        launcher = checkout / "launch_coldmd.sh"
        shutil.copy2(LAUNCHER, launcher)
        environment = self.environment.copy()
        for name in ("PROJECT_DIR", "RUN_BASE", "LOG_DIR"):
            environment.pop(name, None)

        result = subprocess.run(
            [
                BASH,
                shell_search_path(launcher),
                "new",
                shell_search_path(self.config),
                "default-run",
            ],
            cwd=self.root,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(checkout.glob("coldmd-runs")), [])
        self.assertEqual(list(checkout.glob("coldmd-logs")), [])
        run_dirs = list((self.root / "coldmd-runs").glob("default-run-*"))
        self.assertEqual(len(run_dirs), 1)
        self.assertTrue((run_dirs[0] / "fake-run-called").is_file())
        self.assertEqual(
            len(list((self.root / "coldmd-logs").glob("preflight-default-run-*"))),
            1,
        )

    def test_resume_rejects_foundation_alias_before_starting_coldmd(self) -> None:
        run_dir = self.root / "alias run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "checkpoint.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (run_dir / "resolved-config.yaml").write_text(
            "calculator:\n  foundation_model: medium-mpa-0\n  model_path: null\n",
            encoding="utf-8",
        )

        result = self.run_launcher("resume", shell_search_path(run_dir))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.calls(), [["--version"]])
        self.assertIn("Strict resume requires a local, content-hashed MACE model", result.stderr)

    def test_fork_checks_snapshot_before_background_child_run(self) -> None:
        parent = self.root / "parent run"
        stage_dir = parent / "stage_structures"
        stage_dir.mkdir(parents=True)
        state = stage_dir / "stage-001-parent-handoff-step-000000000100.extxyz"
        state.write_text("synthetic\n", encoding="utf-8")

        result = self.run_launcher(
            "fork",
            shell_search_path(self.config),
            shell_search_path(parent),
            state.name,
            "branch-run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        fork_calls = [call for call in self.calls() if call and call[0] == "fork"]
        self.assertEqual(len(fork_calls), 2)
        self.assertIn("--check-only", fork_calls[0])
        self.assertNotIn("--check-only", fork_calls[1])
        output_dir = Path(fork_calls[1][fork_calls[1].index("--output-dir") + 1])
        if not output_dir.is_absolute():
            output_dir = self.root / output_dir
        self.assertTrue((output_dir / "fake-fork-called").is_file())
        self.assertIn("Fork preflight completed successfully", result.stdout)

    def test_invalid_startup_delay_is_rejected_before_background_run(self) -> None:
        run_dir = self.root / "local model run"
        (run_dir / "checkpoints").mkdir(parents=True)
        (run_dir / "checkpoints" / "checkpoint.json").write_text(
            "{}\n", encoding="utf-8"
        )
        (run_dir / "resolved-config.yaml").write_text(
            "calculator:\n  kind: mace\n  model_path: /models/local.model\n",
            encoding="utf-8",
        )

        result = self.run_launcher(
            "resume",
            shell_search_path(run_dir),
            extra_environment={"STARTUP_CHECK_SECONDS": "invalid"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == "resume" for call in self.calls()))
        self.assertIn("STARTUP_CHECK_SECONDS must be a non-negative", result.stderr)


if __name__ == "__main__":
    unittest.main()
