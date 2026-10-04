# Cold Compression MD v2.1.2

ColdMD는 ASE와 MACE 같은 DFT-trained potential을 이용해 등방 cold
compression/decompression stage sequence를 실행하고 분석하는 프로젝트입니다.
v2.1.2는 v2.1.1의 계산·분석 동작을 유지하며 Git 저장소를 단일 개발 위치로
사용합니다. v2.1 계열에서는 기존 run의 **원자 위치와 모멘텀을 함께** 물려받아
새 계산으로 갈라지는 `fork`를 지원합니다.

## v2.1 계열 핵심 기능

- `coldmd fork`: 부모 run의 `stage_structures/*.extxyz`에서 cell, PBC,
  positions, momenta를 상속하고 새 YAML로 독립 계산을 시작합니다.
- 모든 stage snapshot을 fork 정본인 `.extxyz`와 관찰용 `.cif`로 동시에 저장합니다.
  따라서 별도 `03-convert_extxyz_to_cif.py`는 필요하지 않습니다.
- NVT MSD CSV/plot은 첫 저장 프레임을 기준으로 하는 single-origin MSD이며 stage 전체
  시간을 표시합니다. Linear/log–log plot의 sampling window 이전 구간은 stage thermo
  plot과 같은 회색 음영으로 표시합니다. 확산계수는 별도의 multi-origin/block
  estimator로 계속 계산합니다.
- P–V stage mean 경로는 branch별 선이 아니라 YAML의 전체 stage 순서를 한 선으로
  연결합니다. branch는 marker 색만 결정하므로 마지막 compression과 첫
  decompression 지점도 연결됩니다. 누락된 stage 데이터는 선을 끊습니다.
- 전체 P–T path plot은 더 이상 생성하지 않습니다.
- 각 stage의 T–t/P–t 또는 T–t/V–t plot은 sampling tail뿐 아니라 stage 전체 시간을
  표시하며, 통계에서 제외한 앞 구간을 회색으로 표시합니다.
- launcher는 `launch_coldmd.sh` 하나만 사용합니다.

## Git 저장소와 계산 데이터

`coldMD/`가 소스, 테스트, 문서, 입력 CIF와 실행 설정 YAML의 관리 위치입니다.
실험별 YAML은 별도 파일로 저장하고 함께 커밋하십시오. 완료된 run의
`resolved-config.yaml`과 manifest는 해당 run의 실제 설정과 출처를 보존합니다.
기존 `03-coldcomp_v211/`에는 v2.1.1 계산 결과가 남아 있습니다.

launcher의 기본 실행 결과와 로그 경로는 저장소의 형제 디렉터리인
`../coldmd-runs/`와 `../coldmd-logs/`입니다. 모델 파일도 저장소 밖에 두고 YAML에서
경로와 SHA-256을 지정합니다. `.gitignore`는 실행 결과, lock, Python 설치·캐시
파일을 제외합니다. 필요하면 `RUN_BASE`와 `LOG_DIR`로 출력 위치를 바꿀 수 있습니다.

## 새 v2.1.2 환경 설치

기존 v2.1.1 환경의 PyTorch/MACE 구성을 활용하려면 새 환경으로 복제한 뒤, 그
환경 안의 구버전 ColdMD만 제거하고 Git 저장소에서 설치하십시오. 기존 환경은
v2.1.1 run의 strict resume용으로 유지합니다.

```bash
cd /data/Python_MLMD_DATA/LSK/coldMD

source /opt/miniforge3/etc/profile.d/conda.sh
conda create -n coldmd-v212 --clone coldmd-v211
conda activate coldmd-v212

python -m pip uninstall -y cold-compression-md
python -m pip install -e ".[mace,test]"
python -m pip check
coldmd --version
# coldmd 2.1.2
```

새로 만드는 환경이라면 `requirements-cpu.txt`, `requirements-cu126.txt`, 또는
`requirements-cu130.txt` 중 장치와 driver에 맞는 파일을 먼저 설치한 뒤
`python -m pip install -e ".[mace,test]"`를 실행합니다.

launcher는 다른 버전이 실수로 실행되는 것을 막기 위해 활성 환경의
`coldmd --version`이 정확히 `2.1.2`인지 확인합니다. 편의를 위한 alias 예시는 다음과
같습니다. 필요하면 해당 줄을 `~/.bashrc`에 직접 추가하십시오.

```bash
alias coldmd212='conda run --no-capture-output -n coldmd-v212 coldmd'
alias launch-coldmd212='bash /data/Python_MLMD_DATA/LSK/coldMD/launch_coldmd.sh'
```

launcher가 직접 환경을 활성화하므로 보통은 다음 설정만으로 충분합니다.

```bash
export CONDA_SH=/opt/miniforge3/etc/profile.d/conda.sh
export COLDMD_ENV=coldmd-v212
```

## 빠른 시작

```bash
cd /data/Python_MLMD_DATA/LSK/coldMD

# YAML/CIF/calculator 검증 + 100-step fixed-cell NVT smoke run
bash launch_coldmd.sh check stage_sequence_pilot.yaml

# preflight 통과 후 짧은 pilot run
bash launch_coldmd.sh new stage_sequence_pilot.yaml pilot
```

`check`는 production 계산을 시작하지 않습니다. `new`는 같은 preflight를 다시 통과한
경우에만 `nohup` background 계산을 시작합니다.

현재 `stage_sequence.yaml`은 15 GPa 부모 상태에서 fork하도록 작성된 설정입니다.
300 K에서 다음 압력 순서를 사용합니다.

```text
compression:    22.5 → 30 GPa
decompression:  22.5 → 15 → 10 → 5 → 0 GPa
```

각 압력점은 `npt_hold`와 `nvt_hold`의 쌍이며, 전체 14 stage입니다. 각 stage는
11 ps, sampling window는 마지막 10 ps입니다. 처음 1 ps만 통계에서 제외하므로
평형 도달 여부는 압력과 부피의 시간 추세로 별도로 확인해야 합니다. 새 CIF에서
0 GPa부터 시작하는 계산에는 초기 stage를 포함한 별도 YAML을 사용하십시오.

## launcher 명령

```text
./launch_coldmd.sh check CONFIG.yaml
./launch_coldmd.sh new CONFIG.yaml [run-label]
./launch_coldmd.sh fork CONFIG.yaml PARENT_RUN STATE.extxyz [run-label]
./launch_coldmd.sh resume RUN_DIRECTORY
```

주요 환경변수는 다음과 같습니다.

```bash
export COLDMD_ENV=coldmd-v212
export CPU_THREADS=24
export DRYRUN_NVT_STEPS=100
export RUN_BASE=/data/Python_MLMD_DATA/LSK/coldmd-runs
export LOG_DIR=/data/Python_MLMD_DATA/LSK/coldmd-logs
```

`new`와 `fork`는 기존 run directory를 덮어쓰지 않습니다. preflight 자료는
`coldmd-logs/preflight-*`, terminal output과 PID는 `coldmd-logs/{new,fork,resume}-*`에
남습니다.

## resume과 fork

두 기능은 목적과 보존 범위가 다릅니다.

| 항목 | `resume` | `fork` |
|---|---|---|
| 목적 | 중단된 동일 계산의 정확한 계속 | 기존 상태에서 다른 경로의 새 계산 |
| 설정 | 부모의 `resolved-config.yaml` 고정 | 사용자가 지정한 새 YAML |
| 입력 | 최신 exact checkpoint | 선택한 stage EXTXYZ snapshot |
| step/time | checkpoint 값을 계속 사용 | `0`/`0 fs`로 재시작 |
| 기준 부피 | 부모 기준을 계속 사용 | source volume을 새 `V0`로 사용 (`V/V0=1`) |
| RNG/integrator/safety history | checkpoint에서 복원 | 새 seed/새 integrator/새 safety history |
| 원자 mechanics | checkpoint 그대로 | cell, PBC, positions, momenta 상속 |
| 출력 | 기존 run에 append | 별도 child run |

### fork 실행

부모 run에서 원하는 snapshot을 먼저 선택합니다. `start`, `end`, `handoff` 모두
가능하지만, 다음 stage가 실제로 상속한 대표 상태에서 갈라지려면 `handoff`를
사용하는 것이 일반적입니다.

```bash
PARENT=/absolute/path/to/coldmd-runs/parent-run
STATE=stage-007-load_15GPa_nvt-handoff-step-000000088000.extxyz

# 계산을 시작하지 않는 fork 전용 검증
coldmd fork stage_sequence.yaml \
  --from-run "$PARENT" \
  --state "$STATE" \
  --check-only --json

# 검증 + background child run
bash launch_coldmd.sh fork stage_sequence.yaml "$PARENT" "$STATE" branch-30gpa
```

`--state`에는 절대경로 또는 `PARENT/stage_structures/` 바로 아래의 파일명을 사용할 수
있습니다. 다음 조건을 모두 확인한 뒤에만 fork가 허용됩니다.

- canonical stage snapshot 이름과 `.extxyz` 형식
- 부모 `run-manifest.json`과 `resolved-config.yaml`의 일치
- 새 YAML의 CIF template과 동일한 atom count, atomic numbers, atom order
- 명시적으로 저장된 유한한 momenta 배열(0 K의 all-zero 배열도 그대로 보존)
- 유한한 cell/positions, 3-D PBC, 최소 원자거리 등의 구조 안전성
- source state에서 새 calculator의 energy/forces/stress probe
- 새 YAML에 지정된 모든 stage integrator의 생성 가능 여부

CIF는 원자 모멘텀을 보존하지 않으므로 **fork 입력으로 사용할 수 없습니다**. 관찰과
외부 도구 입력에는 자동 생성된 CIF를 사용하고, fork에는 같은 stem의 EXTXYZ를
사용하십시오.

부모 run은 읽기 전용입니다. child의 `run-manifest.json`, checkpoint provenance,
`coldmd.log`에는 parent run/config/manifest/source 경로와 SHA-256, source stage/event,
부모 step/time, source volume/temperature가 기록됩니다. 이 값들은 lineage일 뿐 child의
step/time 기준으로 이어지지 않습니다.

v2.0 또는 v2.1.1 run의 canonical stage EXTXYZ로 v2.1.2 child를 fork할 수 있습니다.
반면 strict `resume`은 source/runtime fingerprint를 대조하므로 기존 계산은 당시
버전의 설치로 resume해야 합니다.

### strict resume

```bash
bash launch_coldmd.sh resume /absolute/path/to/coldmd-runs/<run-directory>
```

strict resume에는 local, content-hashed MACE `model_path`가 필요합니다. 압력 경로,
timestep, thermostat/barostat, seed 또는 calculator를 바꿀 목적이라면 resume이 아니라
fork를 사용하십시오.

## stage handoff와 구조 파일

각 stage의 sampling window에서 complete snapshot을 수집하고 평균 체적에 가장 가까운
실제 snapshot 하나를 선택합니다. cell, positions, momenta를 함께 다음 stage로 넘기며
artificial affine rescaling은 하지 않습니다.

`output.write_stage_structures: true`이면 각 경계마다 같은 stem의 두 파일이 생깁니다.

| 파일 | 역할 |
|---|---|
| `*-start-*.extxyz` / `.cif` | stage가 실제로 시작한 상태 |
| `*-end-*.extxyz` / `.cif` | 시간순 마지막 MD 상태 |
| `*-handoff-*.extxyz` / `.cif` | sampling-window 대표 상태 |

EXTXYZ에는 cell, PBC, positions, momenta가 있으므로 fork 정본입니다. CIF는 구조 관찰용
동반 파일이며 momenta가 없습니다. `coldmd.log`의 stage event에는 두 경로가 모두
기록됩니다.

## 분석

```bash
RUN_DIR=/absolute/path/to/coldmd-runs/<run-directory>

coldmd analyze "$RUN_DIR"
coldmd analyze "$RUN_DIR" --stage load_50GPa_nvt --stage unload_50GPa_nvt
coldmd analyze "$RUN_DIR" --all-stages
coldmd analyze "$RUN_DIR" --force
```

주요 v2.1 계열 분석 기준은 다음과 같습니다.

- `pv_hysteresis_mean_sd.png`는 유한한 stage mean을 `stage_order` 순서로 연결합니다.
  compression/decompression/recovery는 선을 나누는 기준이 아니라 marker 색상입니다.
- 전체 P–T path plot인 `pt_path_mean_sd.png`는 생성하지 않습니다.
- NVT stage의 `msd_full_stage.csv`, 선형 plot, log–log plot은 첫 저장 프레임에 대한
  single-origin displacement이며 마지막 저장 프레임까지 표시합니다. CSV 시간 열은
  `elapsed_time_ps`입니다. 두 plot 모두 sampling window 이전 구간을 회색으로
  음영 처리합니다.
- `diffusion_estimates.csv`의 확산계수/불확도/판정은 통계성을 위해 별도의
  multi-origin late-time fit과 block estimator를 사용합니다.
- 각 stage의 `sampling_thermo.csv`는 역사적인 파일명을 유지하지만 stage 전체 row를
  담습니다. `stage_time_ps`와 `in_sampling_window` 열로 통계 구간을 구분합니다.
- `sampling_thermo_T_P.png`(NVT) 또는 `sampling_thermo_T_V.png`(NPT/ramp)는 stage 전체
  시간을 표시하고, sampling 통계에서 제외한 앞 구간을 회색으로 음영 처리합니다.
- changing-cell NPT/volume-ramp에는 affine cell motion이 섞이므로 MSD를 만들지
  않습니다.

## 주요 출력

| 경로 | 내용 |
|---|---|
| `run-manifest.json` | 상태, config/model/input provenance, fork lineage |
| `resolved-config.yaml` | 실제 실행에 고정된 YAML |
| `thermo.csv` | step/time/stage/T/P/V/force/distance/stress |
| `trajectory-*.traj` | segmented ASE trajectory와 metadata sidecar |
| `stage_structures/` | EXTXYZ fork state + CIF companion |
| `checkpoints/checkpoint.json` + `.npz` | 최신 정상 exact checkpoint |
| `checkpoints/emergency.*` | 오류 순간 forensic state; 일반 resume용 아님 |
| `analysis/` | 원시 데이터를 바꾸지 않는 CSV/PNG/JSON/report |

## 주요 프로젝트 파일

| 파일/디렉터리 | 역할 |
|---|---|
| `stage_sequence.yaml` | 15 GPa 부모 상태에서 이어지는 14-stage fork 경로 |
| `stage_sequence_pilot.yaml` | 짧은 stage-transition 안정성 시험 |
| `CS_relaxed_NBO_BO.cif` | 기본 260-atom 입력 identity template |
| `launch_coldmd.sh` | check/new/fork/resume 단일 launcher |
| `src/coldmd/` | config, dynamics, engine, storage, analysis, reporting, CLI |
| `tests/` | unit/integration/launcher/fork contracts |
| `docs/V2_WORKFLOW.md` | stage, fork, 분석 기준의 보충 설명 |

## 검증과 주의사항

```bash
python -m pip check
MPLCONFIGDIR=/tmp/coldmd-mpl PYTHONPATH=src:tests \
  python -m unittest discover -s tests -v
bash -n launch_coldmd.sh
```

`dryrun`은 readiness smoke test이지 production wall-time estimator나 과학적 평형성
검증이 아닙니다. 새 pressure increment, NPT length, `barostat_tau_fs`, timestep,
potential은 짧은 pilot으로 확인하십시오. 이 workflow는 등방 cell을 전제로 하며 shock,
shear, uniaxial strain 또는 piston simulation이 아닙니다. 압축/회수 구조에 대한 MACE의
정확도는 실제로 방문한 조성, 압력, 밀도, coordination, 짧은 결합거리에서 별도의 DFT
reference로 검증해야 합니다.
