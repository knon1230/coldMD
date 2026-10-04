# ColdMD v2.1 workflow

## Stage-explicit protocol

ColdMD v2는 global mode 대신 각 stage의 실제 제어변수를 명시합니다.

| Stage kind | 제어 | 주 용도 |
|---|---|---|
| `nvt_hold` | complete cell 고정 + thermostat | 고정 밀도 sampling, MSD |
| `npt_hold` | isotropic MTK pressure control | 준정적 압력점 완화 |
| `volume_ramp` | prescribed isotropic volume | 속도 의존적 compression/decompression |

`branch`는 연구 경로(`compression`, `decompression`, `reference`, `recovery`)이고
`purpose`는 stage 역할입니다. 실행과 P–V 선 연결은 항상 YAML에 적힌 stage 순서를
따릅니다. branch는 plotting marker를 분류할 뿐 경로를 분리하지 않습니다.

모든 stage는 `sampling_window_ps`를 가집니다. 이 tail은 통계 구간인 동시에 다음
stage로 넘길 snapshot 후보군입니다.

```text
stage 전체 궤적
    ├─ 앞 구간: equilibration/transient (통계 제외)
    └─ sampling tail: complete snapshot 후보
             │
             ├─ 후보 평균 volume 계산
             └─ 평균에 가장 가까운 실제 snapshot 선택
                         │
                         └─ cell + positions + momenta handoff
```

## Fresh run, resume, fork

### Fresh run

입력 CIF로 원자 정체성과 초기 구조를 만들고 설정에 따라 속도를 한 번 초기화합니다.
step/time/V/V0는 0/0/1에서 시작합니다.

### Resume

동일 run의 exact checkpoint를 이어 씁니다. config, input CIF, local MACE bytes,
ColdMD source/runtime/accelerator fingerprint, RNG, cursor, mechanical state가 모두
호환되어야 합니다. NHC/MTK 내부 extended state를 완전하게 serialize할 수 없는 stage는
안전한 stage boundary checkpoint에서만 resume합니다.

### Fork

Fork는 continuation이 아니라 lineage가 기록된 새 계산입니다.

```text
parent stage EXTXYZ
   ├─ inherited: atomic order, cell, PBC, positions, momenta
   └─ provenance only: parent step/time/stage/event/hash

child YAML
   ├─ new protocol/calculator/seed/output
   ├─ new integrator and safety state
   └─ local origin: step=0, time=0, source volume=V0
```

Fork 입력은 parent run의 `stage_structures/` 바로 아래에 있는 canonical single-frame
EXTXYZ로 한정합니다. CIF는 momentum을 저장하지 않으므로 허용하지 않습니다. 새 YAML의
CIF는 atomic numbers/order 및 구조 metadata를 제공하는 identity template이며,
mechanical state는 EXTXYZ 값으로 교체됩니다.

```bash
coldmd fork CHILD.yaml \
  --from-run /path/to/parent-run \
  --state stage-...-handoff-step-............extxyz \
  --check-only --json

coldmd fork CHILD.yaml \
  --from-run /path/to/parent-run \
  --state stage-...-handoff-step-............extxyz \
  --output-dir /path/to/child-run
```

`--check-only`는 source hash/mechanics, template identity, calculator properties와 모든
child stage integrator 생성을 확인하되 MD나 output run을 시작하지 않습니다.

## Stage structures

start/end/handoff마다 동일 stem의 두 파일을 임시 파일로 먼저 쓴 뒤 publish합니다.

- `.extxyz`: cell/PBC/positions/momenta를 가진 fork-capable canonical state
- `.cif`: 시각화와 구조 교환용 companion; momentum 없음

Resume reconciliation은 checkpoint 이후의 EXTXYZ와 CIF를 함께 격리하므로 두 형식의
output prefix가 같은 checkpoint 경계를 따릅니다.

## Preflight and launcher

```bash
bash launch_coldmd.sh check stage_sequence_pilot.yaml
bash launch_coldmd.sh new stage_sequence_pilot.yaml pilot
bash launch_coldmd.sh fork stage_sequence.yaml /path/to/15GPa-parent STATE.extxyz branch-30gpa
bash launch_coldmd.sh resume /path/to/run
```

Fresh `check/new`는 calculator validation, 모든 integrator 생성, 입력 cell의 fixed-NVT
smoke trajectory를 수행합니다. Fork preflight는 실제 source mechanics에서 calculator와
integrator를 확인합니다. 어느 preflight도 wall time 또는 과학적 평형성을 추정하지
않습니다.

현재 `stage_sequence.yaml`은 15 GPa 부모 상태에서 시작하는 14-stage fork 경로입니다.
새 CIF에서 시작할 때에는 초기 압력점을 포함한 YAML을 사용합니다. launcher는
`coldmd-v212` 환경과 정확한 ColdMD 2.1.2 설치를 기본으로 요구하고, 결과와 로그를
기본적으로 Git 저장소의 형제 디렉터리인 `../coldmd-runs/`, `../coldmd-logs/`에 둡니다.

## Analysis criteria

### P–V path

Stage tail mean ± sample SD를 stage order로 배치합니다. 한 개의 neutral path line이 모든
연속 stage를 잇고, branch별 marker/errorbar가 그 위에 표시됩니다. 따라서
compression→decompression 전환도 연결됩니다. 중간 stage에 유한한 mean이 없으면 NaN
gap으로 선을 끊어 누락 구간을 건너뛰지 않습니다.

Global P–T path figure는 생성하지 않습니다. 온도는 stage summary CSV와 per-stage time
plot에서 확인합니다.

### Stage time plots

`sampling_thermo.csv`와 `sampling_thermo_T_{P,V}.png`는 이름의 이전 호환성을 유지하지만
내용은 full-stage입니다. `stage_time_ps`는 stage 첫 저장 row를 0으로 두고,
`in_sampling_window`가 통계 tail을 표시합니다. PNG는 통계 제외 구간을 회색으로
shade합니다.

### MSD and diffusion

NVT의 공개 MSD curve는 첫 저장 stage frame 하나를 원점으로 하는

\[
\mathrm{MSD}_\alpha(t)=\frac{1}{N_\alpha}\sum_{i\in\alpha}
|\mathbf r_i(t)-\mathbf r_i(0)|^2
\]

이며 full stored duration을 표시합니다. Periodic crossing은 fractional coordinate에서
unwrap한 뒤 고정 reference cell로 변환합니다.

Diffusion coefficient는 이 single-origin curve를 그대로 fit하지 않습니다. 통계성을
위해 별도 multi-origin MSD, late-time linear fit, block coefficient uncertainty를
사용하고 `diffusion_estimates.csv`에 기록합니다. NPT/ramp는 changing-cell affine motion
때문에 transport MSD 대상이 아닙니다. Linear/log–log MSD plot은 stage 전체를 유지하면서
설정된 sampling window 이전 구간을 stage thermo plot과 같은 회색 음영으로 표시합니다.
