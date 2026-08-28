# Handoff — Compact Torque / Raw-Data Analyzer

기존 `quantum-oscillation-analysis-dashboard`(src/App.tsx, 약 6,500줄)에서 **Raw Data 탭과
Torque 탭 기능만** 뽑아 독립 실행 프로그램으로 재작성한다. FFT/dHvA 주파수 분석, Landau fan,
Branch/Symmetry/Harmonics, Firebase, Gemini는 전부 버린다.

---

## 1. 입력 데이터 포맷 (NHMFL torque sweep)

- 확장자 `.csv` / `.dat` / `.txt`, 파일 **1개 = 자기장 sweep 1회 = 각도(BLM) 1개**.
- 헤더 블록이 앞에 붙는다: `Key: value` 형식 메타 라인들, `***** ...`, `End of Sequence History`.
  데이터 헤더 행은 `End of Sequence History` **다음의 첫 비어있지 않은 줄**. 그 마커가 없으면
  `offsets:`, `scale factors:`, `magnet:`, `trigger`, `this data`, `*****`, `date:`, `time:`
  키워드를 포함하지 않으면서 구분자로 2개 이상 컬럼이 나오는 첫 줄을 헤더로 잡는다.
- 파싱은 PapaParse: `{ header: true, dynamicTyping: true, skipEmptyLines: true, comments: '#' }`.
- 데이터 컬럼 예: `B`(또는 field/magnetic 포함), `26_Output1`, `26_Output2`, `temp`, `BLM position`.
  - **`_` 앞부분이 prefix**(예: `26`)이고, prefix 하나를 고르면 그 그룹 전체가 한 신호 세트다.
  - **Output1 = in-phase(=torque τ), Output2 = quadrature.** Output1이 있으면 Output2는 컬럼
    선택·피팅·플롯·export 어디에서도 쓰지 않는다. Output2만 있는 파일은 그대로 쓴다.
    (원본 `dropOutput2()` 규칙을 그대로 가져올 것)
  - 온도는 temp 계열 컬럼의 **첫 값** 하나만 쓴다.
- 각도(BLM) 결정 순서 — 원본 `resolveBlm()` 로직 유지:
  1) `readAngleFromTitle`(기본 on): 파일명에서 첫 숫자 (`26down.csv` → 26)
  2) `step` 모드: 파일 인덱스+1
  3) `metadata` 모드: `blm|position|angle|step` 포함 컬럼의 첫 값 → 없으면 메타데이터
  마지막에 `angle = blm / blmPerDegree + blmOffset` (둘 다 UI 입력).

---

## 2. 원본에서 **그대로 이식**할 코어 (src/App.tsx 줄 번호)

수식·수치 알고리즘은 새로 짜지 말고 아래를 복사해서 순수 모듈로 옮긴다.

| 대상 | 줄 | 역할 |
|---|---|---|
| `decimateMinMax` | 220 | 플롯 다운샘플(min/max 보존) |
| `buildOlsModel` / `olsSolve` / `olsR2` | 320 / 348 / 370 | QR 기반 OLS, 설계행렬 재사용 |
| `cosineDesign` | 386 | Σ cos/sin(nα) 설계행렬 |
| `sineFourierDesign` | 402 | τ(α)=ΣAₙsin(nα+φₙ) 설계행렬 |
| `ALL_HARMONICS`, `fitCosineSeries`, `buildCosineFitResult` | 418–463 | 각도 cos 피팅 |
| `lerpFast` | 464 | 정렬된 B배열에서 선형보간(범위 밖 클램프) |
| `fitPolynomialH` | 481 | τ(H)=aH+bH²+cH³ |
| `buildFourierResult` | 504 | β → {n, A, φ(deg)} |
| `dropOutput2` | 771 | Output2 배제 규칙 |
| `resolveBCol` / `resolveBlm` | 891 / 906 | B 컬럼·각도 결정 |
| `prefixes` / `targetColsFor` | 1194 / 1208 | prefix 감지, 선택→컬럼 매핑 |
| `fileNumericViews` | 1231 | **핵심 캐시**: 파일별 B 1회 파싱·정렬 + 컬럼별 Float64Array 캐시 |
| `fieldSliceData` | 1263 | 여러 B값에서 보간한 R vs 각도 |
| `rawPlotDisplayData` / `rawPlotFitData` | 1289 / 1307 | 단일 B slice(또는 Max B)에서 R vs 각도 |
| `cosFitAtSlice` / `cosFitCurve` | 1313 / 1319 | 그 slice의 cos 피팅 곡선 |
| `cosScanPrepared` | 1333 | 스캔용 per-file 뷰 |
| `fieldAngleGrid` | 1351 | **0.2 T 격자 × 각도** 값 행렬 (전방 포인터 1패스 보간) |
| `scanGrid` | 1403 | 각 필드스텝마다 OLS, 동일 각도조합은 분해 캐시 |
| `cosHarmonicScanData(+B2)` | 1441 / 1460 | Aₙ(B) 스캔 |
| `harmonicScanDiag` | 1467 | 결과가 빌 때 원인 문구 |
| `tauScaleFields` | 1495 | "2,5,10,15,20" 파싱 |
| `fieldSweepBase` / `fieldSweepByAngle` | 1511 / 1537 | τ(H), τ/H, 저필드 3차 피팅, H_sf, τ_sat |
| `fieldSweepPlot` | 1576 | fast/full 표시 밀도 |
| `fourierScanData` | 1586 | 필드별 Fourier 계수 |
| `scalingVerifData` | 1600 | τ/H² vs α collapse |
| 파일 로더 | 2027–2200 | 헤더 스킵 + PapaParse + 각도/B컬럼 자동 선택 |
| Raw 탭 UI | 3942–4471 | 아래 3절 |
| Torque 탭 UI | 4472–4858 | 아래 3절 |

> 6,500줄 전체를 읽지 말고 위 범위만 offset/limit으로 읽어라.

---

## 3. 구현할 화면

### Raw 탭
1. **컨트롤 바** — 컬럼/prefix 드롭다운, 각도(BLM) 토글 칩(All/None), CSV export 버튼들,
   활성 시리즈 30개 초과 시 성능 경고.
2. **Angular Distribution** — 선택한 B slice(또는 Max B)에서 `R vs 각도` 산점 + cos 피팅 곡선.
   harmonic 집합 n ∈ {1,2,3,4,6,12} 체크박스, 계수/R² 표시. 점 개수 < 1+2·(harmonic 수)면 피팅 생략.
3. **Field Sweep** — 활성 각도별 `R vs B` 라인 + 앞 100점 데이터 테이블.
4. **Peak Transition Field** — `[maxDerivMinB, maxDerivMaxB]` 구간에서 |dR/dB|를 계산(중점 기준
   전방차분) → window 7 이동평균 → 최대값 위치 B를 각도별로 찍는다 (원본 2400–2440줄).
5. **Angular Dependence at Fixed Field** — `fieldSliceValues`(기본 1,5,10,15,20,25,30 T)에서
   보간한 `R vs 각도` 다중 시리즈.
6. **Harmonic Amplitudes vs Field** — 0.2 T 스텝마다 cos 피팅해 얻은 Aₙ을 `vs B`와 `vs B²`로.
   비어 있으면 `harmonicScanDiag` 문구를 그대로 띄운다.

*(원본의 "Individual Oscillatory Curves (ΔR vs 1/B)"는 다항식 배경제거 파이프라인이 필요하므로
이번 버전에서는 제외한다.)*

### Torque 탭
1. **Field-Sweep Decomposition** — `τ(H) = a·H + b·H² + c·H³`를 `H ≤ H_sf limit`(기본 10 T)
   구간에 피팅. 고필드에서부터 내려오며 `|τ − fit| / max|τ| ≤ resid threshold`(기본 0.05)를
   처음 만족하는 H를 **H_sf(spin-flop)** 로 정의. `τ_sat` = H_sf 위 구간 상위 20%(최소 3점) 평균.
   출력: `τ/H vs H` 오버레이(각도별 색), `a(α)`, `b(α)`, `H_sf(α)`, `τ_sat(α)` 산점도.
2. **Angle-Sweep Fourier** — `τ(α) = Σ Aₙ sin(nα + φₙ)`, n = 1..n_max(기본 6)를 0.2 T마다 OLS.
   `Aₙ vs H`(n=2,4,6), `A₂/A₆ vs H`, R² 표시. 고필드 상위 10% 평균 A₆로
   **k₆ = A₆_sat / (6·V)** (V = 시료 부피 mm³ 입력). 각도 수 < 2·n_max면 안내 문구.
3. **Scaling Verification** — 입력한 필드 리스트(기본 "2,5,10,15,20")에서 `τ/H² vs α`(collapse
   테스트)와 `τ vs α`.

모든 차트는 CSV로 내보낼 수 있어야 한다.

---

## 4. 버리는 것

`src/firebase.ts`, 로그인/익명인증, Firestore·localStorage 설정 동기화, `@google/genai`,
FFT 일체(`fft.js`, `getWindowCached`, `linearInterpUniform`, `downsampleFFT`, `MiniFFT`,
peak detection, threshold UI), `processFileData`, Dashboard/Assemble/Landau/Branch/Harmonics/
Symmetry 탭, `complexMode`, `polyDegree`/spike/smoothing 등 FFT 전처리 파라미터.

---

## 5. 원본을 그대로 베끼면 안 되는 지점 (버그 유발)

- **`uniqueBlms`가 `allRawSeries`(= "Process All" 결과)에서 나온다.** 새 앱에는 Process 버튼이
  없으므로 `fileNumericViews`의 `blm`에서 직접 만들어야 한다. 안 그러면 각도 칩이 항상 비어 있다.
- 같은 이유로 `rawPlotData`, `maxDerivPlotData`도 Process All에서 채워진다 → 파일 로드 직후
  memo로 계산되게 바꾼다.
- `scansArmed` 게이팅은 탭 지연 로딩용이었다. 새 앱은 두 탭뿐이므로 제거하되, `fieldAngleGrid`와
  스캔들은 반드시 `useMemo`로 유지한다(무거움).
- `scanGrid`의 OLS 모델 캐시(동일 각도조합 → 분해 재사용)를 없애지 마라. 없애면 필드스텝마다
  QR 분해가 다시 돌아 수십 배 느려진다.
- `lerpFast`와 `fieldAngleGrid`의 보간은 **범위 밖 클램프** 동작이 같아야 한다(둘이 같은 물리량을
  다른 경로로 계산한다).
- `fileNumericViews`의 컬럼 캐시를 우회해 매번 `parseFloat(String(row[col]))` 하지 마라.

---

## 6. 기술 스택 / 구조

React 19 + TypeScript + Vite + Recharts + PapaParse + lucide-react + Tailwind v4.
원본은 전부 App.tsx 한 파일이지만 **이번엔 나눈다**:

```
src/
  lib/parse.ts     헤더 스킵·PapaParse·각도/B컬럼 결정·dropOutput2
  lib/fit.ts       OLS, cosineDesign, sineFourierDesign, fitPolynomialH, buildFourierResult
  lib/grid.ts      fileNumericViews, fieldAngleGrid, scanGrid, lerpFast, decimateMinMax
  lib/export.ts    CSV 다운로드
  components/RawTab.tsx, TorqueTab.tsx, charts/*
  App.tsx          파일 로드 + 탭 + 공유 상태
```

## 7. 수용 기준

- CSV/DAT 여러 개를 드롭 → 각도 자동 인식 → 컬럼(prefix) 선택 → **Process 버튼 없이** 6개 Raw
  차트와 3개 Torque 섹션이 바로 렌더된다.
- Output1/Output2가 같이 있는 파일에서 Output2는 어디에도 나타나지 않는다.
- 파일 20개 × 20,000행에서 컬럼/각도 전환이 1초 이내.
- 모든 차트에 CSV export 버튼.
- 다크 배경(#141414 / #E4E3E0 / accent #F27D26), monospace 라벨 — 원본 톤 유지.
