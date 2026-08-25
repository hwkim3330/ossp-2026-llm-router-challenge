<!--
SPDX-FileCopyrightText: Copyright 2026 metamong
SPDX-License-Identifier: Apache-2.0
-->

# metamong router — OSSP 2026 SKT Efficient LLM Routing Challenge

접수번호 168 · 팀명 메타몽 · 지정과제(SK텔레콤)

프롬프트만 보고 `ax31-light`, `ax31`, `axk1-think` 중 하나를 고르되, 등급별
예산 한도를 넘지 않는 라우터입니다. 한도를 넘긴 등급은 0점이므로, 이 구현의
설계 중심은 정확도가 아니라 **예산 초과를 구조적으로 막는 것**입니다.
측정 근거는 [`FINDINGS.md`](FINDINGS.md)에 있습니다.

## 파일

| 경로 | 내용 |
| --- | --- |
| `router_run.py` | 컨테이너 진입점. 등급 하나를 받아 `submission.v1` JSON 하나를 원자적으로 씁니다. |
| `router_features.py` | 프롬프트 텍스트 추출과 손수 만든 수치 특징. |
| `train_artifact.py` | 공개 Train+Dev 로 모델을 적합해 `artifact/router.pkl` 을 만듭니다. |
| `artifact/router.pkl` | 이미지에 굽는 학습 결과물. |
| `Dockerfile` | `linux/arm64` 제출 이미지. |

## 동작

1. 프롬프트를 word 1–2gram + char\_wb 3–5gram TF-IDF 로 벡터화합니다.
2. 업그레이드 모델마다 **이득**(Ridge)과 **비용**(HistGradientBoosting 분위
   회귀)을 예측합니다.
3. 예산 상한은 전량 `ax31-light` 비용의 배수인데 컨테이너는 그 값을 받지
   못하므로, 경량 모델의 비용도 같은 방식으로 **낮은 분위(0.75)** 에서
   추정합니다. 과대 추정은 상한을 부풀려 실격 위험을 만들고 과소 추정은 점수만
   남기므로 한쪽으로만 틀리게 만듭니다.
4. 0.5 분위 비용으로 예측 이득/비용 순 그리디 구매를 채운 뒤, 같은 장바구니를
   **0.95 분위로 다시 계산**해 그 비관적 청구서도 들어갈 때까지 가장 효율이
   낮은 구매부터 버립니다. 상한의 3% 는 여유로 남깁니다.

## 재현

```console
# 공개 자료 준비 (저장소 루트에서)
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py

# 학습 결과물 재생성
python3 router/train_artifact.py --out router/artifact/router.pkl

# 이미지 빌드
docker buildx build --platform linux/arm64 --load \
  --file router/Dockerfile --tag ossp-router:local router/

# 공식 자원 한도로 세 등급 실행
PYTHONPATH=src python3 tools/check_runtime.py --image ossp-router:local \
  --report build/runtime-check.json
```

## 학습 결과물 명세

`SUBMISSION.md` 가 요구하는 항목입니다.

* **이름·용도** — `artifact/router.pkl`. TF-IDF 어휘 2종, 업그레이드 모델별
  이득 회귀(Ridge), 비용 분위 회귀(HistGradientBoosting), 경량 모델 비용 회귀.
* **공개 업스트림** — 없음. 기반 모델 없이 이 저장소의 공개 Train+Dev
  2,640 문항으로 처음부터 적합했습니다.
* **고정 리비전과 SHA-256** —
  `c9fa33cd5d05cea7f4b77d5e577ab8ae28819fb8e3de4f466e241f0c5fdac681`
  (8,771,602 바이트), 커밋 `fc843a2` 에서 재적합한 판입니다.
* **라이선스** — Apache-2.0. 학습에 쓴 자료의 출처와 고지는 저장소의
  [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) 와
  [`DATA_LICENSES.md`](../DATA_LICENSES.md) 를 따릅니다.
* **변환 없음** — 별도 포맷 변환 도구를 쓰지 않았고, `train_artifact.py` 의
  출력이 그대로 이미지에 들어갑니다.

이미지는 실행 중 아무것도 내려받지 않으며, 네트워크 없이 동작합니다.

## 기반 이미지와 의존성

참조 Dockerfile 은 Alpine 을 쓰지만 scikit-learn 과 scipy 에는 musl 휠이 없어
Debian slim 을 씁니다. `RUNTIME.md` 는 기반 이미지를 권고로 두고, 빌드 시점에
공개 의존성을 설치하는 것을 허용합니다.

| 구성요소 | 버전 | 라이선스 |
| --- | --- | --- |
| python (`python:3.12-slim-bookworm`) | 3.12.14 | PSF-2.0 |
| numpy | 2.5.1 | BSD-3-Clause |
| scipy | 1.17.0 | BSD-3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |
| joblib | 1.5.3 | BSD-3-Clause |
| threadpoolctl | 3.6.0 | BSD-3-Clause |

버전은 `artifact/router.pkl` 을 적합한 환경과 정확히 같게 고정했습니다.
scikit-learn 은 적합된 모델을 unpickle 하므로 버전이 어긋나면 조용히 예측이
달라질 수 있습니다. 빌드 단계에서 `-W error::UserWarning` 으로 아티팩트를
불러 보아, 버전이 어긋나면 평가가 아니라 빌드가 실패하도록 했습니다.

## 저장소 정책 검사에 대한 참고

`tests/test_repository_policy.py` 의
`test_public_tree_has_no_internal_paths_secrets_or_model_artifacts` 는 UTF-8 로
읽히지 않는 파일을 모두 `binary:` 로 보고하므로 `artifact/router.pkl` 에서
실패합니다. 이 검사는 과제 저장소를 공개할 때 쓰는 것이고, `SUBMISSION.md` 는
반대로 **학습한 분류기를 포함했다면 가중치를 공개 위치에 두라**고 요구합니다.
그래서 아티팩트를 그대로 커밋했습니다. 나머지 실패 2건과 오류 2건은
`router/` 없이 상류 커밋에서도 그대로 재현되는 기존 문제입니다.
