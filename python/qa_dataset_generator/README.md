# QA Dataset Generator

## 개요
QA Dataset Generator는 입력된 문서에 대해 Q&A 데이터셋을 생성한다.

명령어 버전이 존재하며 수행 방법은 다음과 같다.

## 명령어 버전 (qa_dataset_generate)
1. 먼저 입력 파일을 profile에 설정된 내용대로 해당 위치에 저장해야 한다.

   (profile은 아래 Profile 변경 항목을 참조하기 바란다.) 

    profile 필드 중 `corpora/prefix` 항목을 본다.

    예를 들어 여기 저장 값이 `sample`이면 `{ROOT_DIR}/data/qa-dataset-generator/corpora`

    디렉토리에 `sample` 디렉토리를 만들어 그 아래 모든 파일을 복사한다.


2. 다음의 명령을 수행한다.
    ```shell
    $ ./qa_dataset_generate
    ```

3. profile 선택 메뉴가 나타나면 번호 1을 입력한다.
    ```shell

    [1] nrf-poc-2024-12-17

    Choose a profile for 'QA dataset generator' by number (q for exit):
    ```

4. 수행이 완료되면 결과 파일은 `${ROOT}/artifact/qa-dataset-generator` 디렉토리

   아래 `export` 디렉토리에 `csv` 파일로 저장된다.


### Profile 변경
`{ROOT_DIR}/python/qa_dataset_generator/profile` 디렉토리에 저장한다.

기존의 Profile을 사용해도 되고 새로 생성해도 된다.

각 항목을 설명하면 다음과 같다.
(반드시 변경 가능이라고 표시된 부분만 변경하기 바란다.)

```yaml
metadata:
  version: v1
  name: nrf-poc-qa-dataset-generator # suite 이름 (변경 가능)
  label: "2024-12-17" # suite 라벨 (변경 가능)
  task: qa-dataset-generator # 태스크 이름
  tags:
  - nrf-poc
  - qa-dataset-generator
  - "2024-12-17"

directive: # 변경 가능하나 디버깅 목적
  langsmith: false
  force: false
  truncate: false
  num_tries: 8
  recovery_time: 0.5

export: 
  table: # 결과를 테이블 형태로 출력
    enable: true # 문서 출력 여부 (변경 가능)
    stem:
    columns: # 변경하지 말것
    - 문제
    - 정답
    - file
    extension: # 필요에 따라 추가할 수 있음 (변경 가능)
    - .csv
    #- .xlsx
    #- .json

  document:
    enable: false # 변경 불가, 사용하지 않음
    keywords:
    extension:

models: # LLM 모델, 변경 가능
- [hcx-003]

corpora: # 입력 파일 설정
- mode:
  - aggregate # 변경하지 말 것
  prefix:
  - "sample" # 입력 문서 서브 디렉토리 (변경 가능)
  stem:
  - "*" # 입력 문서에 대한 와일드 카드 (변경 가능)
  extension: # 입력 문서 파일 포맷 (변경 가능)
  - .hwpx 
  #- .hwp
  #- .pdf 

cases: # 변경하지 말 것
- module:
  - vanila_*qa*_dataset_generator
  content_kr:
  - generate_prompt
  parameter:
  - chunk_size
  - chunk_overlap
  - num_datasets
  - max_tokens

parameter:
  chunk_size:
  - 500 # chunk size 변경 가능
  chunk_overlap:
  - 50 # chunk overlap 변경 가능
  max_tokens:
  - 600 # 최대 토큰 수 변경 가능 (특히 HCX-003일 경우 중요)
  num_datasets:
  - 100 # 생성할 데이터셋 개수

content_kr: # 변경하지 말 것 (변경하고 싶으면 해당 파일 내용을 변경할 것)
  generate_prompt:
  - pt-qagen-kr.yaml
```

