# Research Report Summarizer

## 개요
Research Report Summarizer는 연구보고서를 서식에 맞게 요약하고 이를
사람이 직접 작성한 요약서와 비교 평가한다.

2가지 버전으로 수행해 볼 수 있다. 첫번째 버전은 명령어 버전이고 다른 버전은
UI 버전이다. 수행 방법은 다음과 같다.


## 명령어 버전 (research_report_summarize)
1. 먼저 입력 파일을 profile에 설정된 내용대로 해당 위치에 저장해야 한다.

   (profile은 아래 Profile 변경 항목을 참조하기 바란다.)

    profile 필드 중 `corpora/prefix` 항목을 본다. 예를 들어 여기 저장 값이 `sample`이면

    `{ROOT_DIR}/data/research-report-summarizer/corpora` 디렉토리에 `sample`

    디렉토리를 만들고 그 아래 모든 파일을 복사한다.

2. 다음의 명령을 수행한다.
    ```shell
    $ ./research_report_summarize
    ```

3. profile 선택 메뉴가 나타나면 번호 1을 입력한다.
    ```shell

    [1] nrf-poc-2024-12-17
    [2] nrf-poc-otf

    Choose a profile for 'Research report summarizer' by number (q for exit):
    ```

4. 수행이 완료되면 결과 파일은 `${ROOT}/artifact/nrf-research-report-summarizer` 

   디렉토리 아래 `export` 디렉토리에 `pdf` 파일과 `docx` 파일이 저장된다.


## UI 버전 (research_report_summarize_st.py)
### 서버 프로그램 수행하기 (streamlit)

1. 다음의 명령을 수행한다.
    ```shell
    $ streamlit run research_report_summarize_st.py
    ```

2. 화면에 URL이 출력되면 `Network URL`을 웹브라우저에 넣는다. (포트번호 유의한다.)
   해당 프로그램이 웹 브라우저에서 수행된다.
    ```shell

    You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.96.28:8501
    ```

### 수행하기
1. 수행할 파일을 업로드한다. (`파일 관리` 항목)
   1. 먼저 `Enter folder for file upload:` 항목을 세팅한다. 기본값은 sample이다.
    서버측 디렉토리에 `sample`이란 서브 디렉토리가 생겨 업로드할 파일이 모두 여기에 저장된다. 이름을 다르게 세팅해도 된다. 

   2. 그 아래 `Browse files` 버튼을 누른다. 파일 창이 뜨면 파일을 선택한다.

      (동시에 여러 개 선택할 수 있다)

   3. 업로드가 완료되면 파일은 항상 서버에 남아 있기 때문에 이후 이 파일을 다시
      업로드할 필요가 없다.


   4. 아래 `Select folders to delete` 항목을 누르면 기존 생성된 서브 디렉토리를

      선택할 수 있다.

      그 아래 `Delete folders`를 누르면 선택한 서브 디렉토리가 모두 삭제된다.
      
      이후 다시 파일을 업로드할 수 있다.

2. 파라미터 세팅을 한다. (`파라미터 세팅` 항목)
    1. `Select LLM model` 선택 박스에서 LLM 모델을 선택한다. 

    2. `Select module` 선택 박스에서 알고리즘을 선택한다. (`stuff_summarizer` 를 적극 권한다.)

    3. `Select input folder`에서 입력할 서브 디렉토리를 선택한다. 이 입력 서브 디렉토리는 `파일 관리` 항목에서 파일 업로드를 통해 생성한 것이다.

3. 명령어를 수행한다.
    1. `Run` 버튼을 수행하면 요약이 시작된다.

    2. 요약이 완료되면 가장 아래의 파일 선택 메뉴에서 파일을 선택한다. 선택한 파일에 대한
    요약 결과가 웹 화면에 출력된다.

    3. `Generate documents` 버튼을 누르면 `pdf` 문서와 `docx` 문서가 생성된다. 생성이
    완료되면 `Download zip files` 버튼이 생긴다. 이 버튼을 누르면 생성된 모든 문서가
    1개의 zip 파일로 압축되어 로컬 머신에 다운로드 된다.

    4. `Clear Cache`를 누르면 기존에 생성되었던 모든 데이터가 다 사라지고 (입력 데이터는 예외) 다시 생성된다. 결과가 제대로 출력되지 않았을 경우 시도해 본다.

### Layout 변경
`{ROOT_DIR}/data/research-report-summarizer/content/report-layout.yaml` 파일을 참조한다.

`source_template`에서 `header` ~ `reference`의 값은 제목을 나타낸다.

제목을 구분하여 전체 보고서 문서를 섹션으로 분할한다.


`summary_template`은 요약 서식을 나타낸다. 

`abstract`, `outcome`, `expectation`, `keyword`의 4 부분으로 나뉜다.

각 부분은 `title`과 `sections`로 이루어져 있다. 

`title`은 각 부분의 제목이다.

`sections`는 요약이나 중심어 추출을 수행할 섹션을 가리킨다.

이 섹션은 앞서 `source_template`에 정의된 섹션이다.

예를 들어 `expectation` 부분은 `contribution`과 `planning` 섹션만을
추출하여 요약을 수행한다.

### Profile 변경
`{ROOT_DIR}/python/research_report_summarizer/profile` 디렉토리에 저장한다.

기존의 Profile을 사용해도 되고 새로 생성해도 된다.

단 `nrf-poc-oft.yaml`은 UI 버전으로 변경하지 말아야 한다.

각 항목을 설명하면 다음과 같다.
(반드시 변경 가능이라고 표시된 부분만 변경하기 바란다.)

```yaml
metadata:
  version: v1
  name: nrf-poc-research-report-summarizer  # suite 이름 (변경 가능)
  label: "2024-12-15" # suite 라벨 (변경 가능)
  task: research-report-summarizer # 태스크 이름
  tags:
  - nrf-poc
  - research-report-summarizer
  - "2024-12-15"

directive: # 변경 가능하나 디버깅 목적
  langsmith: false
  force: false
  truncate: false
  num_tries: 8
  recovery_time: 0.5

export:
  table: # 결과를 테이블 형태로 출력함 (변경 가능하나 디버깅 목적)
    enable: true
    stem:
    columns:
    extension: 
    #- .csv
    #- .json
    #- .xlsx

  document: # 문서 출력 (변경가능)
    enable: true # 문서 출력 여부 (변경 가능)
    keywords: # 파일 구분할 때 사용 (변경 가능)
    - institution
    - name
    extension: # 파일 포맷 
    - .pdf
    - .docx

models: # LLM 모델, 변경 가능
- [gpt-4o, gpt-4o]

corpora: # 입력 파일 설정
- mode:
  - aggregate # 변경하지 말것
  prefix:
  - sample # 입력 문서 서브 디렉토리 (변경 가능)
  stem:
  - "*" # 입력 문서에 대한 와일드 카드 (변경 가능)
  extension: # 입력 문서 파일 포맷 (변경 가능)
  - .hwpx
  - .hwp
  - .docx
  - .pdf

cases: # 변경 하지 말 것
- module:
  - stuff_summarizer
  content_kr:
  - extract_header_prompt
  - extract_summary_prompt
  - similarity_prompt
  - summarize_prompt
  - keyword_prompt
  parameter:
  - max_tokens
  - num_keywords

parameter:
  max_tokens: # 최대 토큰 (변경 가능)
  - 800
  num_keywords: # 중심어 추출할 때 중심어 개수 (변경 가능)
  - 5

content_kr: # 변경하지 말 것 (변경하고 싶으면 해당 파일 내용을 변경할 것)
  extract_header_prompt:
  - pt-xhead-kr.yaml
  extract_summary_prompt:
  - pt-xsum-kr.yaml
  similarity_prompt:
  - pt-sim-kr.yaml
  summarize_prompt:
  - pt-sum-kr.yaml
  keyword_prompt:
  - pt-kwd-kr.yaml
```