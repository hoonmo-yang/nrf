# NRF POC

## 설치 환경
* UBUNTU 리눅스 (LTS 18.04 이상)
* Python 3 (버전 3.12 이상)

## 설치 과정
### Miniconda 설치
1. Miniconda 설치 스크립트를 다운로드 받는다.
```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. 다운로드 받은 스크립트를 수행한다.
```shell
$ bash ./Miniconda3-latest-Linux-x86_64.sh
```

3. 설치 과정에서 질문에 답변한다. 
    1. *Please, press ENTER to continue*: 엔터를 친다.
    2. 계속 스페이스를 입력한다.
    3. *Do you accept the lincese terms? [yes|no]*: yes를 입력하고 엔터를 친다.
    4. *Minconda3 will now be installed ...*: 엔터를 친다
    4. *You can undo this by running ...*: yes를 입력하고 엔터를 친다.

4. 아래의 메시지가 출력되면 성공적으로 설치 완료되었다.
```
Thank you for installing miniconda3!
```

5. conda 환경을 활성화시킨다. 쉘프롬프트에 `(base)`가 뜬 것을 확인한다.
```shell
$ source ~/.bashrc
(base) $
```

6. 내가 사용할 conda 환경을 생성한다. 아래 예의 nrf는 환경 이름이다. 다른 이름을
원하면 이름을 변경한다.
```shell
$ conda create -n nrf python=3.12
```

7. 내가 만든 conda 환경으로 변경한다.
```shell
$ conda activate nrf
```

**주의 사항**: 로그인할 때마다 `conda activate nrf`를 수행해야 한다.

그러지 않을 경우, 프로그램 수행이 제대로 되지 않는다.

로그인할 때마다 자동으로 활성화하는 방법은 `~/.bashrc`에 해당 명령어를 입력한다.
```.bashrc
...
conda activate nrf
```

### 소프트웨어 설치
1. 설치를 원하는 디렉토리로 이동한 후 아래의 명령어를 수행한다. 
```shell
$ cd "원하는 디렉토리"
$ git clone https://github.com/hoonmo-yang/nrf.git
```

2. `nrf` 디렉토리가 생성된 것을 확인하면 `nrf` 디렉토리로 이동한다.
이제부터 `nrf` 디렉토리를 `{BASE_DIR}`이라 명명한다.
```shell
$ cd nrf
```

3. Make 명령어를 수행한다.
```shell
$ make install
```

4. 아래 명령어를 수행한다.
```shell
$ source ~/.bashrc
```

이제 모든 설치가 완료 되었다.

### API-key activation
`{BASE_DIR}/cf/.env` 파일을 연다.

파일의 내용은 다음과 같다.
```
# OpenAI ChatGPT
OPENAI_API_KEY=<insert key value>

# Langsmith setup
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=<insert key value>
LANGCHAIN_PROJECT=<insert your project name>

# Naver HyperClovaX Langchain Support
NCP_CLOVASTUDIO_API_KEY=<insert key value>
NCP_APIGW_API_KEY=<insert key value>
NCP_CLOVASTUDIO_APP_EMBEDDING_ID=<insert key value>
NCP_CLOVASTUDIO_APP_ID=$NCP_CLOVASTUDIO_APP_EMBEDDING_ID
```

`<insert ...>`라고 되어 있는 부분에 해당 값을 넣으면 된다.

해당 값은 각 site에서 발급 받을 수 있다.

`OPENAI_API_KEY`는 유료이며 반드시 받아야 한다.

Langsmith 관련된 라이센스 키는 코드를 디버깅하기 위해서 필요하며
개발 용도가 아니면 없어도 무방하다.
Langsmith는 무료이며 가입 후 사이트에서 `LANGCHAIN_API_KEY`와 `LANGCHAIN_PROJECT` 값을 얻어야 한다.

Naver HyperClovaX도 동일하다. 다만 HCP-003 LLM 모듈을 사용하지 않을 것이면
해당 키는 없어도 된다.

### 소프트웨어 수행
1. 연구보고서 요약기를 수행하고 싶으면 아래 디렉토리로 이동하고 해당 디렉토리의 `README.md`를 참조한다.
```shell
$ cd python/research_report_summarizer
```

2. Q&A 데이터셋 생성기를 수행하고 싶으면 아래 디렉토리로 이동하고 해당 디렉토리의 `README.md`를 참조한다.
```shell
$ cd python/qa_dataset_generator
```