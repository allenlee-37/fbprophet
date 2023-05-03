# Description
- fb-prophet을 간단하게 활용할 수 있도록 만든 파이썬 프로그램
    - 엑셀파일을 정해진 형태로 정리해서 입력하면 바로 활용 가능
- facebook의 prophet 라이브러리를 활용해 시계열 예측 데이터를 출력함
- 세가지 데이터를 출력함
    1. 원본 예측 데이터: fb-prophet을 돌린 첫 결과 그대로 저장한 엑셀 파일
    2. 예측값과 예측 상한선, 하한선 엑셀 파일
    3. 연간 예측값, 예측 상한선, 하한선 엑셀 파일
## Environment
- M1 macOS Ventura 13.2.1(22D68)에서 만들어졌음
    -  이외의 requirements.txt로 필요 패키지 설치 가능
## Prerequisite
- 
## Files
- .venv: 사용된 가상환경
- /raw_data/{도시이름}: 한국관광데이터랩에서 다운받은 엑셀 파일을 이곳에 저장<br>폴더명을 도시이름으로 지정할 것
- /result: 예측 결과가 저장
- run.py: 실행 파일
- requirements.txt: 설치 파일 목록
## Usage
파이썬 파일 (run.py)를 실행하고, 도시명을 지정해줌
### Usage examples
    `python3 run.py -r <도시명(폴더명)>`