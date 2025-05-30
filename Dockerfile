FROM python:3.10-slim

WORKDIR /app

# 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 포트 설정
EXPOSE 8004

# 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]