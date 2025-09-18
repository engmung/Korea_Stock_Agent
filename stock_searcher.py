import os
import logging
import asyncio
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

import FinanceDataReader as fdr
import pandas as pd

# 환경 변수 로드
load_dotenv()

# 로깅 설정 - 간결하게 변경
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# StockSearcher 클래스 - 종목 검색 기능
class StockSearcher:
    """주식 정보 검색 클래스 - 종목 코드 및 기간 추출에 특화"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다")
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash"
    
    async def extract_stock_codes(self, text: str) -> Dict[str, Any]:
        """텍스트에서 주식 종목 코드를 추출합니다."""
        try:
            response_text = await self._query_gemini(text, "extract_stocks")
            
            # 응답 텍스트에서 종목코드 추출 (정규표현식 이용)
            import re
            code_match = re.search(r'(\d{6})', response_text)
            
            if code_match:
                stock_code = code_match.group(1)
                
                return {
                    "status": "success",
                    "prompt": text,
                    "response": response_text,
                    "stock_code": stock_code
                }
            else:
                return {
                    "status": "not_found",
                    "prompt": text,
                    "response": response_text
                }
        except Exception as e:
            logger.error(f"텍스트에서 종목 추출 중 오류 발생: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def extract_time_period(self, text: str) -> Dict[str, Any]:
        """텍스트에서 기간 정보를 추출합니다."""
        try:
            response_text = await self._query_gemini(text, "extract_period")
            response_data = json.loads(response_text)
            
            # 응답에서 시작일과 종료일 추출
            start_date = response_data.get("start_date")
            end_date = response_data.get("end_date")
            
            # 날짜 값 검증 및 기본값 설정
            today = datetime.now().strftime("%Y-%m-%d")
            one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            if not end_date or not self._is_valid_date(end_date):
                end_date = today
            
            if not start_date or not self._is_valid_date(start_date):
                start_date = one_month_ago
                
            # 날짜 순서 확인 (시작일이 종료일보다 늦으면 교체)
            if start_date > end_date:
                start_date, end_date = end_date, start_date
            
            return {
                "status": "success",
                "prompt": text,
                "start_date": start_date,
                "end_date": end_date
            }
        except Exception as e:
            logger.error(f"텍스트에서 기간 추출 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 기간 제공
            return {
                "status": "error",
                "error": str(e),
                "start_date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "end_date": datetime.now().strftime("%Y-%m-%d")
            }
    
    def _is_valid_date(self, date_str: str) -> bool:
        """날짜 문자열이 유효한지 확인합니다."""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    async def _query_gemini(self, prompt: str, query_type: str = "extract_stocks") -> str:
        """Gemini API에 쿼리를 보내고 종목코드나 기간을 추출합니다."""
        try:
            if query_type == "extract_stocks":
                # 종목 추출 시스템 지시사항
                system_instruction = """
                내가 주는 텍스트에서 언급된 한국주식 종목들의 코드를 찾아줘. 
                종목코드는 6자리 형식(예: 005930)으로 제공해줘.
                구글 검색을 사용해서 정확한 종목 코드를 찾아줘.
                영어나 숫자는 발음그대로 한국어로 바꿔 검색하면 나올거야.
                종목명과 종목코드만 간결하게 알려줘.
                """
                user_prompt = f"다음 주식 종목의 코드를 찾아줘: {prompt}"
            else:
                # 기간 추출 관련 기존 코드 유지...
                system_instruction = """
                내가 주는 텍스트에서 백테스팅할 기간 정보를 추출해줘.
                시작일(start_date)과 종료일(end_date)을 YYYY-MM-DD 형식으로 제공해.
                정확한 날짜가 없으면 상대적인 표현(예: '1년 전', '3개월 전', '2주 전')을 오늘 기준으로 계산해서 변환해줘.
                응답은 다음 JSON 형식으로 제공해줘: {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}
                종료일이 언급되지 않았으면 종료일은 오늘로 설정해줘.
                시작일이 언급되지 않았으면 기본적으로 종료일로부터 1개월 전으로 설정해줘.
                """
                user_prompt = f"다음 텍스트에서 백테스팅할 기간 정보를 추출해줘: {prompt}"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)]
                )
            ]
            
            # 구글 검색 도구만 사용 - JSON 응답 형식 제거
            generate_content_config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],  # 구글 검색 사용 설정
                temperature=0,  # 일관된 응답을 위해 온도 낮게 설정
                system_instruction=[types.Part.from_text(text=system_instruction)]
            )
            
            # 비동기로 API 호출 실행
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=generate_content_config
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 호출 중 오류: {str(e)}")
            # 오류 발생 시 대체 응답 반환
            if query_type == "extract_stocks":
                return f"종목코드 추출 중 오류가 발생했습니다: {str(e)}"
            else:
                return f"기간 추출 중 오류가 발생했습니다: {str(e)}"

    # 종목명으로 직접 종목 코드 검색하는 메서드 추가
    async def search_stocks_by_name(self, names: List[str]) -> List[Dict[str, str]]:
        """종목명으로 KRX 종목 목록에서 종목 코드 검색"""
        try:
            krx_listing = await asyncio.to_thread(fdr.StockListing, 'KRX')
            results = []
            
            for name in names:
                # 가장 유사한 종목 찾기
                matches = krx_listing[krx_listing['Name'].str.contains(name, case=False)]
                if not matches.empty:
                    ticker = matches.iloc[0]['Symbol']
                    matched_name = matches.iloc[0]['Name']
                    results.append({
                        "input_name": name,
                        "matched_name": matched_name,
                        "ticker": ticker
                    })
            
            return results
        except Exception as e:
            logger.error(f"종목명으로 코드 검색 중 오류: {str(e)}")
            return []