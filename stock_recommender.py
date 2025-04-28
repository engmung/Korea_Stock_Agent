import os
import json
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Gemini API 키
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

class PromptManager:
    """노션 페이지에서 프롬프트를 관리하는 클래스"""
    
    @staticmethod
    async def get_analysis_prompt(page_id: str) -> Optional[str]:
        """에이전트 페이지에서 분석 프롬프트 추출"""
        from notion_utils import get_notion_page_content
        
        content = await get_notion_page_content(page_id)
        
        # '## 시스템 프롬프트' 섹션 이후의 내용 추출
        pattern = r'## 시스템 프롬프트\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def get_parsing_prompt() -> str:
        """파싱용 고정 프롬프트 반환"""
        return """당신은 투자 분석 텍스트를 구조화된 데이터로 변환하는 전문가입니다.
        
주어진 투자 분석 텍스트에서 다음 정보를 추출하여 정확히 지정된 JSON 형식으로 변환하세요:
1. 추천 종목명 (최대 5개) - 반드시 실제 한국 주식시장에 존재하는 구체적인 종목명이어야 합니다
2. 각 종목별 추천 이유
3. 예상 수익률 (숫자 또는 범위)
4. 위험도 (높음/중간/낮음)
5. 전체 포트폴리오 구성 논리

매우 중요한 규칙:
- 종목명은 절대로 "관심 종목", "추천 종목", "주의 종목" 같은 카테고리 레이블이 아니라 반드시 실제 종목명(예: 삼성전자, SK하이닉스, NAVER, 카카오)을 사용해야 합니다.
- 텍스트에서 명확하게 언급된 종목만 포함하세요.
- 종목명이 불분명하거나 일반적인 카테고리만 언급된 경우, 해당 종목은 포함하지 마세요.

응답은 다음 JSON 형식만 사용하세요:
{
  "recommended_stocks": [
    {
      "name": "종목명(실제 주식 종목명)",
      "reasoning": "추천 이유",
      "expected_return": "예상 수익률",
      "risk_level": "위험도"
    }
  ],
  "portfolio_logic": "포트폴리오 구성 논리"
}

원래 텍스트에 없는 정보는 합리적으로 추정하되, 추정이 불가능한 경우 "미제공"으로 표시하세요.
오직 JSON 형식만 반환하고 다른 설명은 포함하지 마세요."""


class GeminiClient:
    """개선된 Gemini API 클라이언트"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        # genai.configure() 사용 안 함
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash-preview-04-17"
        # gemini-2.5-pro-preview-03-25
        # gemini-2.5-flash-preview-04-17
    
    async def analyze_with_agent_prompt(self, system_prompt: str, analysis_data: str) -> str:
        """에이전트 프롬프트로 종목 분석 수행"""
        generation_config = {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,
            "response_mime_type": "text/plain",
        }
        
        # 수정된 Gemini 호출 방식
        logger.info("분석 LLM 호출 중...")
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=analysis_data)]
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[types.Part.from_text(text=system_prompt)]
        )
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=generate_content_config
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API 호출 중 오류: {str(e)}")
            return f"분석 중 오류 발생: {str(e)}"
    
    async def parse_analysis(self, analysis_result: str, parsing_prompt: str) -> Dict[str, Any]:
        """분석 결과를 구조화된 JSON으로 파싱"""
        # 수정된 Gemini 호출 방식
        logger.info("파싱 LLM 호출 중...")
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"다음 투자 분석 결과를 파싱해주세요:\n\n{analysis_result}")]
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,  # 파싱은 낮은 온도로 정확성 추구
            response_mime_type="text/plain",
            system_instruction=[types.Part.from_text(text=parsing_prompt)]
        )
        
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=generate_content_config
            )
            
            # 응답에서 JSON 추출
            parsed_text = response.text
            json_match = re.search(r'({.*})', parsed_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                logger.error("JSON 형식을 찾을 수 없습니다")
                return {"error": "Invalid response format"}
        except Exception as e:
            logger.error(f"파싱 중 오류: {str(e)}")
            return {"error": str(e)}

async def create_reports_summary(reports: List[Dict[str, Any]]) -> str:
    """보고서 데이터를 요약하여 문자열로 반환합니다."""
    if not reports:
        return "분석된 보고서가 없습니다."
    
    # 최대 5개 보고서만 요약
    summary_reports = reports[:5]
    
    summary = "최근 보고서 요약:\n\n"
    
    for i, report in enumerate(summary_reports):
        # 출처 정보 강조하여 표시
        summary += f"{i+1}. [{report['channel']}] {report['title']} (날짜: {report['published_date']})\n"
        
        if "stocks" in report and report["stocks"]:
            stock_names = [stock["name"] for stock in report["stocks"][:3]]
            summary += f"   - 주요 언급 종목: {', '.join(stock_names)}\n"
        
        summary += "\n"
    
    if len(reports) > 5:
        summary += f"그 외 {len(reports) - 5}개 보고서 정보는 종목 데이터에 통합되었습니다.\n"
    
    return summary

async def create_stocks_summary(stocks: List[Dict[str, Any]]) -> str:
    """종목 데이터를 요약하여 문자열로 반환합니다."""
    if not stocks:
        return "분석된 종목이 없습니다."
    
    # 최대 10개 종목만 요약
    summary_stocks = stocks[:10]
    
    summary = "종목 요약 (언급 빈도 및 최신 언급 순):\n\n"
    
    for i, stock in enumerate(summary_stocks):
        summary += f"{i+1}. {stock['name']}"
        
        if stock.get("code"):
            summary += f" ({stock['code']})"
        
        summary += f" - {stock['report_count']}회 언급, 최근 언급: {stock['recent_mention']}\n"
        
        # 출처 정보 추가 (가장 최근 언급된 보고서)
        if stock.get("reports") and len(stock["reports"]) > 0:
            latest_report = stock["reports"][0]
            summary += f"   - 최근 출처: [{latest_report.get('channel', '')}] {latest_report.get('title', '')}\n"
        
        if stock.get("recommendation"):
            summary += f"   - 추천 강도: {stock['recommendation']}\n"
        
        if stock.get("investment_horizon"):
            summary += f"   - 투자 기간: {stock['investment_horizon']}\n"
        
        if stock.get("reasons") and len(stock["reasons"]) > 0:
            reasons = stock["reasons"][:3]  # 최대 3개 이유만 표시
            summary += f"   - 주요 언급 이유: {' / '.join(reasons)}\n"
        
        summary += "\n"
    
    if len(stocks) > 10:
        summary += f"그 외 {len(stocks) - 10}개 종목이 분석되었습니다.\n"
    
    return summary

async def create_stocks_summary(stocks: List[Dict[str, Any]]) -> str:
    """종목 데이터를 요약하여 문자열로 반환합니다."""
    if not stocks:
        return "분석된 종목이 없습니다."
    
    # 최대 10개 종목만 요약
    summary_stocks = stocks[:10]
    
    summary = "종목 요약 (언급 빈도 및 최신 언급 순):\n\n"
    
    for i, stock in enumerate(summary_stocks):
        summary += f"{i+1}. {stock['name']}"
        
        if stock.get("code"):
            summary += f" ({stock['code']})"
        
        summary += f" - {stock['report_count']}회 언급, 최근 언급: {stock['recent_mention']}\n"
        
        if stock.get("recommendation"):
            summary += f"   - 추천 강도: {stock['recommendation']}\n"
        
        if stock.get("investment_horizon"):
            summary += f"   - 투자 기간: {stock['investment_horizon']}\n"
        
        if stock.get("reasons") and len(stock["reasons"]) > 0:
            reasons = stock["reasons"][:3]  # 최대 3개 이유만 표시
            summary += f"   - 주요 언급 이유: {' / '.join(reasons)}\n"
        
        summary += "\n"
    
    if len(stocks) > 10:
        summary += f"그 외 {len(stocks) - 10}개 종목이 분석되었습니다.\n"
    
    return summary


async def recommend_stocks(
    agent, 
    analyzed_reports: Dict[str, Any],
    max_stocks: int = 5,
    investment_period: int = 7,
    worker_id: str = None,
    gemini_api_manager = None
) -> Dict[str, Any]:
    """
    두 단계 프로세스로 종목 추천:
    1. 분석 LLM: 투자 에이전트 프롬프트로 분석
    2. 파싱 LLM: 결과를 구조화된 JSON으로 변환
    
    Args:
        agent: 투자 에이전트 객체
        analyzed_reports: 분석된 보고서 데이터
        max_stocks: 추천할 최대 종목 수
        investment_period: 투자 기간 (일)
        worker_id: 워커 ID (병렬 처리용)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
        
    Returns:
        추천 종목 정보 딕셔너리
    """
    # 로그 접두어 (워커 ID가 있으면 포함)
    log_prefix = f"[{worker_id}] " if worker_id else ""
    
    try:
        # API 키 확인 - worker_id가 있고 gemini_api_manager가 제공된 경우 사용
        api_key = None
        if gemini_api_manager and worker_id:
            api_key = await gemini_api_manager.get_api_key(worker_id)
        else:
            api_key = GEMINI_API_KEY
            
        if not api_key:
            logger.error(f"{log_prefix}GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
            return {"error": "GEMINI_API_KEY not set"}
        
        # 보고서 데이터 검증
        if not analyzed_reports or "reports" not in analyzed_reports:
            logger.warning(f"{log_prefix}분석된 보고서 데이터가 없습니다.")
            return {"error": "No report data available"}
        
        # 노션에서 에이전트 프롬프트 가져오기
        from report_selector import get_agent_prompt
        agent_prompt = await get_agent_prompt(agent.page_id)
        if not agent_prompt:
            logger.error(f"{log_prefix}에이전트 {agent.agent_name}의 프롬프트를 찾을 수 없습니다.")
            # 기본 프롬프트 사용
            agent_prompt = f"""당신은 투자 전문가입니다. 주어진 데이터를 분석하여 최적의 투자 종목을 추천해주세요.
            {agent.agent_name} 에이전트로서, 각 종목에 대한 분석 및 추천 이유를 자세히 설명해주세요."""
            logger.info(f"{log_prefix}기본 프롬프트로 대체합니다.")
        
        # 투자 기간 텍스트 변환
        investment_period_text = f"{investment_period}일"
        if investment_period <= 7:
            investment_horizon = "단기"
        elif investment_period <= 30:
            investment_horizon = "중기"
        else:
            investment_horizon = "장기"
        
        # 보고서 내용 포맷팅
        reports_content = ""
        for i, report in enumerate(analyzed_reports.get("reports", [])):
            reports_content += f"### 보고서 {i+1}: {report.get('title', '')}\n"
            reports_content += f"채널: {report.get('channel', '')}\n"
            reports_content += f"날짜: {report.get('published_date', '')}\n"
            
            # 보고서 내용 가져오기
            if "id" in report:
                # API 관리자 사용 여부에 따라 다른 방식으로 호출
                if gemini_api_manager and worker_id and "notion_api_manager" in globals():
                    from notion_utils import get_notion_page_content_with_manager
                    content = await get_notion_page_content_with_manager(report["id"], notion_api_manager)
                else:
                    from notion_utils import get_notion_page_content
                    content = await get_notion_page_content(report["id"])
                reports_content += f"내용:\n{content}\n\n"
        
        # 분석 프롬프트 생성
        analysis_prompt = f"""
당신은 투자 분석 및 추천 전문가입니다. 다음 투자 에이전트의 투자 철학과 전략에 맞게 제공된 보고서들을 분석하고 최적의 종목을 추천해주세요.

## 투자 에이전트 프롬프트
{agent_prompt}

## 분석할 보고서 목록 (총 {len(analyzed_reports.get("reports", []))}개)
{reports_content}

## 요청
1. 위 보고서들을 심층 분석하여 에이전트의 투자 철학에 맞는 종목을 식별해주세요.
2. 투자 기간은 {investment_period_text}({investment_horizon})입니다.
3. 총 {max_stocks}개의 종목을 추천해주세요.

각 종목에 대해 다음 정보를 제공해주세요:
1. 종목명 (실제 한국 주식시장 종목명)
2. 추천 이유 및 투자 논리
3. 위험도 평가 (높음/중간/낮음)

전체 포트폴리오 구성 논리도 함께 제시해주세요.
"""
        
        # Gemini 클라이언트 생성 - API 관리자 사용 여부에 따라 방식 결정
        if gemini_api_manager and worker_id:
            gemini_client = GeminiClient(api_key=api_key)
        else:
            gemini_client = GeminiClient()
        
        # 1단계: 분석 수행
        logger.info(f"{log_prefix}에이전트 '{agent.agent_name}'의 프롬프트로 분석 시작...")
        analysis_result = await gemini_client.analyze_with_agent_prompt(
            system_prompt="당신은 투자 분석 및 추천 전문가입니다. 투자 에이전트의 철학에 맞는 최적의 종목을 추천해주세요.",
            analysis_data=analysis_prompt
        )
        
        logger.info(f"{log_prefix}분석 완료, 결과 파싱 시작...")
        
        # 2단계: 분석 결과 파싱
        parsing_prompt = """
당신은 투자 분석 텍스트를 구조화된 데이터로 변환하는 전문가입니다.
        
주어진 투자 분석 텍스트에서 다음 정보를 추출하여 정확히 지정된 JSON 형식으로 변환하세요:
1. 추천 종목명 (최대 5개) - 반드시 실제 한국 주식시장에 존재하는 구체적인 종목명이어야 합니다
2. 각 종목별 추천 이유
3. 위험도 (높음/중간/낮음)
4. 전체 포트폴리오 구성 논리

매우 중요한 규칙:
- 종목명은 절대로 "관심 종목", "추천 종목", "주의 종목" 같은 카테고리 레이블이 아니라 반드시 실제 종목명(예: 삼성전자, SK하이닉스, NAVER, 카카오)을 사용해야 합니다.
- 텍스트에서 명확하게 언급된 종목만 포함하세요.

응답은 다음 JSON 형식만 사용하세요:
{
  "recommended_stocks": [
    {
      "name": "종목명(실제 주식 종목명)",
      "reasoning": "추천 이유",
      "risk_level": "위험도"
    }
  ],
  "portfolio_logic": "포트폴리오 구성 논리"
}

원래 텍스트에 없는 정보는 합리적으로 추정하되, 추정이 불가능한 경우 "미제공"으로 표시하세요.
오직 JSON 형식만 반환하고 다른 설명은 포함하지 마세요.
"""
        parsed_recommendation = await gemini_client.parse_analysis(analysis_result, parsing_prompt)
        
        # 원본 분석 결과도 포함
        parsed_recommendation["analysis_text"] = analysis_result
        
        # 추천 종목들의 티커 코드 매핑
        stock_tickers = {}
        
        # 추천된 종목이 있는 경우 종목 코드 검색
        if "recommended_stocks" in parsed_recommendation:
            from stock_searcher import StockSearcher
            
            # API 관리자 사용 여부에 따라 방식 결정
            if gemini_api_manager and worker_id:
                stock_searcher = StockSearcher(api_key=api_key)
            else:
                stock_searcher = StockSearcher(api_key=GEMINI_API_KEY)
            
            for stock in parsed_recommendation["recommended_stocks"]:
                stock_name = stock.get("name", "")
                if stock_name:
                    # 종목 코드 검색
                    search_result = await stock_searcher.extract_stock_codes(f"{stock_name} 주식 종목코드")
                    response_text = search_result.get("response", "")
                    
                    # 티커 코드 추출 시도
                    ticker_match = re.search(r'(\d{6})', response_text)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                        # 매핑에 추가
                        stock_tickers[ticker] = stock_name
                        # 추천 종목에 티커 정보 추가
                        stock["ticker"] = ticker
                    else:
                        # KRX 직접 검색 시도
                        try:
                            import FinanceDataReader as fdr
                            import pandas as pd
                            krx_listing = await asyncio.to_thread(fdr.StockListing, 'KRX')
                            # 이름 기반 검색
                            matches = krx_listing[krx_listing['Name'].str.contains(stock_name, case=False)]
                            if not matches.empty:
                                ticker = matches.iloc[0]['Symbol']
                                stock_name_exact = matches.iloc[0]['Name']
                                logger.info(f"{log_prefix}KRX 검색으로 종목 '{stock_name}'에 대한 코드 '{ticker}' 찾음")
                                stock_tickers[ticker] = stock_name_exact
                                stock["ticker"] = ticker
                                stock["name"] = stock_name_exact  # 정확한 종목명으로 업데이트
                        except Exception as e:
                            logger.info(f"{log_prefix}KRX 검색 중 오류, 건너뜀: {str(e)}")
        
        # 백테스팅에서 활용할 수 있도록 티커->종목명 매핑도 함께 저장
        parsed_recommendation["stock_tickers"] = stock_tickers
        
        logger.info(f"{log_prefix}종목 추천 완료: {len(parsed_recommendation.get('recommended_stocks', []))}개 종목")
        return parsed_recommendation
        
    except Exception as e:
        logger.error(f"{log_prefix if 'log_prefix' in locals() else ''}종목 추천 중 오류: {str(e)}")
        return {"error": str(e)}

def extract_json_from_text(text: str) -> str:
    """텍스트에서 JSON 부분만 추출합니다."""
    # ```json과 ``` 사이의 내용 추출 시도
    import re
    
    # 패턴 1: ```json과 ``` 사이의 내용
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    
    if match:
        return match.group(1).strip()
    
    # 패턴 2: 단순히 { 로 시작하고 } 로 끝나는 부분
    json_pattern2 = r'(\{[\s\S]*\})'
    match = re.search(json_pattern2, text)
    
    if match:
        return match.group(1).strip()
    
    return ""