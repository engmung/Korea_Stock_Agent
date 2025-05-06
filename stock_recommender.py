import os
import json
import logging
import asyncio
import re
import uuid
import time
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
1. 추천 종목명 - 반드시 실제 한국 주식시장에 존재하는 구체적인 종목명이어야 합니다
2. 각 종목별 추천 이유
3. 전체 포트폴리오 구성 논리

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
    }
  ],
  "portfolio_logic": "포트폴리오 구성 논리"
}

원래 텍스트에 없는 정보는 합리적으로 추정하되, 추정이 불가능한 경우 "미제공"으로 표시하세요.
오직 JSON 형식만 반환하고 다른 설명은 포함하지 마세요."""


class GeminiClient:
    """개선된 Gemini API 클라이언트 - response_schema를 활용한 JSON 응답 직접 생성"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash-preview-04-17"
        logger.debug("GeminiClient 초기화 완료")
    
    async def analyze_with_structured_output(self, system_prompt: str, analysis_data: str, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """에이전트 프롬프트로 종목 분석 수행하고 구조화된 JSON 응답 직접 반환"""
        request_id = f"req-{uuid.uuid4()}"
        log_prefix = f"[{worker_id or 'main'}][{request_id}]" 
        logger.info(f"{log_prefix} Gemini API 구조화 분석 요청 시작")
        start_time = time.time()
        
        # JSON 스키마 정의 - 티커 필드 제거
        response_schema = {
            "type": "object",
            "properties": {
                "recommended_stocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["name", "reasoning"]
                    }
                },
                "portfolio_logic": {"type": "string"}
            },
            "required": ["recommended_stocks", "portfolio_logic"]
        }
        
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=analysis_data)]
            )
        ]
        
        # 시스템 프롬프트 앞에 지시사항 추가
        enhanced_system_prompt = f"이전의 지시사항은 전부 잊어버리세요. {system_prompt}"
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=[types.Part.from_text(text=enhanced_system_prompt)]
        )
        
        try:
            logger.debug(f"{log_prefix} API 호출 시작: model={self.model_name}")
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=generate_content_config
            )
            api_time = time.time() - start_time
            logger.info(f"{log_prefix} API 호출 완료 (소요 시간: {api_time:.2f}초)")
            
            # JSON 응답 자동 파싱
            if hasattr(response, 'parsed'):
                logger.debug(f"{log_prefix} 자동 파싱된 응답 사용")
                result = response.parsed
            else:
                # Fallback: 텍스트에서 JSON 파싱
                logger.debug(f"{log_prefix} 텍스트 응답에서 JSON 파싱")
                result = json.loads(response.text)
            
            # 기본 필드 확인 및 보장
            if "recommended_stocks" not in result:
                result["recommended_stocks"] = []
            if "portfolio_logic" not in result:
                result["portfolio_logic"] = ""
                
            total_time = time.time() - start_time
            logger.info(f"{log_prefix} 구조화 분석 완료 (총 시간: {total_time:.2f}초, 추천 종목 수: {len(result['recommended_stocks'])})")
            return result
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"{log_prefix} Gemini API 호출 중 오류 발생 (경과 시간: {error_time:.2f}초): {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "recommended_stocks": [],
                "portfolio_logic": f"분석 중 오류 발생: {str(e)}"
            }

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
    max_stocks: int = 5,  # 사용되지 않지만 호환성 유지
    investment_period: int = 7,
    worker_id: str = None,
    gemini_api_manager = None
) -> Dict[str, Any]:
    """
    단일 API 호출로 종목 추천 수행:
    1. Schema를 사용하여 구조화된 JSON으로 직접 응답 받음
    
    Args:
        agent: 투자 에이전트 객체
        analyzed_reports: 분석된 보고서 데이터
        max_stocks: 추천할 최대 종목 수 (에이전트 프롬프트에서 처리되므로 무시됨)
        investment_period: 투자 기간 (일)
        worker_id: 워커 ID (병렬 처리용)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
        
    Returns:
        추천 종목 정보 딕셔너리
    """
    job_id = f"job-{uuid.uuid4()}"
    log_prefix = f"[{worker_id or 'main'}][{job_id}]"
    start_time = time.time()
    logger.info(f"{log_prefix} 종목 추천 프로세스 시작 - 에이전트: {agent.agent_name}")
    
    try:
        # API 키 확인 - worker_id가 있고 gemini_api_manager가 제공된 경우 사용
        api_key = None
        if gemini_api_manager and worker_id:
            api_key = await gemini_api_manager.get_api_key(worker_id)
            logger.debug(f"{log_prefix} gemini_api_manager에서 API 키 획득")
        else:
            api_key = GEMINI_API_KEY
            logger.debug(f"{log_prefix} 환경 변수에서 API 키 사용")
            
        if not api_key:
            logger.error(f"{log_prefix} GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
            return {"error": "GEMINI_API_KEY not set"}
        
        # 보고서 데이터 검증
        if not analyzed_reports or "reports" not in analyzed_reports:
            logger.warning(f"{log_prefix} 분석된 보고서 데이터가 없습니다.")
            return {"error": "No report data available"}
        
        reports_count = len(analyzed_reports.get("reports", []))
        logger.info(f"{log_prefix} 총 {reports_count}개 보고서 분석 예정")
        
        # 노션에서 에이전트 프롬프트 가져오기 시작
        prompt_fetch_start = time.time()
        from report_selector import get_agent_prompt
        agent_prompt = await get_agent_prompt(agent.page_id)
        prompt_fetch_time = time.time() - prompt_fetch_start
        
        if not agent_prompt:
            logger.warning(f"{log_prefix} 에이전트 프롬프트를 찾을 수 없어 기본값 사용 (조회 시간: {prompt_fetch_time:.2f}초)")
            # 기본 프롬프트 사용
            agent_prompt = f"""당신은 투자 전문가입니다. 주어진 데이터를 분석하여 최적의 투자 종목을 추천해주세요.
            {agent.agent_name} 에이전트로서, 각 종목에 대한 분석 및 추천 이유를 자세히 설명해주세요."""
        else:
            logger.debug(f"{log_prefix} 에이전트 프롬프트 조회 완료 (소요 시간: {prompt_fetch_time:.2f}초)")
        
        
        # 보고서 내용 포맷팅 시작
        content_fetch_start = time.time()
        reports_content = ""
        reports_fetched = 0
        
        for i, report in enumerate(analyzed_reports.get("reports", [])):
            reports_content += f"### 보고서 {i+1}: {report.get('title', '')}\n"
            reports_content += f"채널: {report.get('channel', '')}\n"
            reports_content += f"날짜: {report.get('published_date', '')}\n"
            
            # 보고서 내용 가져오기
            if "id" in report:
                try:
                    # API 관리자 사용 여부에 따라 다른 방식으로 호출
                    if gemini_api_manager and worker_id and "notion_api_manager" in globals():
                        from notion_utils import get_notion_page_content_with_manager
                        content = await get_notion_page_content_with_manager(report["id"], notion_api_manager)
                    else:
                        from notion_utils import get_notion_page_content
                        content = await get_notion_page_content(report["id"])
                    
                    reports_content += f"내용:\n{content}\n\n"
                    reports_fetched += 1
                except Exception as e:
                    logger.warning(f"{log_prefix} 보고서 {report.get('id')} 내용 조회 실패: {str(e)}")
                    reports_content += f"내용: 조회 실패\n\n"
        
        content_fetch_time = time.time() - content_fetch_start
        logger.info(f"{log_prefix} 보고서 내용 조회 완료: {reports_fetched}/{reports_count}개 (소요 시간: {content_fetch_time:.2f}초)")
        
        # 분석 프롬프트 생성
        analysis_prompt = f"""
당신은 투자 분석 및 추천 전문가입니다. 다음 투자 에이전트의 투자 철학과 전략에 맞게 제공된 보고서들을 분석하고 최적의 종목을 추천해주세요.

## 투자 에이전트 프롬프트
{agent_prompt}

## 분석할 보고서 목록 (총 {len(analyzed_reports.get("reports", []))}개)
{reports_content}

## 요청
1. 위 보고서들을 심층 분석하여 에이전트의 투자 철학에 맞는 종목을 식별해주세요.
2. 투자 기간은 {investment_period}일입니다.

중요: 종목명은 절대로 "관심 종목", "추천 종목", "주의 종목" 같은 카테고리 레이블이 아니라 반드시 실제 종목명(예: 삼성전자, SK하이닉스, NAVER, 카카오)을 사용해야 합니다.
"""
        
        # Gemini 클라이언트 생성 - API 관리자 사용 여부에 따라 방식 결정
        if gemini_api_manager and worker_id:
            gemini_client = GeminiClient(api_key=api_key)
            logger.debug(f"{log_prefix} API 관리자 키로 GeminiClient 생성")
        else:
            gemini_client = GeminiClient()
            logger.debug(f"{log_prefix} 환경변수 키로 GeminiClient 생성")
        
        # 단일 호출로 구조화된 결과 획득
        logger.info(f"{log_prefix} 구조화된 종목 분석 요청 시작")
        system_prompt = "당신은 투자 분석 및 추천 전문가입니다. 투자 에이전트의 철학에 맞는 최적의 종목을 추천해주세요."
        result = await gemini_client.analyze_with_structured_output(
            system_prompt=system_prompt,
            analysis_data=analysis_prompt,
            worker_id=worker_id
        )
        
        # 원본 분석 텍스트는 portfolio_logic으로 간주
        result["analysis_text"] = result.get("portfolio_logic", "")
        
        total_time = time.time() - start_time
        recommended_count = len(result.get("recommended_stocks", []))
        logger.info(f"{log_prefix} 종목 추천 완료: {recommended_count}개 종목 (총 소요 시간: {total_time:.2f}초)")
        return result
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"{log_prefix} 종목 추천 중 오류 발생 (경과 시간: {total_time:.2f}초): {str(e)}", exc_info=True)
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