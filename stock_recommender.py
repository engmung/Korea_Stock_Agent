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

class GeminiClient:
    """개선된 Gemini API 클라이언트 - response_schema를 활용한 JSON 응답 직접 생성"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash"
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
3. 최대한 면밀히 분석하고, 추천 이유를 아주 상세하게 작성해 주세요.

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