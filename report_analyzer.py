import os
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Notion DB ID 환경 변수
NOTION_SCRIPT_DB_ID = os.environ.get("NOTION_SCRIPT_DB_ID")

class Report:
    """분석 보고서 클래스"""
    def __init__(self, page_id: str, title: str, channel: str, content: str, url: str, published_date: str):
        self.page_id = page_id
        self.title = title
        self.channel = channel
        self.content = content  # 마크다운 형식의 보고서 내용
        self.url = url  # 원본 YouTube 영상 URL
        self.published_date = published_date


async def find_relevant_reports(
    agent, 
    backtest_date: Optional[str] = None,
    max_reports: int = 30,
    debug_info: Dict[str, Any] = None,
    worker_id: str = None,
    notion_api_manager = None,
    gemini_api_manager = None
) -> List[Report]:
    """
    에이전트의 조건에 맞는 보고서를 검색합니다.
    
    Args:
        agent: 투자 에이전트 객체
        backtest_date: 백테스팅 날짜 (기본: 현재 날짜)
        max_reports: 최대 보고서 수
        debug_info: 디버깅 정보를 저장할 딕셔너리
        worker_id: 워커 ID (병렬 처리용)
        notion_api_manager: 노션 API 관리자 (선택 사항)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
        
    Returns:
        관련 보고서 목록
    """
    # 로그 접두어 (워커 ID가 있으면 포함)
    log_prefix = f"[{worker_id or 'main'}] " 
    
    try:
        # 백테스팅 날짜 설정 (기본값: 현재 날짜)
        if backtest_date is None:
            backtest_date = datetime.now().isoformat()
        elif isinstance(backtest_date, str):
            # ISO 형식이 아니면 변환 시도
            if "T" not in backtest_date:
                backtest_date = f"{backtest_date}T00:00:00Z"
        
        logger.info(f"{log_prefix}백테스팅 날짜 {backtest_date} 이전의 보고서만 검색합니다.")
        
        # 채널 조건
        channel_condition = None
        if agent.target_channels:
            channel_options = []
            for channel in agent.target_channels:
                channel_options.append({
                    "property": "채널명",
                    "select": {
                        "equals": channel
                    }
                })
            
            if len(channel_options) == 1:
                channel_condition = channel_options[0]
            else:
                channel_condition = {
                    "or": channel_options
                }
        
        # 키워드 조건 (제목 기준)
        keyword_condition = None
        if agent.target_keywords:
            keyword_options = []
            for keyword in agent.target_keywords:
                keyword_options.append({
                    "property": "제목",
                    "title": {
                        "contains": keyword
                    }
                })
            
            if len(keyword_options) == 1:
                keyword_condition = keyword_options[0]
            else:
                keyword_condition = {
                    "or": keyword_options
                }
        
        # 날짜 조건 - 백테스팅 날짜 이전의 보고서만
        date_condition = {
            "property": "영상 날짜",
            "date": {
                "before": backtest_date  # 백테스팅 날짜 이전
            }
        }
        
        # 필터 조합
        filter_conditions = []
        
        if channel_condition:
            filter_conditions.append(channel_condition)
        
        if keyword_condition:
            filter_conditions.append(keyword_condition)
            
        filter_conditions.append(date_condition)
        
        # 최종 필터
        filter_obj = {
            "and": filter_conditions
        }
        
        # 정렬 및 페이지 크기 제한 추가 (최신 데이터 우선)
        request_body = {
            "filter": filter_obj,
            "sorts": [
                {
                    "property": "영상 날짜",
                    "direction": "descending"  # 최신 날짜 우선
                }
            ],
            "page_size": max_reports  # 최대 검색 수 제한
        }
        
        # Notion DB 쿼리 - API 관리자 사용 여부에 따라 다른 방식으로 호출
        if notion_api_manager:
            script_pages = await notion_api_manager.query_notion_database(NOTION_SCRIPT_DB_ID, request_body)
        else:
            from notion_utils import query_notion_database
            script_pages = await query_notion_database(NOTION_SCRIPT_DB_ID, request_body)
        
        logger.info(f"{log_prefix}에이전트 조건에 맞는 보고서 {len(script_pages)}개 찾음 (최대 {max_reports}개, 날짜 {backtest_date} 이전)")
        
        # 보고서 메타데이터 객체 생성 (내용은 비워둠)
        candidate_reports_metadata = []
        for page in script_pages:
            properties = page.get("properties", {})
            
            # 기본 정보 추출
            title = ""
            channel = "기타"
            url = ""
            published_date = ""
            
            # 제목 (키워드)
            if "제목" in properties and "title" in properties["제목"]:
                title_obj = properties["제목"]["title"]
                if title_obj and len(title_obj) > 0:
                    title = title_obj[0]["plain_text"]
            
            # 채널명
            if "채널명" in properties and "select" in properties["채널명"]:
                select_obj = properties["채널명"]["select"]
                if select_obj:
                    channel = select_obj["name"]
            
            # URL
            if "URL" in properties and "url" in properties["URL"]:
                url = properties["URL"]["url"]
            
            # 영상 날짜
            if "영상 날짜" in properties and "date" in properties["영상 날짜"]:
                date_obj = properties["영상 날짜"]["date"]
                if date_obj and "start" in date_obj:
                    published_date = date_obj["start"]
            
            # 페이지 ID와 메타데이터만 저장
            candidate_reports_metadata.append({
                "page_id": page.get("id"),
                "title": title,
                "channel": channel,
                "url": url,
                "published_date": published_date
            })
        
        
        # 데이터 선별 에이전트를 사용하여 적합한 보고서 선택
        from report_selector import select_reports_by_agent_preference
        
        # API 관리자 사용 여부에 따라 다른 방식으로 호출
        if gemini_api_manager and worker_id:
            selection_result = await select_reports_by_agent_preference(
                agent=agent,
                candidate_reports_metadata=candidate_reports_metadata,
                backtest_date=backtest_date,
                debug_info=debug_info,
                worker_id=worker_id,
                gemini_api_manager=gemini_api_manager
            )
        else:
            selection_result = await select_reports_by_agent_preference(
                agent=agent,
                candidate_reports_metadata=candidate_reports_metadata,
                backtest_date=backtest_date,
                debug_info=debug_info
            )
        
        # 선택된 보고서 ID 및 선택 이유 추출
        selected_report_ids = selection_result["selected_report_ids"]
        selection_info = selection_result["selection_info"]
        
        # 디버깅 정보 저장
        if debug_info is not None:
            debug_info["report_selection_result"] = {
                "total_candidates": len(candidate_reports_metadata),
                "selected_count": len(selected_report_ids),
                "selection_strategy": selection_info.get("strategy", ""),
                "selection_details": selection_info.get("details", [])
            }
        
        logger.info(f"{log_prefix}데이터 선별 에이전트가 {len(selected_report_ids)}개 보고서 선택 (전략: {selection_info.get('strategy', '')})")
        
        # 선택된 보고서만 필터링
        selected_reports_meta = [report_meta for report_meta in candidate_reports_metadata 
                              if report_meta["page_id"] in selected_report_ids]
        
        # 병렬로 보고서 내용 가져오기
        async def fetch_report_with_content(report_meta):
            if notion_api_manager:
                # API 관리자 통한 페이지 내용 가져오기
                content = await get_notion_page_content_with_manager(
                    report_meta["page_id"], 
                    notion_api_manager
                )
            else:
                from notion_utils import get_notion_page_content
                content = await get_notion_page_content(report_meta["page_id"])
                
            return Report(
                page_id=report_meta["page_id"],
                title=report_meta["title"],
                channel=report_meta["channel"],
                content=content,
                url=report_meta["url"],
                published_date=report_meta["published_date"]
            )
        
        # 병렬 실행
        fetch_tasks = [fetch_report_with_content(report_meta) for report_meta in selected_reports_meta]
        selected_reports = await asyncio.gather(*fetch_tasks)
        
        return selected_reports
        
    except Exception as e:
        logger.error(f"{log_prefix}관련 보고서 검색 중 오류: {str(e)}")
        return []

async def get_notion_page_content_with_manager(page_id: str, notion_api_manager) -> str:
    """
    API 관리자를 통해 노션 페이지 컨텐츠(블록)를 조회합니다.
    
    Args:
        page_id: 노션 페이지 ID
        notion_api_manager: 노션 API 관리자
        
    Returns:
        페이지 컨텐츠 (마크다운 형식)
    """
    # 이 함수는 향후 구현 예정
    # 현재는 기존 함수를 호출
    from notion_utils import get_notion_page_content
    return await get_notion_page_content(page_id)

async def analyze_reports_with_llm(
    reports: List[Report], 
    agent, 
    max_stocks: int = 5, 
    investment_period: int = 7,
    worker_id: str = None,
    notion_api_manager = None,
    gemini_api_manager = None
) -> Dict[str, Any]:
    """
    LLM을 사용하여 보고서를 분석하고 종목을 추천합니다.
    내부적으로 recommend_stocks 함수를 호출하여 중복을 제거합니다.
    
    Args:
        reports: 분석할 보고서 리스트
        agent: 투자 에이전트 객체
        max_stocks: 추천할 최대 종목 수
        investment_period: 투자 기간 (일)
        worker_id: 워커 ID (병렬 처리용)
        notion_api_manager: 노션 API 관리자 (선택 사항)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
        
    Returns:
        분석된 보고서 데이터 및 추천 종목
    """
    # 로그 접두어
    log_prefix = f"[{worker_id or 'main'}] "
    logger.info(f"{log_prefix}analyze_reports_with_llm 시작: 보고서 {len(reports)}개")
    
    try:
        if not reports:
            logger.warning(f"{log_prefix}분석할 보고서가 없습니다.")
            return {
                "reports": [],
                "stocks": [],
                "total_reports": 0,
                "total_stocks": 0,
                "recommended_stocks": [],
                "portfolio_logic": "분석할 보고서가 없습니다."
            }
        
        # 1. 보고서 데이터를 recommend_stocks 함수에서 사용할 형식으로 변환
        analyzed_reports = {
            "reports": [
                {
                    "id": report.page_id,
                    "title": report.title,
                    "channel": report.channel,
                    "published_date": report.published_date,
                    "content": report.content
                } for report in reports
            ],
            "total_reports": len(reports)
        }
        
        # 2. recommend_stocks 함수 호출 (이미 개선된 함수)
        from stock_recommender import recommend_stocks
        
        result = await recommend_stocks(
            agent=agent,
            analyzed_reports=analyzed_reports,
            max_stocks=max_stocks,
            investment_period=investment_period,
            worker_id=worker_id,
            gemini_api_manager=gemini_api_manager
        )
        
        # 3. recommend_stocks의 반환값을 analyze_reports_with_llm의 형식에 맞게 변환
        final_result = {
            "reports": [{"id": r.page_id, "title": r.title, "channel": r.channel, "published_date": r.published_date} for r in reports],
            "stocks": result.get("recommended_stocks", []),
            "total_reports": len(reports),
            "total_stocks": len(result.get("recommended_stocks", [])),
            "recommended_stocks": result.get("recommended_stocks", []),
            "portfolio_logic": result.get("portfolio_logic", ""),
            "analysis_text": result.get("analysis_text", "")
        }
        
        logger.info(f"{log_prefix}analyze_reports_with_llm 완료: {len(final_result['recommended_stocks'])}개 종목 추천")
        return final_result
        
    except Exception as e:
        logger.error(f"{log_prefix}LLM 보고서 분석 중 오류: {str(e)}", exc_info=True)
        return {
            "reports": [],
            "stocks": [],
            "total_reports": 0,
            "total_stocks": 0,
            "error": str(e)
        }