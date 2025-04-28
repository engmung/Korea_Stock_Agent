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
        self.stocks = []  # 추출된 종목 리스트


class StockMention:
    """보고서에서 언급된 종목 정보"""
    def __init__(self, name: str, code: str = None):
        self.name = name
        self.code = code
        self.recommendation = ""  # 매수, 적극매수, 중립, 매도 등
        self.horizon = ""  # 단기, 중기, 장기 등
        self.reasons = []  # 언급 이유 리스트
        self.report_count = 1  # 언급된 보고서 수
        self.recent_mention = None  # 가장 최근 언급 날짜
        self.mentioned_reports = []  # 언급된 보고서 ID 리스트


async def find_relevant_reports(
    agent, 
    backtest_date: Optional[str] = None,
    max_reports: int = 40,
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
    log_prefix = f"[{worker_id}] " if worker_id else ""
    
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
    # 로그 접두어 (워커 ID가 있으면 포함)
    log_prefix = f"[{worker_id}] " if worker_id else ""
    
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
            
        # 투자 기간 텍스트 변환
        investment_period_text = f"{investment_period}일"
        if investment_period <= 7:
            investment_horizon = "단기"
        elif investment_period <= 30:
            investment_horizon = "중기"
        else:
            investment_horizon = "장기"
            
        # 에이전트 프롬프트 가져오기
        from report_selector import get_agent_prompt
        agent_prompt = await get_agent_prompt(agent.page_id)
        
        if not agent_prompt:
            logger.warning(f"{log_prefix}에이전트 프롬프트를 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
            agent_prompt = f"{agent.agent_name} 에이전트는 다음과 같은 투자 철학을 가지고 있습니다: {agent.investment_philosophy}"
            
        # 보고서 내용 포맷팅
        reports_content = ""
        for i, report in enumerate(reports):
            reports_content += f"### 보고서 {i+1}: {report.title}\n"
            reports_content += f"채널: {report.channel}\n"
            reports_content += f"날짜: {report.published_date}\n"
            reports_content += f"내용:\n{report.content}\n\n"
            
        # 프롬프트 생성
        analysis_prompt = f"""
당신은 투자 분석 및 추천 전문가입니다. 다음 투자 에이전트의 투자 철학과 전략에 맞게 제공된 보고서들을 분석하고 최적의 종목을 추천해주세요.

## 투자 에이전트 프롬프트
{agent_prompt}

## 분석할 보고서 목록 (총 {len(reports)}개)
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

        # Gemini API 호출 방식 결정
        from stock_recommender import GeminiClient
        
        if gemini_api_manager and worker_id:
            # API 관리자 사용
            api_key = await gemini_api_manager.get_api_key(worker_id)
            gemini_client = GeminiClient(api_key=api_key)
        else:
            # 기존 방식
            gemini_client = GeminiClient(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # 분석 및 추천 수행
        logger.info(f"{log_prefix}LLM을 사용하여 보고서 분석 및 종목 추천 시작...")
        analysis_result = await gemini_client.analyze_with_agent_prompt(
            system_prompt="당신은 투자 분석 및 추천 전문가입니다. 투자 에이전트의 철학에 맞는 최적의 종목을 추천해주세요.",
            analysis_data=analysis_prompt
        )
        
        logger.info(f"{log_prefix}분석 완료, 결과 파싱 시작...")
        
        # 결과 파싱
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
        
        # 원본 분석 결과 포함
        parsed_recommendation["analysis_text"] = analysis_result
        
        # 종목 티커 코드 매핑 추가
        stock_tickers = {}
        
        if "recommended_stocks" in parsed_recommendation:
            from stock_searcher import StockSearcher
            
            # API 관리자 사용 여부에 따라 방식 결정
            if gemini_api_manager and worker_id:
                api_key = await gemini_api_manager.get_api_key(worker_id)
                stock_searcher = StockSearcher(api_key=api_key)
            else:
                stock_searcher = StockSearcher(api_key=os.environ.get("GEMINI_API_KEY"))
            
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
                        stock_tickers[ticker] = stock_name
                        stock["ticker"] = ticker
                    else:
                        # KRX 직접 검색 시도
                        try:
                            import FinanceDataReader as fdr
                            krx_listing = await asyncio.to_thread(fdr.StockListing, 'KRX')
                            matches = krx_listing[krx_listing['Name'].str.contains(stock_name, case=False)]
                            if not matches.empty:
                                ticker = matches.iloc[0]['Symbol']
                                stock_name_exact = matches.iloc[0]['Name']
                                logger.info(f"{log_prefix}KRX 검색으로 종목 '{stock_name}'에 대한 코드 '{ticker}' 찾음")
                                stock_tickers[ticker] = stock_name_exact
                                stock["ticker"] = ticker
                                stock["name"] = stock_name_exact
                        except Exception as e:
                            logger.info(f"{log_prefix}KRX 검색 중 오류, 건너뜀: {str(e)}")
        
        # 백테스팅을 위한 티커 매핑 추가
        parsed_recommendation["stock_tickers"] = stock_tickers
        
        # 기본 분석 결과 구조에 맞게 추가 데이터 구성
        result = {
            "reports": [{"id": r.page_id, "title": r.title, "channel": r.channel, "published_date": r.published_date} for r in reports],
            "stocks": parsed_recommendation.get("recommended_stocks", []),
            "total_reports": len(reports),
            "total_stocks": len(parsed_recommendation.get("recommended_stocks", [])),
            "recommended_stocks": parsed_recommendation.get("recommended_stocks", []),
            "portfolio_logic": parsed_recommendation.get("portfolio_logic", ""),
            "analysis_text": analysis_result
        }
        
        logger.info(f"{log_prefix}LLM 분석 완료: {len(result['recommended_stocks'])}개 종목 추천")
        return result
        
    except Exception as e:
        logger.error(f"{log_prefix}LLM 보고서 분석 중 오류: {str(e)}")
        return {
            "reports": [],
            "stocks": [],
            "total_reports": 0,
            "total_stocks": 0,
            "error": str(e)
        }

async def analyze_reports(
    reports: List[Report],
    worker_id: str = None,
    notion_api_manager = None,
    gemini_api_manager = None
) -> Dict[str, Any]:
    """
    보고서를 분석하여 종목 정보를 추출합니다.
    
    Args:
        reports: 분석할 보고서 리스트
        worker_id: 워커 ID (병렬 처리용)
        notion_api_manager: 노션 API 관리자 (선택 사항)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
        
    Returns:
        분석된 보고서 데이터 딕셔너리
    """
    try:
        # 로그 접두어 (워커 ID가 있으면 포함)
        log_prefix = f"[{worker_id}] " if worker_id else ""
        
        analyzed_reports = []
        stock_mentions = {}  # 종목별 언급 정보 통합
        
        for report in reports:
            # 보고서에서 종목 정보 추출
            stocks = await extract_stocks_from_report(report)
            report.stocks = stocks
            
            # 종목별 언급 정보 통합
            for stock in stocks:
                stock_key = stock.get("name")
                
                if stock_key in stock_mentions:
                    # 기존 종목 정보 업데이트
                    stock_mentions[stock_key]["report_count"] += 1
                    stock_mentions[stock_key]["reports"].append({
                        "id": report.page_id,
                        "title": report.title,
                        "channel": report.channel,
                        "date": report.published_date,
                        "recommendation": stock.get("recommendation", "언급"),
                        "investment_horizon": stock.get("investment_horizon", "")
                    })
                    
                    # 최신 언급일 업데이트
                    if report.published_date > stock_mentions[stock_key]["recent_mention"]:
                        stock_mentions[stock_key]["recent_mention"] = report.published_date
                        
                    # 추천 강도 업데이트 (최신 추천 우선)
                    if stock.get("recommendation"):
                        stock_mentions[stock_key]["recommendation"] = stock.get("recommendation")
                        
                    # 투자 기간 업데이트 (최신 언급 우선)
                    if stock.get("investment_horizon"):
                        stock_mentions[stock_key]["investment_horizon"] = stock.get("investment_horizon")
                        
                    # 언급 이유 추가
                    if stock.get("reasons"):
                        for reason in stock.get("reasons", []):
                            if reason not in stock_mentions[stock_key]["reasons"]:
                                stock_mentions[stock_key]["reasons"].append(reason)
                else:
                    # 새 종목 정보 추가
                    stock_mentions[stock_key] = {
                        "name": stock_key,
                        "code": stock.get("code", ""),
                        "recommendation": stock.get("recommendation", "언급"),
                        "investment_horizon": stock.get("investment_horizon", ""),
                        "reasons": stock.get("reasons", []),
                        "report_count": 1,
                        "recent_mention": report.published_date,
                        "reports": [{
                            "id": report.page_id,
                            "title": report.title,
                            "channel": report.channel,
                            "date": report.published_date,
                            "recommendation": stock.get("recommendation", "언급"),
                            "investment_horizon": stock.get("investment_horizon", "")
                        }]
                    }
            
            # 분석된 보고서 추가
            analyzed_reports.append({
                "id": report.page_id,
                "title": report.title,
                "channel": report.channel,
                "url": report.url,
                "published_date": report.published_date,
                "stocks": stocks
            })
        
        # 최신 언급일 순으로 종목 정렬
        sorted_stocks = sorted(
            stock_mentions.values(), 
            key=lambda x: (x["recent_mention"], x["report_count"]), 
            reverse=True
        )
        
        logger.info(f"{log_prefix}보고서 분석 완료: {len(reports)}개 보고서, {len(sorted_stocks)}개 종목 추출")
        
        return {
            "reports": analyzed_reports,
            "stocks": sorted_stocks,
            "total_reports": len(reports),
            "total_stocks": len(sorted_stocks)
        }
        
    except Exception as e:
        logger.error(f"{log_prefix if 'log_prefix' in locals() else ''}보고서 분석 중 오류: {str(e)}")
        return {
            "reports": [],
            "stocks": [],
            "total_reports": 0,
            "total_stocks": 0,
            "error": str(e)
        }

async def extract_stocks_from_report(report: Report) -> List[Dict[str, Any]]:
    """
    보고서 내용에서 종목 정보를 추출합니다.
    
    Args:
        report: 분석할 보고서 객체
        
    Returns:
        추출된 종목 정보 리스트
    """
    try:
        # 내용이 너무 길면 필요한 부분만 추출 (성능 최적화)
        MAX_CONTENT_LENGTH = 20000  # 최대 처리 길이 제한
        content = report.content
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH]
            logger.info(f"보고서 내용이 너무 길어 처음 {MAX_CONTENT_LENGTH} 문자만 처리합니다.")
        
        stocks = []
        
        # 보고서가 마크다운 형식이라고 가정하고 파싱
        
        # 섹션 별로 나누기
        sections = []
        current_section = {"title": "기본", "content": ""}
        
        for line in content.split('\n'):
            # 새로운 섹션 시작 (h2 제목 - ## 으로 시작)
            if line.startswith('## '):
                # 이전 섹션 저장
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # 새 섹션 시작
                current_section = {
                    "title": line[3:].strip(),
                    "content": ""
                }
            else:
                # 현재 섹션에 내용 추가
                current_section["content"] += line + "\n"
        
        # 마지막 섹션 저장
        if current_section["content"].strip():
            sections.append(current_section)
        
        # 섹션 별로 종목 정보 추출
        for section in sections:
            section_title = section["title"]
            section_content = section["content"]
            
            # 종목명이 섹션 제목인 경우
            if section_title and section_title != "기본":
                # 종목 코드가 있는지 확인 (괄호 안에 숫자 6자리)
                code_match = re.search(r'\((\d{6})\)', section_title)
                stock_code = code_match.group(1) if code_match else None
                
                # 종목명에서 코드 제거
                stock_name = re.sub(r'\s*\(\d{6}\)\s*', '', section_title).strip()
                
                # 기본 종목 정보
                stock_info = {
                    "name": stock_name,
                    "code": stock_code,
                    "recommendation": None,
                    "investment_horizon": None,
                    "reasons": [],
                    "source_channel": report.channel,  # 출처 채널 추가
                    "source_title": report.title,     # 출처 제목 추가
                    "source_date": report.published_date  # 출처 날짜 추가
                }
                
                # 섹션 내용에서 추천 강도, 투자 기간, 이유 추출
                recommend_patterns = {
                    "적극매수": ["적극\s*매수", "강력\s*매수", "강매", "강한\s*매수"],
                    "매수": ["매수", "매수\s*추천", "비중\s*확대"],
                    "중립": ["중립", "관망", "홀딩"],
                    "매도": ["매도", "비중\s*축소"]
                }
                
                horizon_patterns = {
                    "단기": ["단기", "1주일", "2주일", "단타"],
                    "중기": ["중기", "한달", "1개월", "2개월", "3개월"],
                    "장기": ["장기", "중장기", "6개월", "1년"]
                }
                
                # 추천 강도 추출
                for rec, patterns in recommend_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, section_content, re.IGNORECASE):
                            stock_info["recommendation"] = rec
                            break
                    if stock_info["recommendation"]:
                        break
                
                # 투자 기간 추출
                for horizon, patterns in horizon_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, section_content, re.IGNORECASE):
                            stock_info["investment_horizon"] = horizon
                            break
                    if stock_info["investment_horizon"]:
                        break
                
                # 추천 이유 추출 (불릿 포인트)
                reason_matches = re.findall(r'[-*]\s+(.*?)(?:\n|$)', section_content)
                if reason_matches:
                    reasons = [reason.strip() for reason in reason_matches if reason.strip()][:7]
                    stock_info["reasons"] = reasons
                
                stocks.append(stock_info)
        
        return stocks
        
    except Exception as e:
        logger.error(f"종목 정보 추출 중 오류: {str(e)}")
        return []