import os
import re
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from functools import wraps
import concurrent.futures

from notion_utils import update_notion_page
from performance_evaluator import backtest_recommendation
from report_analyzer import find_relevant_reports, analyze_reports_with_llm
from apscheduler.schedulers.background import BackgroundScheduler

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Notion DB ID
NOTION_AGENT_DB_ID = os.environ.get("NOTION_AGENT_DB_ID")

# 백테스팅 실행 중 여부를 확인하는 플래그
is_backtest_running = False

# 백테스팅 예약 스케줄러를 실행하는 동기 함수
def run_backtest_scheduler():
    """
    APScheduler에서 호출할 동기 함수입니다.
    백테스팅 시스템을 시작합니다.
    """
    global is_backtest_running
    
    logger.info("===== 백테스팅 스케줄러 실행 시작 =====")
    
    # 이미 실행 중이면 스킵
    if is_backtest_running:
        logger.info("이전 백테스팅이 아직 실행 중입니다. 이번 예약 체크는 건너뜁니다.")
        return
        
    try:
        # 실행 중 플래그 설정
        is_backtest_running = True
        logger.info("백테스팅 실행 플래그를 True로 설정")
        
        # 병렬 처리 시스템을 통한 백테스팅 시도
        logger.info("병렬 처리 시스템을 통한 백테스팅 시도")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # 타임아웃 없이 작업 실행
                future = executor.submit(run_async_in_thread, start_parallel_system)
                logger.info("병렬 시스템 작업 제출됨")
                
                # 결과를 기다림 (타임아웃 없음)
                future.result()
                logger.info("병렬 시스템 실행 완료")
                
            except Exception as e:
                logger.error(f"병렬 백테스팅 시스템 실행 중 오류: {str(e)}")
                import traceback
                logger.error(f"병렬 시스템 오류 상세: {traceback.format_exc()}")
    except Exception as e:
        logger.error(f"백테스팅 스케줄러 전체 실행 중 오류: {str(e)}")
        import traceback
        logger.error(f"스케줄러 오류 상세: {traceback.format_exc()}")
    finally:
        # 실행 완료 후 플래그 해제
        is_backtest_running = False
        logger.info("백테스팅 실행 플래그를 False로 재설정")
        logger.info("===== 백테스팅 스케줄러 실행 완료 =====")

def run_async_in_thread(async_func):
    """
    별도의 스레드에서 비동기 함수를 실행합니다.
    새 이벤트 루프를 생성하고 비동기 함수를 실행합니다.
    """
    # 새 이벤트 루프 생성
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 비동기 함수 실행
        return loop.run_until_complete(async_func())
    finally:
        # 루프 닫기
        loop.close()

async def start_parallel_system():
    """병렬 백테스팅 시스템을 시작하고 완료될 때까지 대기합니다."""
    from backtest_parallel import BacktestParallelSystem
    
    # 워커 수 설정 (환경 변수 또는 기본값 3)
    num_workers = int(os.environ.get("BACKTEST_NUM_WORKERS", 3))
    
    # 병렬 시스템 생성
    system = BacktestParallelSystem(num_workers=num_workers)
    
    try:
        # 시스템 시작
        await system.start()
        
        # 백테스팅 사이클 실행
        await system.run_backtest_cycle()
        
        # 모든 작업 완료 대기 (최대 1시간)
        await system.wait_for_completion(timeout=3600)
        
    finally:
        # 시스템 중지
        await system.stop()

async def parse_date_ranges(schedule_text: str) -> List[Dict[str, str]]:
    """
    예약 문자열에서 날짜 범위를 파싱합니다.
    줄바꿈 또는 쉼표로 구분된 형식 지원:
    
    지원하는 형식:
    - MMDD~MMDD (기본 형식, 현재 연도 사용)
    - YYMMDD~YYMMDD (연도 포함 형식)
    
    형식 1 (줄바꿈):
    0522~0526
    240601~240605
    
    형식 2 (쉼표):
    0522~0526, 0601~0605, 240701~240705
    
    Returns:
        List[Dict[str, str]]: 시작일과 종료일 딕셔너리 리스트
    """
    date_ranges = []
    
    # 입력이 비어있으면 빈 리스트 반환
    if not schedule_text:
        return []
        
    # 텍스트 정리
    cleaned_text = schedule_text.strip()
    
    # 줄바꿈으로 분리
    lines = [line.strip() for line in cleaned_text.split('\n')]
    
    # 각 줄을 개별 항목으로 처리
    items = []
    for line in lines:
        if not line:
            continue
            
        # 각 줄 내에서 쉼표로 분리된 항목 처리
        if ',' in line:
            items.extend([item.strip() for item in line.split(',') if item.strip()])
        else:
            items.append(line)
    
    logger.info(f"분석된 예약 항목: {items}")
    
    current_year = datetime.now().year
    current_year_short = current_year % 100  # 현재 연도의 마지막 두 자리
    
    for item in items:
        # 날짜 범위 파싱 - 연도 포함 형식(YYMMDD~YYMMDD)과 기본 형식(MMDD~MMDD) 모두 지원
        match_with_year = re.match(r'(\d{6})~(\d{6})', item)
        match_without_year = re.match(r'(\d{4})~(\d{4})', item)
        
        if match_with_year:
            # YYMMDD~YYMMDD 형식 처리
            start_yymmdd, end_yymmdd = match_with_year.groups()
            
            try:
                # 연도, 월, 일 분리
                start_yy, start_mm, start_dd = int(start_yymmdd[:2]), int(start_yymmdd[2:4]), int(start_yymmdd[4:])
                end_yy, end_mm, end_dd = int(end_yymmdd[:2]), int(end_yymmdd[2:4]), int(end_yymmdd[4:])
                
                # 2000년대 연도로 변환 (20YY)
                start_year = 2000 + start_yy
                end_year = 2000 + end_yy
                
                # 날짜 유효성 검증 및 변환
                start_date = f"{start_year}-{start_mm:02d}-{start_dd:02d}"
                end_date = f"{end_year}-{end_mm:02d}-{end_dd:02d}"
                
                logger.info(f"백테스팅 일정 추가(연도 포함): {start_date} ~ {end_date} (원본: {item})")
                
                date_ranges.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "original_text": item
                })
            except ValueError as e:
                logger.error(f"유효하지 않은 날짜 형식(연도 포함): {item} - {str(e)}")
                
        elif match_without_year:
            # MMDD~MMDD 형식 처리 (기존 로직)
            start_mmdd, end_mmdd = match_without_year.groups()
            
            try:
                start_mm, start_dd = int(start_mmdd[:2]), int(start_mmdd[2:])
                end_mm, end_dd = int(end_mmdd[:2]), int(end_mmdd[2:])
                
                # 현재 연도 사용
                start_year = end_year = current_year
                
                # 날짜 유효성 검증 및 변환
                start_date = f"{start_year}-{start_mm:02d}-{start_dd:02d}"
                end_date = f"{end_year}-{end_mm:02d}-{end_dd:02d}"
                
                # 시작일이 종료일보다 나중인 경우 처리
                if start_date > end_date:
                    # 종료일이 다음 해로 넘어간 경우
                    end_date = f"{current_year + 1}-{end_mm:02d}-{end_dd:02d}"
                
                logger.info(f"백테스팅 일정 추가(기본): {start_date} ~ {end_date} (원본: {item})")
                
                date_ranges.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "original_text": item
                })
            except ValueError as e:
                logger.error(f"유효하지 않은 날짜 형식(기본): {item} - {str(e)}")
        else:
            logger.warning(f"잘못된 형식의 예약 항목 건너뜀: {item}")
    
    return date_ranges

async def update_schedule_text(page_id: str, current_schedule: str, completed_item: str, notion_api_manager=None) -> bool:
    """
    완료된 백테스팅 일정을 예약 문자열에서 제거합니다.
    
    Args:
        page_id (str): 에이전트 페이지 ID
        current_schedule (str): 현재 예약 문자열
        completed_item (str): 완료된 예약 항목
        notion_api_manager: 노션 API 관리자 (선택 사항)
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # 대괄호 제거 (있는 경우)
        cleaned_text = current_schedule.strip()
        if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
            cleaned_text = cleaned_text[1:-1]
        
        # 쉼표로 분리하여 각 항목 추출 - 공백까지 정확하게 처리
        items = []
        for item in cleaned_text.split(','):
            item_trimmed = item.strip()
            if item_trimmed:
                items.append(item_trimmed)
        
        logger.info(f"원본 예약 항목들: {items}")
        
        # 완료된 항목 정확히 찾아 제거
        completed_item_trimmed = completed_item.strip()
        
        if completed_item_trimmed in items:
            logger.info(f"정확히 일치하는 항목 제거: '{completed_item_trimmed}'")
            items.remove(completed_item_trimmed)
        else:
            # 정확히 일치하는 항목이 없으면 다른 방법으로 시도
            # MMDD~MMDD 형식을 기준으로 비교
            match = re.match(r'(\d{4})~(\d{4})', completed_item_trimmed)
            if match:
                start_mmdd, end_mmdd = match.groups()
                logger.info(f"날짜 범위로 항목 검색: {start_mmdd}~{end_mmdd}")
                
                # 각 항목을 순회하며 동일한 날짜 범위를 가진 항목 찾기
                for item in items[:]:  # 복사본으로 순회
                    item_match = re.match(r'(\d{4})~(\d{4})', item)
                    if item_match and item_match.groups() == (start_mmdd, end_mmdd):
                        logger.info(f"날짜 범위가 일치하는 항목 제거: '{item}'")
                        items.remove(item)
                        break
        
        # 남은 항목이 있으면 다시 쉼표로 연결
        new_schedule = ", ".join(items) if items else ""
        
        # Notion 페이지 업데이트
        properties = {
            "백테스팅 예약": {
                "rich_text": [{
                    "text": {
                        "content": new_schedule
                    }
                }] if new_schedule else []
            }
        }
        
        logger.info(f"예약 문자열 업데이트: '{current_schedule}' -> '{new_schedule}' (완료 항목: '{completed_item}')")
        
        # API 관리자 사용 여부에 따라 다른 방식으로 호출
        if notion_api_manager:
            return await notion_api_manager.update_notion_page(page_id, properties)
        else:
            return await update_notion_page(page_id, properties)
    
    except Exception as e:
        logger.error(f"예약 문자열 업데이트 중 오류: {str(e)}")
        return False

async def disable_stock_recommendation(page_id: str, notion_api_manager=None) -> bool:
    """
    종목 추천 체크박스를 비활성화합니다.
    
    Args:
        page_id (str): 에이전트 페이지 ID
        notion_api_manager: 노션 API 관리자 (선택 사항)
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # 종목 추천 체크박스 비활성화 (체크 해제)
        properties = {
            "종목추천": {
                "checkbox": False
            }
        }
        
        logger.info(f"종목 추천 체크박스 비활성화: 페이지 ID {page_id}")
        
        # API 관리자 사용 여부에 따라 다른 방식으로 호출
        if notion_api_manager:
            return await notion_api_manager.update_notion_page(page_id, properties)
        else:
            return await update_notion_page(page_id, properties)
    
    except Exception as e:
        logger.error(f"종목 추천 체크박스 비활성화 중 오류: {str(e)}")
        return False

async def process_agent_schedules(
    agent: Dict[str, Any], 
    worker_id: str = None,
    notion_api_manager = None,
    gemini_api_manager = None,
    dispatcher = None  # 디스패처 참조 추가
) -> None:
    """
    에이전트의 백테스팅 예약과 종목 추천 설정을 처리합니다.
    
    Args:
        agent (Dict[str, Any]): 에이전트 정보
        worker_id (str): 워커 ID (병렬 처리용)
        notion_api_manager: 노션 API 관리자 (선택 사항)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
        dispatcher: 디스패처 참조 (예약 분산 처리용)
    """
    page_id = agent.get("id")
    properties = agent.get("properties", {})
    
    # 로그 접두어 (워커 ID가 있으면 포함)
    log_prefix = f"[{worker_id}] " if worker_id else ""
    
    # 에이전트 이름 가져오기
    agent_name = "Unknown"
    if "에이전트명" in properties and "title" in properties["에이전트명"]:
        title_objs = properties["에이전트명"]["title"]
        if title_objs and len(title_objs) > 0:
            agent_name = title_objs[0].get("plain_text", "Unknown")
    
    # 1. 종목 추천 체크박스 확인
    is_recommendation_enabled = False
    if "종목추천" in properties and "checkbox" in properties["종목추천"]:
        is_recommendation_enabled = properties["종목추천"]["checkbox"]
    
    if is_recommendation_enabled:
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 종목 추천 활성화 상태 확인됨")
        
        try:
            # 종목 추천 실행
            await process_stock_recommendation(
                page_id=page_id, 
                agent_name=agent_name,
                worker_id=worker_id,
                notion_api_manager=notion_api_manager,
                gemini_api_manager=gemini_api_manager
            )
            
            # 추천 후 체크박스 비활성화
            await disable_stock_recommendation(
                page_id=page_id,
                notion_api_manager=notion_api_manager
            )
            logger.info(f"{log_prefix}에이전트 '{agent_name}'의 종목 추천 처리 완료 및 체크박스 비활성화")
        except Exception as e:
            logger.error(f"{log_prefix}종목 추천 실행 중 오류: {str(e)}")
    
    # 2. '백테스팅 예약' 속성 확인
    if "백테스팅 예약" not in properties:
        return
    
    schedule_prop = properties["백테스팅 예약"]
    
    # 속성 타입 확인
    if "rich_text" not in schedule_prop or not schedule_prop["rich_text"]:
        return
    
    # 예약 문자열 가져오기
    schedule_text = schedule_prop["rich_text"][0]["plain_text"].strip() if schedule_prop["rich_text"] else ""
    
    if not schedule_text:
        return
    
    logger.info(f"{log_prefix}에이전트 '{agent_name}' 백테스팅 예약 처리: '{schedule_text}'")
    
    # 날짜 범위 파싱
    date_ranges = await parse_date_ranges(schedule_text)
    logger.info(f"{log_prefix}파싱된 백테스팅 예약: {len(date_ranges)}개")
    
    # 현재 날짜
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 처리할 예약 개수 확인 (오늘 이전이거나 같은 날짜 범위)
    pending_backtests = [dr for dr in date_ranges if dr["end_date"] <= today]
    
    if pending_backtests:
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 처리 예정 백테스팅: {len(pending_backtests)}개")
        for i, bt in enumerate(pending_backtests):
            logger.info(f"{log_prefix}  {i+1}. {bt['start_date']} ~ {bt['end_date']} (원본: {bt['original_text']})")
    
    # 디스패처가 제공되면 개별 예약을 분산 처리
    if dispatcher and len(pending_backtests) > 1:
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 {len(pending_backtests)}개 예약을 분산 처리합니다.")
        
        # 각 예약을 개별 작업으로 디스패처에 전달
        for date_range in pending_backtests:
            # 예약 작업 생성
            backtest_task = {
                "agent_id": page_id,
                "agent_name": agent_name,
                "date_range": date_range,
                "type": "backtest_reservation"
            }
            
            # 디스패처에 작업 전달
            await dispatcher.dispatch_backtest_task(backtest_task)
        
        # 예약 처리 후 필드 비우기 (개별 작업이 완료될 때 수행되므로 여기서는 하지 않음)
        return
    
    # 디스패처가 없거나 예약이 1개 이하인 경우 기존 방식으로 처리
    processed_any = False
    
    # 각 백테스팅 예약 처리
    for date_range in pending_backtests:
        start_date = date_range["start_date"]
        end_date = date_range["end_date"]
        original_text = date_range["original_text"]
        
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 백테스팅 실행: {start_date} ~ {end_date} (원본: {original_text})")
        
        try:
            # 백테스팅 실행 - API 관리자 사용 여부에 따라 다른 방식으로 호출
            if gemini_api_manager and worker_id:
                # 병렬 처리 시 워커 ID 전달
                result = await backtest_recommendation(
                    page_id=page_id, 
                    start_date=start_date,
                    end_date=end_date,
                    investment_amount=1000000,  # 기본 투자금액
                    worker_id=worker_id,
                    notion_api_manager=notion_api_manager,
                    gemini_api_manager=gemini_api_manager
                )
            else:
                # 기존 방식
                result = await backtest_recommendation(
                    page_id=page_id, 
                    start_date=start_date,
                    end_date=end_date,
                    investment_amount=1000000  # 기본 투자금액
                )
            
            if result.get("status") == "success":
                processed_any = True
                logger.info(f"{log_prefix}백테스팅 완료: {original_text}")
            else:
                logger.error(f"{log_prefix}백테스팅 실패: {result.get('message')}")
        except Exception as e:
            logger.error(f"{log_prefix}백테스팅 실행 중 오류: {str(e)}")
    
    # 하나라도 처리되었으면 예약 필드 비우기
    if processed_any:
        await clear_schedule_text(
            page_id=page_id,
            notion_api_manager=notion_api_manager
        )
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 모든 예약 처리 완료, 예약 필드 비움")
    else:
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 처리할 백테스팅 예약이 없거나 모두 실패했습니다.")

async def process_stock_recommendation(
    page_id: str, 
    agent_name: str, 
    worker_id: str = None,
    notion_api_manager = None,
    gemini_api_manager = None
) -> None:
    """
    종목 추천 처리를 실행합니다.
    
    Args:
        page_id (str): 에이전트 페이지 ID
        agent_name (str): 에이전트 이름
        worker_id (str): 워커 ID (병렬 처리용)
        notion_api_manager: 노션 API 관리자 (선택 사항)
        gemini_api_manager: Gemini API 관리자 (선택 사항)
    """
    # 로그 접두어 (워커 ID가 있으면 포함)
    log_prefix = f"[{worker_id}] " if worker_id else ""
    
    try:
        # 에이전트 로드
        from investment_agent import InvestmentAgent
        
        if notion_api_manager:
            # API 관리자를 통해 로드 (커스텀 메서드 호출이 필요)
            agent = await InvestmentAgent.load_from_notion_with_manager(
                page_id, notion_api_manager)
        else:
            # 기존 방식
            agent = await InvestmentAgent.load_from_notion(page_id)
        
        if not agent:
            logger.error(f"{log_prefix}에이전트를 찾을 수 없습니다: {page_id}")
            return
        
        # 현재 날짜
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 최근 보고서 검색
        reports = await find_relevant_reports(
            agent=agent,
            max_reports=30,  # 최대 30개 보고서 가져오기
            worker_id=worker_id,
            notion_api_manager=notion_api_manager,
            gemini_api_manager=gemini_api_manager
        )
        
        if not reports:
            logger.warning(f"{log_prefix}에이전트 '{agent_name}'의 조건에 맞는 최근 보고서를 찾을 수 없습니다.")
            return
        
        # 투자 기간은 기본 7일로 설정
        investment_period = 7
        
        # 보고서 분석 및 종목 추천
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 종목 추천 분석 시작 (보고서 {len(reports)}개 사용)")
        
        recommendation_result = await analyze_reports_with_llm(
            reports=reports,
            agent=agent,
            max_stocks=5,
            investment_period=investment_period,
            worker_id=worker_id,
            notion_api_manager=notion_api_manager,
            gemini_api_manager=gemini_api_manager
        )
        
        # 추천 종목 수 확인
        recommended_stocks = recommendation_result.get("recommended_stocks", [])
        if not recommended_stocks:
            logger.warning(f"{log_prefix}에이전트 '{agent_name}'의 추천 종목이 없습니다.")
            return
        
        # 노션 DB에 추천 기록 저장
        from notion_utils import create_recommendation_record
        
        # 제목 형식 설정 (n종목추천)
        title_prefix = f"{len(recommended_stocks)}종목추천"
        
        # 추천 기록 저장
        if notion_api_manager:
            # API 관리자를 통해 저장 (커스텀 함수 필요)
            await create_recommendation_record_with_manager(
                agent_page_id=page_id,
                recommendations=recommendation_result,
                investment_period=investment_period,
                title_prefix=title_prefix,  # 커스텀 제목 형식 전달
                notion_api_manager=notion_api_manager
            )
        else:
            # 기존 방식
            await create_recommendation_record(
                agent_page_id=page_id,
                recommendations=recommendation_result,
                investment_period=investment_period,
                title_prefix=title_prefix  # 커스텀 제목 형식 전달
            )
        
        logger.info(f"{log_prefix}에이전트 '{agent_name}'의 종목 추천 처리 완료: {len(recommended_stocks)}개 종목")
        
    except Exception as e:
        logger.error(f"{log_prefix}종목 추천 처리 중 오류: {str(e)}")
        raise  # 상위 함수에서 처리할 수 있도록 예외 다시 발생

async def clear_schedule_text(page_id: str, notion_api_manager=None) -> bool:
    """
    예약 필드를 완전히 비웁니다.
    
    Args:
        page_id (str): 에이전트 페이지 ID
        notion_api_manager: 노션 API 관리자 (선택 사항)
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # 빈 예약 필드로 업데이트
        properties = {
            "백테스팅 예약": {
                "rich_text": []  # 빈 배열로 설정하여 필드 비우기
            }
        }
        
        logger.info(f"예약 필드 비우기: 페이지 ID {page_id}")
        
        # API 관리자 사용 여부에 따라 다른 방식으로 호출
        if notion_api_manager:
            return await notion_api_manager.update_notion_page(page_id, properties)
        else:
            return await update_notion_page(page_id, properties)
    
    except Exception as e:
        logger.error(f"예약 필드 비우기 중 오류: {str(e)}")
        return False

async def run_auto_recommendations():
    """
    매일 정해진 시간에 '20개'와 '3개' 에이전트의 종목 추천을 자동으로 실행합니다.
    종목추천 체크박스 상태와 관계없이 실행됩니다.
    """
    logger.info("===== 자동 종목 추천 실행 시작 =====")

    try:
        from notion_utils import find_agent_by_name

        # 처리할 에이전트 목록
        agent_names = ['20개', '3개']

        for agent_name in agent_names:
            try:
                logger.info(f"에이전트 '{agent_name}' 처리 시작")

                # 에이전트 찾기
                page_id = await find_agent_by_name(agent_name)
                if not page_id:
                    logger.warning(f"에이전트 '{agent_name}'를 찾을 수 없습니다.")
                    continue

                # 종목 추천 실행 (기존 max_stocks=5 사용)
                await process_stock_recommendation(
                    page_id=page_id,
                    agent_name=agent_name
                )

                logger.info(f"에이전트 '{agent_name}'의 종목 추천 완료")

            except Exception as e:
                logger.error(f"에이전트 '{agent_name}' 처리 중 오류: {str(e)}")
                continue

        logger.info("===== 자동 종목 추천 실행 완료 =====")

    except Exception as e:
        logger.error(f"자동 종목 추천 실행 중 오류: {str(e)}")

def run_auto_recommendations_sync():
    """동기 함수 래퍼 (APScheduler용)"""
    import asyncio
    asyncio.run(run_auto_recommendations())

async def check_backtest_schedules() -> None:
    """
    Notion DB에서 백테스팅 예약이 있거나 종목 추천이 활성화된 에이전트를 확인하고 처리합니다.
    (병렬 처리 시스템으로 대체되어 더 이상 직접 호출되지 않음)
    """
    try:
        logger.info("백테스팅 예약 및 종목 추천 확인 시작")

        from notion_utils import query_notion_database

        # 에이전트 DB 쿼리
        agents = await query_notion_database(NOTION_AGENT_DB_ID)

        logger.info(f"총 {len(agents)}개 에이전트 확인 중")

        for agent in agents:
            await process_agent_schedules(agent)

        logger.info(f"백테스팅 예약 및 종목 추천 확인 완료: {len(agents)}개 에이전트 처리")
    except Exception as e:
        logger.error(f"백테스팅 예약 및 종목 추천 확인 중 오류: {str(e)}")

# 스케줄러 인스턴스 생성 - BackgroundScheduler 사용
scheduler = BackgroundScheduler(
    daemon=False,  # 데몬 모드 비활성화 - 중요한 수정
    job_defaults={
        'misfire_grace_time': 120,  # 작업 지연 허용 시간 증가(초)
        'coalesce': True,           # 누락된 작업 하나로 통합
        'max_instances': 1          # 동일 작업의 최대 인스턴스 수 제한
    }
)

def start_scheduler():
    """백테스팅 스케줄러를 시작합니다."""
    # 새벽 1시부터 7시를 제외한 다른 시간들에서 5분 단위로 실행
    # 10, 15, 20, ... 55분에 실행 (5분 단위로)
    
    # 1시부터 7시까지 제외하는 시간 제한 설정
    hour_restriction = "8-23,0"  # 8시부터 23시까지, 그리고 0시
    
    # 10분부터 55분까지 5분 단위로 스케줄 추가 - 작업 ID 변경
    scheduler.add_job(
        run_backtest_scheduler,
        'cron',
        hour=hour_restriction,
        minute='10,15,20,25,30,35,40,45,50,55',
        id='backtest_scheduler_main',  # 작업 ID 변경
        replace_existing=True          # 기존 작업 대체
    )

    # 자동 종목 추천 스케줄러 추가 (매일 7:30, 12:30, 19:30)
    scheduler.add_job(
        run_auto_recommendations_sync,
        'cron',
        hour='7,12,19',  # 7시, 12시, 19시
        minute='30',      # 30분
        id='auto_recommendation_scheduler',
        replace_existing=True
    )

    # 스케줄러 시작
    scheduler.start()
    logger.info("백테스팅 스케줄러 시작됨 (새벽 1시-7시 제외, 매 시간 10-55분 5분 단위로 실행)")
    logger.info("자동 종목 추천 스케줄러 시작됨 (매일 07:30, 12:30, 19:30 실행)")

    # 스케줄러 활성화 상태 확인을 위해 즉시 한 번 실행
    logger.info("스케줄러 상태 확인을 위한 즉시 실행")
    run_backtest_scheduler()
# 아래는 API 관리자를 통한 함수 호출을 위한 래퍼 함수들

async def create_recommendation_record_with_manager(
    agent_page_id: str, 
    recommendations: Dict[str, Any], 
    investment_period: int, 
    title_prefix: str = None,
    notion_api_manager = None
) -> bool:
    """
    API 관리자를 통해 추천 기록을 생성합니다.
    """
    from notion_utils import create_recommendation_record
    from datetime import timedelta
    
    try:
        # 추천 종목 및 비중
        stock_names = []
        if "recommended_stocks" in recommendations:
            for stock in recommendations["recommended_stocks"]:
                if "name" in stock and stock["name"]:
                    stock_names.append(stock["name"])
                    
        # 현재 날짜 및 예상 종료일
        current_date = datetime.now()
        end_date = current_date + timedelta(days=investment_period)
        
        # 투자 비중 텍스트
        weights = "균등 비중"  # 기본값
        
        # 제목 설정 (종목추천 형식 또는 기본 형식)
        if title_prefix:
            # 타이틀에 공백이 있는지 확인하고 적절하게 처리
            title = f"{title_prefix} {current_date.strftime('%Y-%m-%d')}"
        else:
            num_stocks = len(stock_names) if stock_names else 0
            title = f"{num_stocks}종목추천 {current_date.strftime('%Y-%m-%d')}"
        
        logger.info(f"저장할 추천 기록 제목: {title}")
        
        # 추천 기록 생성
        recommendation_data = {
            "title": title,
            "agent_page_id": agent_page_id,
            "start_date": current_date,
            "end_date": end_date,
            "stocks": stock_names,
            "weights": weights,
            "recommendation_type": "신규 추천"  # 새로운 필드 추가
        }
        
        # 이 부분이 API 관리자를 사용하도록 수정 필요 - 별도 구현 필요
        # 현재는 기존 함수를 호출
        # 수정 전: return await create_recommendation_record(recommendation_data)
        # 수정 후:
        return await create_recommendation_record(
            agent_page_id=agent_page_id,
            recommendations=recommendations,  # 이 인자 추가
            investment_period=investment_period  # 이 인자 추가
        )
        
    except Exception as e:
        logger.error(f"추천 기록 저장 중 오류: {str(e)}")
        return False