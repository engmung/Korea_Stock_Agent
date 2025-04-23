import os
import re
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from functools import wraps
import concurrent.futures

from notion_utils import query_notion_database, update_notion_page
from performance_evaluator import backtest_recommendation
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
    별도의 스레드에서 이벤트 루프를 생성하고 비동기 함수를 실행합니다.
    """
    global is_backtest_running
    
    # 이미 실행 중이면 스킵
    if is_backtest_running:
        logger.info("이전 백테스팅이 아직 실행 중입니다. 이번 예약 체크는 건너뜁니다.")
        return
        
    try:
        # 실행 중 플래그 설정
        is_backtest_running = True
        logger.info("백테스팅 스케줄러 실행 - 새 스레드에서 비동기 함수 호출")
        
        # 새 스레드에서 이벤트 루프 실행
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread, check_backtest_schedules)
            try:
                # 결과 대기 (예외가 있으면 다시 발생)
                future.result()
            except Exception as e:
                logger.error(f"백테스팅 스케줄러 실행 중 오류: {str(e)}")
    finally:
        # 실행 완료 후 플래그 해제
        is_backtest_running = False

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

async def parse_date_ranges(schedule_text: str) -> List[Dict[str, str]]:
    """
    예약 문자열에서 날짜 범위를 파싱합니다.
    형식: '0522~0526, 0601~0605' 또는 '0522~0526'
    
    Returns:
        List[Dict[str, str]]: 시작일과 종료일 딕셔너리 리스트
    """
    date_ranges = []
    
    # 쉼표로 구분된 항목들을 분리
    if not schedule_text:
        return []
        
    # 대괄호 제거 (있는 경우)
    cleaned_text = schedule_text.strip()
    if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
        cleaned_text = cleaned_text[1:-1]
    
    # 쉼표로 분리
    range_items = [item.strip() for item in cleaned_text.split(',')]
    
    current_year = datetime.now().year
    
    for range_item in range_items:
        # 날짜 범위 파싱 (형식: MMDD~MMDD)
        match = re.match(r'(\d{4})~(\d{4})', range_item)
        
        if match:
            start_mmdd, end_mmdd = match.groups()
            
            # MMDD 형식을 YYYY-MM-DD 형식으로 변환
            start_mm, start_dd = int(start_mmdd[:2]), int(start_mmdd[2:])
            end_mm, end_dd = int(end_mmdd[:2]), int(end_mmdd[2:])
            
            # 날짜 유효성 검증 및 변환
            try:
                start_date = f"{current_year}-{start_mm:02d}-{start_dd:02d}"
                end_date = f"{current_year}-{end_mm:02d}-{end_dd:02d}"
                
                # 시작일이 종료일보다 나중인 경우 처리
                if start_date > end_date:
                    # 종료일이 다음 해로 넘어간 경우
                    end_date = f"{current_year + 1}-{end_mm:02d}-{end_dd:02d}"
                
                date_ranges.append({
                    "start_date": start_date,
                    "end_date": end_date,
                    "original_text": range_item
                })
            except ValueError:
                logger.error(f"유효하지 않은 날짜 형식: {range_item}")
    
    return date_ranges

async def update_schedule_text(page_id: str, current_schedule: str, completed_item: str) -> bool:
    """
    완료된 백테스팅 일정을 예약 문자열에서 제거합니다.
    
    Args:
        page_id (str): 에이전트 페이지 ID
        current_schedule (str): 현재 예약 문자열
        completed_item (str): 완료된 예약 항목
        
    Returns:
        bool: 업데이트 성공 여부
    """
    try:
        # 대괄호 제거 (있는 경우)
        cleaned_text = current_schedule.strip()
        if cleaned_text.startswith('[') and cleaned_text.endswith(']'):
            cleaned_text = cleaned_text[1:-1]
        
        # 쉼표로 분리하여 각 항목 추출
        items = [item.strip() for item in cleaned_text.split(',') if item.strip()]
        
        # 완료된 항목 정확히 찾아 제거
        if completed_item in items:
            items.remove(completed_item)
        else:
            # 정확히 일치하는 항목이 없으면 다른 방법으로 시도
            # MMDD~MMDD 형식을 기준으로 비교
            match = re.match(r'(\d{4})~(\d{4})', completed_item)
            if match:
                start_mmdd, end_mmdd = match.groups()
                # 각 항목을 순회하며 동일한 날짜 범위를 가진 항목 찾기
                for item in items[:]:  # 복사본으로 순회
                    item_match = re.match(r'(\d{4})~(\d{4})', item)
                    if item_match and item_match.groups() == (start_mmdd, end_mmdd):
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
        return await update_notion_page(page_id, properties)
    
    except Exception as e:
        logger.error(f"예약 문자열 업데이트 중 오류: {str(e)}")
        return False

async def process_agent_schedules(agent: Dict[str, Any]) -> None:
    """
    에이전트의 백테스팅 예약을 처리합니다.
    
    Args:
        agent (Dict[str, Any]): 에이전트 정보
    """
    page_id = agent.get("id")
    properties = agent.get("properties", {})
    
    # '백테스팅 예약' 속성 확인
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
    
    # 에이전트 이름 가져오기
    agent_name = "Unknown"
    if "에이전트명" in properties and "title" in properties["에이전트명"]:
        title_objs = properties["에이전트명"]["title"]
        if title_objs and len(title_objs) > 0:
            agent_name = title_objs[0].get("plain_text", "Unknown")
    
    logger.info(f"에이전트 '{agent_name}' 예약 처리: '{schedule_text}'")
    
    # 날짜 범위 파싱
    date_ranges = await parse_date_ranges(schedule_text)
    
    # 현재 날짜
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 처리할 예약 개수 확인
    pending_backtests = [dr for dr in date_ranges if dr["end_date"] <= today]
    if pending_backtests:
        logger.info(f"에이전트 '{agent_name}'의 처리 예정 백테스팅: {len(pending_backtests)}개")
    
    for date_range in date_ranges:
        start_date = date_range["start_date"]
        end_date = date_range["end_date"]
        original_text = date_range["original_text"]
        
        # 종료일이 오늘이거나 지난 경우만 백테스팅 실행
        if end_date <= today:
            logger.info(f"에이전트 '{agent_name}'의 백테스팅 실행: {start_date} ~ {end_date} (원본: {original_text})")
            
            try:
                # 백테스팅 실행
                result = await backtest_recommendation(
                    page_id=page_id, 
                    start_date=start_date,
                    end_date=end_date,
                    investment_amount=1000000  # 기본 투자금액
                )
                
                if result.get("status") == "success":
                    # 성공적으로 백테스팅이 완료되면 예약에서 제거
                    updated = await update_schedule_text(page_id, schedule_text, original_text)
                    if updated:
                        logger.info(f"백테스팅 완료 및 예약 항목 제거 성공: {original_text}")
                    else:
                        logger.warning(f"백테스팅은 완료되었으나 예약 항목 제거 실패: {original_text}")
                else:
                    logger.error(f"백테스팅 실패: {result.get('message')}")
            except Exception as e:
                logger.error(f"백테스팅 실행 중 오류: {str(e)}")

async def check_backtest_schedules() -> None:
    """
    Notion DB에서 백테스팅 예약이 있는 에이전트를 확인하고 처리합니다.
    """
    try:
        logger.info("백테스팅 예약 확인 시작")
        
        # 에이전트 DB 쿼리
        agents = await query_notion_database(NOTION_AGENT_DB_ID)
        
        logger.info(f"총 {len(agents)}개 에이전트 확인 중")
        
        for agent in agents:
            await process_agent_schedules(agent)
            
        logger.info(f"백테스팅 예약 확인 완료: {len(agents)}개 에이전트 처리")
    except Exception as e:
        logger.error(f"백테스팅 예약 확인 중 오류: {str(e)}")

# 스케줄러 인스턴스 생성 - BackgroundScheduler 사용
scheduler = BackgroundScheduler()

def start_scheduler():
    """백테스팅 스케줄러를 시작합니다."""
    # 새벽 1시부터 7시를 제외한 다른 시간들에서 5분 단위로 실행
    # 10, 15, 20, ... 55분에 실행 (5분 단위로)
    
    # 1시부터 7시까지 제외하는 시간 제한 설정
    hour_restriction = "8-23,0"  # 8시부터 23시까지, 그리고 0시
    
    # 10분부터 55분까지 5분 단위로 스케줄 추가
    scheduler.add_job(
        run_backtest_scheduler,
        'cron',
        hour=hour_restriction,
        minute='10,15,20,25,30,35,40,45,50,55',
        id='backtest_scheduler_5min'
    )
    
    # 스케줄러 시작
    scheduler.start()
    logger.info("백테스팅 스케줄러 시작됨 (새벽 1시-7시 제외, 매 시간 10-55분 5분 단위로 실행)")

    # 스케줄러 활성화 상태 확인을 위해 즉시 한 번 실행
    logger.info("스케줄러 상태 확인을 위한 즉시 실행")
    run_backtest_scheduler()