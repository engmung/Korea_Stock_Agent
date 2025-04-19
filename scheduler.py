import logging
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Dict, Any, List, Optional

from notion_utils import (
    query_notion_database, 
    update_notion_page, 
    check_script_exists, 
    create_script_report_page,
    reset_all_channels,
    REFERENCE_DB_ID, 
    SCRIPT_DB_ID
)
from youtube_utils import process_channel_url, get_video_transcript, parse_upload_date
from gemini_analyzer import analyze_script_with_gemini

logger = logging.getLogger(__name__)

# 글로벌 스케줄러 인스턴스
scheduler = None

async def process_channel(page: Dict[str, Any]) -> bool:
    """특정 채널 페이지를 처리하여 새 스크립트를 생성합니다."""
    try:
        # 페이지 속성 가져오기
        properties = page.get("properties", {})
        page_id = page.get("id")
        
        # 활성화 상태 확인
        is_active = False
        active_property = properties.get("활성화", {})
        if "checkbox" in active_property:
            is_active = active_property["checkbox"]
        
        # 활성화되지 않은 항목은 건너뛰기
        if not is_active:
            logger.info("비활성화된 채널입니다. 스킵합니다.")
            return False
        
        # 제목(키워드) 가져오기
        keyword = ""
        title_property = properties.get("제목", {})
        if "title" in title_property and title_property["title"]:
            keyword = title_property["title"][0]["plain_text"].strip()
        
        # URL 가져오기
        channel_url = ""
        url_property = properties.get("URL", {})
        if "url" in url_property:
            channel_url = url_property["url"]
        
        # 채널명 가져오기
        channel_name = "기타"
        channel_property = properties.get("채널명", {})
        if "select" in channel_property and channel_property["select"]:
            channel_name = channel_property["select"]["name"]
            
        # 콘텐츠 유형 가져오기
        content_type = "단독 진행"
        content_type_property = properties.get("콘텐츠 유형", {})
        if "select" in content_type_property and content_type_property["select"]:
            content_type = content_type_property["select"]["name"]
            
        # 투자 스타일 가져오기
        investment_style = []
        style_property = properties.get("투자 스타일", {})
        if "multi_select" in style_property:
            investment_style = [item["name"] for item in style_property["multi_select"]]
        
        if not channel_url or not keyword:
            logger.warning(f"채널 URL 또는 키워드가 없습니다. 스킵합니다.")
            return False
        
        # 유튜브 채널 URL이 아니면 스킵
        if not "youtube.com/@" in channel_url:
            logger.warning(f"유효한 YouTube 채널 URL이 아닙니다: {channel_url}")
            return False
        
        logger.info(f"Processing channel: {channel_url} with keyword: {keyword}")
        
        # 채널에서 키워드가 포함된 최신 영상 찾기
        latest_video = await process_channel_url(channel_url, keyword)
        
        if not latest_video:
            logger.warning(f"채널에서 키워드가 포함된 영상을 찾을 수 없습니다: {channel_url}")
            return False

        # 라이브 예정(Upcoming) 또는 라이브 중(Live) 영상인 경우 처리하지 않고 활성화 상태 유지
        if latest_video.get("is_upcoming", False) or latest_video.get("is_live", False):
            status = "라이브 예정" if latest_video.get("is_upcoming", False) else "라이브 중"
            logger.info(f"{status} 영상입니다: {latest_video['title']}. 활성화 상태 유지하고 다음에 다시 확인합니다.")
            return False
        
        # 이미 스크립트가 있는지 확인
        if await check_script_exists(latest_video["url"]):
            logger.info(f"이미 스크립트가 존재합니다: {latest_video['title']}")
            
            # 스크립트가 이미 존재하면 활성화 상태를 비활성화로 변경
            await update_notion_page(page_id, {
                "활성화": {"checkbox": False}
            })
            logger.info(f"채널 {channel_name}의 활성화 상태를 비활성화로 변경했습니다.")
            
            return True
        
        # 스크립트 가져오기
        script = await get_video_transcript(latest_video["video_id"])
        
        # 스크립트가 없거나 에러 메시지를 반환한 경우
        if not script or script.startswith("스크립트를 가져올 수 없습니다"):
            logger.warning(f"스크립트를 가져올 수 없습니다: {latest_video['title']}")
            logger.info(f"자막이 비활성화되었거나 추출할 수 없는 영상입니다. 채널 '{channel_name}'을 활성화 상태로 유지합니다.")
            return False
        
        # 스크립트가 있는 경우 페이지 생성
        # 영상 날짜 파싱 - 정확한 업로드 날짜로 변환
        upload_date_datetime = parse_upload_date(latest_video.get("upload_date", ""))
        upload_date_iso = upload_date_datetime.isoformat()
        
        # 영상 날짜 - UTC로 변환
        utc_upload_date = upload_date_datetime - timedelta(hours=9)

        # 스크립트 DB에 새 페이지 생성 (속성 설정)
        properties = {
            # 제목은 참고용 DB의 키워드만 사용
            "제목": {
                "title": [
                    {
                        "text": {
                            "content": keyword
                        }
                    }
                ]
            },
            # URL 속성 (기존의 원본 영상)
            "URL": {
                "url": latest_video["url"]
            },
            # 영상 날짜 - UTC 기준으로 저장
            "영상 날짜": {
                "date": {
                    "start": utc_upload_date.isoformat()
                }
            },
            # 채널명 속성
            "채널명": {
                "select": {
                    "name": channel_name
                }
            },
            # 영상 길이 속성 추가
            "영상 길이": {
                "rich_text": [
                    {
                        "text": {
                            "content": latest_video.get("video_length", "알 수 없음")
                        }
                    }
                ]
            },
            # 인용 횟수 초기화
            "인용 횟수": {
                "number": 0
            },
            # 출연자 정보
            "출연자": {
                "multi_select": []  # 초기에는 비어있음, 필요시 채울 수 있음
            }
        }
        
        # 디버깅 정보 로깅
        logger.info(f"Creating page for video: {latest_video['title']}")
        logger.info(f"Keyword: {keyword}, Channel: {channel_name}")
        logger.info(f"Upload date: {upload_date_datetime.strftime('%Y-%m-%d')}")
        
        try:
            # Gemini로 스크립트 분석 - 스크립트는 분석에만 사용하고 결과에는 포함하지 않음
            logger.info(f"Gemini API로 스크립트 분석 시작: {latest_video['title']}")
            analysis = await analyze_script_with_gemini(script, latest_video['title'], channel_name)
            
            # 분석 결과만 사용 (원본 스크립트 제외)
            combined_content = analysis
            logger.info("AI 분석 보고서가 성공적으로 생성되었습니다.")
        except Exception as e:
            logger.error(f"AI 분석 중 오류 발생: {str(e)}")
            # 분석 실패 시 간단한 오류 메시지 저장 (스크립트 포함하지 않음)
            combined_content = f"# AI 분석 보고서\n\n## 분석 오류\n\n분석 과정에서 오류가 발생했습니다: {str(e)}"
            logger.warning("AI 분석에 실패했습니다. 오류 메시지를 저장합니다.")
        
        # 수정된 내용으로 페이지 생성
        script_page = await create_script_report_page(SCRIPT_DB_ID, properties, combined_content)
        
        if script_page:
            logger.info(f"스크립트+보고서 페이지 생성 완료: {keyword}")
            
            # 스크립트 생성 성공 시 채널 비활성화
            await update_notion_page(page_id, {
                "활성화": {"checkbox": False}
            })
            logger.info(f"채널 {channel_name}의 활성화 상태를 비활성화로 변경했습니다.")
            
            return True
        else:
            logger.error(f"스크립트+보고서 페이지 생성 실패: {keyword}")
            # 페이지 생성에 실패한 경우 활성화 상태 유지
            logger.info(f"스크립트 생성 실패로 채널 '{channel_name}'을 활성화 상태로 유지합니다.")
            return False
        
    except Exception as e:
        logger.error(f"채널 처리 중 오류: {str(e)}")
        return False

async def process_channels_by_setting() -> None:
    """모든 활성화된 채널을 처리합니다."""
    logger.info("채널 처리 시작")
    
    try:
        # 참고용 DB의 모든 채널 가져오기
        reference_pages = await query_notion_database(REFERENCE_DB_ID)
        logger.info(f"참고용 DB에서 {len(reference_pages)}개의 채널을 가져왔습니다.")
        
        # 활성화된 채널만 선택
        active_channels = []
        for page in reference_pages:
            properties = page.get("properties", {})
            
            # 활성화 상태 확인
            is_active = False
            active_property = properties.get("활성화", {})
            if "checkbox" in active_property:
                is_active = active_property["checkbox"]
            
            if is_active:
                active_channels.append(page)
        
        logger.info(f"처리할 활성화된 채널 {len(active_channels)}개를 찾았습니다.")
        
        if not active_channels:
            logger.info("처리할 활성화된 채널이 없습니다.")
            return
        
        # 채널 처리 - API 제한 고려하여 순차적으로 처리
        success_count = 0
        
        for index, channel_page in enumerate(active_channels):
            try:
                channel_name = "Unknown"
                properties = channel_page.get("properties", {})
                if "채널명" in properties and "select" in properties["채널명"] and properties["채널명"]["select"]:
                    channel_name = properties["채널명"]["select"]["name"]
                    
                logger.info(f"채널 처리 시작 ({index+1}/{len(active_channels)}): {channel_name}")
                success = await process_channel(channel_page)
                
                if success:
                    success_count += 1
                    logger.info(f"채널 처리 성공: {channel_name}")
                else:
                    logger.warning(f"채널 처리 실패: {channel_name}")
                    
                # 다음 채널 처리 전 1분 대기 (API 제한 1분당 2개 고려)
                # 마지막 항목이 아니면 대기
                if index < len(active_channels) - 1:
                    logger.info(f"API 제한 준수를 위해 10초초 대기 중...")
                    await asyncio.sleep(10)  # 1분 대기
                    
            except Exception as e:
                logger.error(f"채널 처리 중 예외 발생: {str(e)}")
                # 다음 채널 처리 전 1분 대기
                if index < len(active_channels) - 1:
                    logger.info(f"오류 후 API 제한 준수를 위해 10초초 대기 중...")
                    await asyncio.sleep(10)  # 1분 대기
        
        logger.info(f"처리 완료: {success_count}/{len(active_channels)} 채널 성공")
    except Exception as e:
        logger.error(f"process_channels_by_setting 실행 중 오류: {str(e)}")

async def reset_channels_daily() -> None:
    """매일 새벽 4시에 모든 채널을 활성화 상태로 초기화합니다."""
    logger.info("모든 채널 활성화 작업 시작")
    success = await reset_all_channels()
    
    if success:
        logger.info("모든 채널이 성공적으로 활성화되었습니다.")
    else:
        logger.error("일부 또는 모든 채널의 활성화에 실패했습니다.")

def setup_scheduler() -> AsyncIOScheduler:
    """스케줄러를 설정하고 작업을 예약합니다."""
    global scheduler
    
    if scheduler is not None:
        scheduler.shutdown()
    
    scheduler = AsyncIOScheduler()
    
    # 새벽 4시에 모든 채널 초기화
    scheduler.add_job(
        reset_channels_daily,
        CronTrigger(hour=4, minute=0),
        id="reset_channels_daily",
        replace_existing=True
    )
    
    # 매시간 정각에 작업 실행 (0-23시)
    for hour in range(24):
        scheduler.add_job(
            process_channels_by_setting,
            CronTrigger(hour=hour, minute=0),
            id=f"process_channels_{hour}",
            replace_existing=True
        )
    
    # 스케줄러 시작
    scheduler.start()
    logger.info("Scheduler has been set up and is running.")
    
    return scheduler

async def simulate_scheduler_at_time(time_setting: int, simulate_only: bool = True) -> Dict[str, Any]:
    """특정 시간 설정에 대한 작업 시뮬레이션"""
    logger.info(f"시간 설정 {time_setting}에 대한 작업 시뮬레이션")
    
    try:
        # 참고용 DB의 모든 채널 조회
        reference_pages = await query_notion_database(REFERENCE_DB_ID)
        logger.info(f"테스트: {len(reference_pages)}개의 채널을 가져왔습니다.")
        
        # 활성화된 채널 찾기
        active_channels = []
        for page in reference_pages:
            properties = page.get("properties", {})
            
            # 활성화 상태 확인
            is_active = False
            active_property = properties.get("활성화", {})
            if "checkbox" in active_property:
                is_active = active_property["checkbox"]
            
            if is_active:
                # 채널명과 키워드 가져오기
                channel_name = "기타"
                if "채널명" in properties and "select" in properties["채널명"] and properties["채널명"]["select"]:
                    channel_name = properties["채널명"]["select"]["name"]
                
                keyword = ""
                if "제목" in properties and "title" in properties["제목"] and properties["제목"]["title"]:
                    keyword = properties["제목"]["title"][0]["plain_text"].strip()
                
                active_channels.append({
                    "channel_name": channel_name,
                    "keyword": keyword,
                    "page_id": page.get("id"),
                    "page": page
                })
        
        logger.info(f"활성화된 채널 {len(active_channels)}개 찾음")
        
        if not simulate_only and active_channels:
            # 실제 실행 모드
            logger.info("실제 채널 처리 실행 시작")
            for i, channel in enumerate(active_channels):
                logger.info(f"채널 처리 중 ({i+1}/{len(active_channels)}): {channel['channel_name']}")
                await process_channel(channel["page"])
                
                # 마지막 항목이 아니면 API 제한을 위해 대기
                if i < len(active_channels) - 1:
                    logger.info("API 제한을 위해 10초 대기")
                    await asyncio.sleep(10)
            
            logger.info("모든 채널 처리 완료")
        
        return {
            "time_setting": time_setting,
            "active_channels": [
                {
                    "channel_name": c["channel_name"],
                    "keyword": c["keyword"]
                } for c in active_channels
            ],
            "total_active": len(active_channels),
            "simulate_only": simulate_only
        }
    except Exception as e:
        logger.error(f"시뮬레이션 중 오류 발생: {str(e)}")
        return {
            "time_setting": time_setting,
            "error": str(e),
            "simulate_only": simulate_only
        }