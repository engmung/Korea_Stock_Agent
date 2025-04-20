import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor

from notion_utils import (
    query_notion_database,
    check_script_exists,
    create_script_report_page,
    REFERENCE_DB_ID,
    SCRIPT_DB_ID
)
from youtube_api_utils import get_channel_id_from_url, get_videos_by_channel_id, get_video_details
from youtube_transcript_api import YouTubeTranscriptApi
from gemini_analyzer import analyze_script_with_gemini

# 환경 변수 로드
load_dotenv()

# YouTube API 키 설정
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    print("YOUTUBE_API_KEY 환경 변수가 설정되지 않았습니다.")
    raise ValueError("YOUTUBE_API_KEY 환경 변수가 필요합니다.")

# YouTube API 클라이언트 생성
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# 동시 처리를 위한 세마포어 - 최대 3개 영상 동시 처리
VIDEO_SEMAPHORE = asyncio.Semaphore(3)

async def get_video_transcript(video_id: str, max_retries: int = 3) -> str:
    """비디오 ID로부터 자막을 가져옵니다."""
    print(f"영상 ID에 대한 자막 가져오기: {video_id}")
    
    for attempt in range(max_retries):
        try:
            # 한국어 자막 시도
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["ko"])
            print(f"{len(transcript_list)}개 항목의 한국어 자막을 찾았습니다")
            return " ".join([entry["text"] for entry in transcript_list])
        except Exception as e:
            print(f"한국어 자막 오류 (시도 {attempt+1}/{max_retries}): {str(e)}")
            
            try:
                # 자동 언어 감지 시도
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                print(f"자동 감지 언어로 자막을 찾았습니다")
                return " ".join([entry["text"] for entry in transcript_list])
            except Exception as e2:
                print(f"자동 감지 자막 오류 (시도 {attempt+1}/{max_retries}): {str(e2)}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return f"스크립트를 가져올 수 없습니다: {str(e2)}"
    
    return "스크립트를 가져올 수 없습니다: 최대 재시도 횟수 초과"

async def get_recent_videos(channel_id: str, max_results: int = 500) -> List[Dict[str, Any]]:
    """특정 채널의 최근 영상 목록 가져오기"""
    videos = []
    next_page_token = None
    
    while len(videos) < max_results:
        video_batch, next_page_token = await get_videos_by_channel_id(
            channel_id=channel_id,
            max_results=min(50, max_results - len(videos)),
            page_token=next_page_token
        )
        
        videos.extend(video_batch)
        
        if not next_page_token or len(video_batch) == 0:
            break
        
        # API 제한 고려하여 약간의 딜레이 추가
        await asyncio.sleep(0.2)
    
    return videos

async def is_shorts(video_title: str, duration_seconds: int) -> bool:
    """숏츠 영상인지 확인"""
    # 제목에 '#shorts' 또는 '#쇼츠' 포함 확인 (대소문자 구분 없이)
    shorts_keywords = ['#shorts', '#쇼츠', '#short', '#short']
    has_shorts_keyword = any(keyword.lower() in video_title.lower() for keyword in shorts_keywords)
    
    # 5분(300초) 이하 영상
    is_short_duration = duration_seconds <= 300
    
    return has_shorts_keyword or is_short_duration

async def process_video(video: Dict[str, Any], channel_name: str, keyword: str) -> bool:
    """
    개별 비디오를 처리하는 비동기 함수 - 세마포어로 동시 처리 제한
    """
    # 세마포어로 동시 처리 제한
    async with VIDEO_SEMAPHORE:
        print(f"영상 처리 시작: {video['title']}")
        
        try:
            # 스크립트 가져오기
            script = await get_video_transcript(video["video_id"])
            
            # 스크립트가 없거나 에러 메시지를 반환한 경우
            if not script or script.startswith("스크립트를 가져올 수 없습니다"):
                print(f"스크립트를 가져올 수 없습니다: {video['title']}")
                return False
            
            # 영상 날짜 파싱
            try:
                published_datetime = datetime.fromisoformat(video["upload_date"].replace("Z", "+00:00"))
                # UTC 시간 (Notion API 사용)
                utc_published_date = published_datetime
            except Exception as e:
                print(f"날짜 파싱 오류: {str(e)}")
                # 오류 시 현재 시간 사용
                utc_published_date = datetime.now() - timedelta(hours=9)
            
            # 스크립트 분석
            try:
                print(f"Gemini API로 스크립트 분석 시작: {video['title']}")
                analysis = await analyze_script_with_gemini(script, video['title'], channel_name)
                
                # 분석 결과만 사용 (원본 스크립트 제외)
                combined_content = analysis
                print("AI 분석 보고서가 성공적으로 생성되었습니다.")
            except Exception as e:
                print(f"AI 분석 중 오류 발생: {str(e)}")
                # 분석 실패 시 간단한 오류 메시지 저장
                combined_content = f"# AI 분석 보고서\n\n## 분석 오류\n\n분석 과정에서 오류가 발생했습니다: {str(e)}"
                print("AI 분석에 실패했습니다. 오류 메시지를 저장합니다.")
            
            # Notion 페이지 속성 설정
            properties = {
                # 제목은 참고용 DB의 키워드 사용 (중요: 프로그램 이름으로 사용)
                "제목": {
                    "title": [
                        {
                            "text": {
                                "content": keyword
                            }
                        }
                    ]
                },
                # URL 속성
                "URL": {
                    "url": video["url"]
                },
                # 영상 날짜 - UTC 시간 기준
                "영상 날짜": {
                    "date": {
                        "start": utc_published_date.isoformat()
                    }
                },
                # 채널명 속성
                "채널명": {
                    "select": {
                        "name": channel_name
                    }
                },
                # 영상 길이 속성
                "영상 길이": {
                    "rich_text": [
                        {
                            "text": {
                                "content": video.get("video_length", "알 수 없음")
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
                    "multi_select": []  # 초기에는 비어있음
                }
            }
            
            # Notion 페이지 생성
            script_page = await create_script_report_page(SCRIPT_DB_ID, properties, combined_content)
            
            if script_page:
                print(f"스크립트+보고서 페이지 생성 완료: {video['title']}")
                return True
            else:
                print(f"스크립트+보고서 페이지 생성 실패: {video['title']}")
                return False
            
        except Exception as e:
            print(f"영상 처리 중 오류: {str(e)}")
            return False

async def process_channel_historical_data(
    channel_info: Dict[str, Any], 
    videos_per_channel: int = 500, 
    process_limit: int = 20,
    target_programs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """특정 채널의 과거 데이터를 처리합니다."""
    channel_url = channel_info.get("url", "")
    keyword = channel_info.get("keyword", "")
    channel_name = channel_info.get("channel_name", "기타")
    
    # 타깃 프로그램 지정된 경우, 이 채널이 해당 키워드를 포함하는지 확인
    if target_programs and keyword not in target_programs:
        print(f"채널 '{channel_name}'(키워드: {keyword})은 타깃 프로그램에 포함되지 않아 건너뜁니다.")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": True,
            "reason": "not_in_target_programs",
            "success_count": 0
        }
    
    print(f"채널 '{channel_name}' 과거 데이터 처리 시작 (키워드: {keyword})")
    
    if not channel_url or not keyword:
        print("채널 URL 또는 키워드가 없습니다. 건너뜁니다.")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": True,
            "reason": "missing_url_or_keyword",
            "success_count": 0
        }
    
    # 채널 ID 가져오기
    channel_id = await get_channel_id_from_url(channel_url)
    if not channel_id:
        print(f"채널 ID를 가져올 수 없습니다: {channel_url}")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": True,
            "reason": "channel_id_not_found",
            "success_count": 0
        }
    
    print(f"채널 ID 가져오기 성공: {channel_id}")
    
    # 최근 영상 가져오기 (기본값 500개)
    videos = await get_recent_videos(channel_id, videos_per_channel)
    if not videos:
        print(f"채널에서 영상을 찾을 수 없습니다: {channel_name}")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": True,
            "reason": "no_videos_found",
            "success_count": 0
        }
    
    print(f"채널 '{channel_name}'에서 {len(videos)}개 영상을 가져왔습니다.")
    
    # 키워드가 포함된 영상 필터링
    matching_videos = []
    for video in videos:
        if keyword.lower() in video["title"].lower():
            matching_videos.append(video)
    
    print(f"키워드 '{keyword}'와 일치하는 영상 {len(matching_videos)}개를 찾았습니다.")
    
    if not matching_videos:
        print(f"키워드 '{keyword}'와 일치하는 영상이 없습니다.")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": True,
            "reason": "no_matching_videos",
            "success_count": 0
        }
    
    # 비디오 ID 목록 추출하여 상세 정보 가져오기
    video_ids = [video["video_id"] for video in matching_videos]
    video_details = await get_video_details(video_ids)
    
    # 각 영상의 상세 정보 병합
    for video in matching_videos:
        details = next((v for v in video_details if v["video_id"] == video["video_id"]), {})
        video["video_length"] = details.get("video_length", "알 수 없음")
        video["duration_seconds"] = details.get("duration_seconds", 0)
        video["view_count"] = details.get("view_count", "0")
    
    # 숏츠 제외
    non_shorts_videos = []
    shorts_count = 0
    
    for video in matching_videos:
        if await is_shorts(video["title"], video.get("duration_seconds", 0)):
            shorts_count += 1
        else:
            non_shorts_videos.append(video)
    
    print(f"숏츠 영상 {shorts_count}개를 제외했습니다.")
    print(f"숏츠를 제외한 영상 {len(non_shorts_videos)}개를 찾았습니다.")
    
    if not non_shorts_videos:
        print(f"숏츠를 제외하고 처리할 영상이 없습니다.")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": True,
            "reason": "no_non_shorts_videos",
            "success_count": 0
        }
    
    # 이미 처리된 영상 필터링
    new_videos = []
    for video in non_shorts_videos:
        exists = await check_script_exists(video["url"])
        if not exists:
            new_videos.append(video)
    
    print(f"아직 처리되지 않은 새 영상 {len(new_videos)}개를 찾았습니다.")
    
    # 처리 한도 설정 (API 제한 고려)
    videos_to_process = new_videos[:process_limit]
    print(f"처리할 영상 {len(videos_to_process)}개 (최대 {process_limit}개)")
    
    if not videos_to_process:
        print(f"채널 '{channel_name}'에 처리할 새 영상이 없습니다.")
        return {
            "channel_name": channel_name,
            "keyword": keyword,
            "skipped": False,
            "videos_found": len(matching_videos),
            "new_videos": 0,
            "success_count": 0
        }
    
    # 비디오들을 병렬로 처리 (최대 3개씩 동시 처리)
    tasks = []
    for video in videos_to_process:
        # 동일한 API 요청이 집중되지 않도록 약간의 지연 추가
        await asyncio.sleep(0.5)
        tasks.append(process_video(video, channel_name, keyword))
    
    # 모든 태스크 실행하고 결과 기다리기
    results = await asyncio.gather(*tasks)
    
    # 성공 개수 계산
    successful_count = sum(1 for result in results if result)
    
    print(f"채널 '{channel_name}' 과거 데이터 처리 완료: {successful_count}/{len(videos_to_process)} 성공")
    
    return {
        "channel_name": channel_name,
        "keyword": keyword,
        "skipped": False,
        "videos_found": len(matching_videos),
        "new_videos": len(new_videos),
        "success_count": successful_count
    }

async def process_all_channels_historical_data(
    videos_per_channel: int = 500, 
    process_limit_per_channel: int = 20,
    target_programs: Optional[List[str]] = None,
    concurrent_channels: int = 1
) -> Dict[str, Any]:
    """모든 채널의 과거 데이터를 처리합니다."""
    print("모든 채널 과거 데이터 처리 시작")
    
    if target_programs:
        print(f"타깃 프로그램: {', '.join(target_programs)}")
    
    try:
        # 참고용 DB의 모든 채널 가져오기
        reference_pages = await query_notion_database(REFERENCE_DB_ID)
        print(f"참고용 DB에서 {len(reference_pages)}개의 채널을 가져왔습니다.")
        
        if not reference_pages:
            print("참고용 DB에서 채널을 찾을 수 없습니다.")
            return {"status": "error", "message": "참고용 DB에서 채널을 찾을 수 없습니다."}
        
        # 채널 정보 추출
        channels = []
        for page in reference_pages:
            properties = page.get("properties", {})
            
            # 채널 정보 추출
            channel_info = {
                "page_id": page.get("id"),
                "keyword": "",
                "url": "",
                "channel_name": "기타"
            }
            
            # 제목(키워드) 가져오기
            title_property = properties.get("제목", {})
            if "title" in title_property and title_property["title"]:
                channel_info["keyword"] = title_property["title"][0]["plain_text"].strip()
            
            # URL 가져오기
            url_property = properties.get("URL", {})
            if "url" in url_property:
                channel_info["url"] = url_property["url"]
            
            # 채널명 가져오기
            channel_property = properties.get("채널명", {})
            if "select" in channel_property and channel_property["select"]:
                channel_info["channel_name"] = channel_property["select"]["name"]
                
            # URL이 YouTube 채널 URL인 경우만 추가
            if "youtube.com/" in channel_info["url"] and channel_info["keyword"]:
                # 타깃 프로그램이 지정된 경우, 해당 프로그램만 포함
                if not target_programs or channel_info["keyword"] in target_programs:
                    channels.append(channel_info)
        
        # 타깃 프로그램 필터링 후 남은 채널 수
        print(f"처리할 YouTube 채널 {len(channels)}개를 찾았습니다.")
        
        if not channels:
            reason = "처리할 YouTube 채널이 없습니다."
            if target_programs:
                reason += f" 타깃 프로그램({', '.join(target_programs)})과 일치하는 채널이 없습니다."
            
            print(reason)
            return {"status": "warning", "message": reason}
        
        # 결과 저장용 변수
        results = []
        total_success = 0
        
        # 채널 그룹으로 나누기 (concurrent_channels 단위로)
        # 동시 처리할 채널 그룹 생성
        channel_groups = [channels[i:i+concurrent_channels] for i in range(0, len(channels), concurrent_channels)]
        
        # 각 그룹별로 채널 처리 (그룹 내에서는 병렬 처리)
        for group_idx, channel_group in enumerate(channel_groups):
            print(f"채널 그룹 {group_idx+1}/{len(channel_groups)} 처리 시작 ({len(channel_group)}개 채널)")
            
            # 그룹 내 채널들을 병렬로 처리
            tasks = []
            for channel in channel_group:
                tasks.append(
                    process_channel_historical_data(
                        channel,
                        videos_per_channel=videos_per_channel,
                        process_limit=process_limit_per_channel,
                        target_programs=target_programs
                    )
                )
            
            # 모든 태스크 실행하고 결과 기다리기
            group_results = await asyncio.gather(*tasks)
            
            # 결과 처리
            for result in group_results:
                results.append(result)
                if not result.get("skipped", False):
                    total_success += result.get("success_count", 0)
            
            # 그룹 간 간격을 두어 API 제한 준수
            if group_idx < len(channel_groups) - 1:
                print(f"다음 채널 그룹 처리 전 10초 대기 중...")
                await asyncio.sleep(10)
        
        return {
            "status": "success",
            "total_channels": len(channels),
            "total_success": total_success,
            "results": results
        }
        
    except Exception as e:
        print(f"과거 데이터 처리 중 오류: {str(e)}")
        return {"status": "error", "message": str(e)}

async def main():
    """메인 함수"""
    print("과거 데이터 처리 프로그램 시작")
    
    try:
        # 모든 채널 과거 데이터 처리
        result = await process_all_channels_historical_data(
            videos_per_channel=500,  # 각 채널에서 가져올 최대 영상 수 (500개로 증가)
            process_limit_per_channel=20,   # 각 채널에서 실제로 처리할 최대 영상 수 (20개로 증가)
            target_programs=["주말라이브 주식싹쓰리", "최현덕"],  # 특정 프로그램만 처리
            concurrent_channels=3  # 3개 채널 동시 처리
        )
        
        print(f"처리 결과: {result['status']}")
        print(f"총 처리된 채널: {result['total_channels']}")
        print(f"성공적으로 처리된 영상: {result['total_success']}")
        
        # 채널별 상세 결과
        for channel_result in result.get("results", []):
            channel_name = channel_result["channel_name"]
            keyword = channel_result["keyword"]
            
            if channel_result.get("skipped", False):
                print(f"채널: {channel_name}, 키워드: {keyword}, 건너뜀: {channel_result.get('reason', '이유 없음')}")
            else:
                print(f"채널: {channel_name}, 키워드: {keyword}, 성공: {channel_result.get('success_count', 0)}")
        
    except Exception as e:
        print(f"프로그램 실행 중 오류: {str(e)}")
    
    print("과거 데이터 처리 프로그램 종료")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"치명적 오류: {str(e)}")