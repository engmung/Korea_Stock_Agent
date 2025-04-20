import os
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# API 키 가져오기
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

async def get_videos_by_channel_id(channel_id: str, max_results: int = 50, 
                                   published_after: Optional[str] = None,
                                   page_token: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    채널 ID로 영상 목록을 가져옵니다.
    
    Args:
        channel_id: YouTube 채널 ID
        max_results: 페이지당 최대 결과 수
        published_after: ISO 8601 형식의 날짜 문자열 (예: 2024-01-01T00:00:00Z)
        page_token: 다음 페이지 토큰
        
    Returns:
        영상 리스트와 다음 페이지 토큰
    """
    try:
        # API 클라이언트는 동기 함수이므로 비동기 실행을 위해 실행 루프에서 실행
        def api_call():
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            
            # 검색 파라미터 설정
            search_params = {
                "part": "snippet",
                "channelId": channel_id,
                "maxResults": max_results,
                "order": "date",  # 날짜순 정렬
                "type": "video"
            }
            
            # 선택적 파라미터 추가
            if page_token:
                search_params["pageToken"] = page_token
                
            if published_after:
                search_params["publishedAfter"] = published_after
            
            # API 호출
            request = youtube.search().list(**search_params)
            return request.execute()
        
        # 비동기로 API 호출
        response = await asyncio.to_thread(api_call)
        
        videos = []
        for item in response.get("items", []):
            # 영상 정보 추출
            snippet = item.get("snippet", {})
            video = {
                "title": snippet.get("title", ""),
                "video_id": item["id"]["videoId"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "upload_date": snippet.get("publishedAt", ""),  # 기존 코드와 키 이름 일관성 유지
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "is_upcoming": False,  # 기본값
                "is_live": False,      # 기본값
                "video_length": "Unknown"  # 기본값
            }
            videos.append(video)
        
        # 페이지네이션 정보
        next_page_token = response.get("nextPageToken")
        
        return videos, next_page_token
    
    except HttpError as e:
        print(f"YouTube API HTTP 오류: {str(e)}")
        return [], None
    except Exception as e:
        print(f"영상 목록 가져오기 오류: {str(e)}")
        return [], None

async def search_videos_in_channel(channel_id: str, keyword: str, max_results: int = 50,
                                   published_after: Optional[str] = None,
                                   page_token: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    채널 내에서 키워드가 포함된 영상을 검색합니다.
    
    Args:
        channel_id: YouTube 채널 ID
        keyword: 검색할 키워드
        max_results: 페이지당 최대 결과 수
        published_after: ISO 8601 형식의 날짜 문자열 (예: 2024-01-01T00:00:00Z)
        page_token: 다음 페이지 토큰
        
    Returns:
        영상 리스트와 다음 페이지 토큰
    """
    try:
        def api_call():
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            
            # 검색 파라미터 설정
            search_params = {
                "part": "snippet",
                "channelId": channel_id,
                "maxResults": max_results,
                "q": keyword,  # 검색어
                "type": "video"
            }
            
            # 선택적 파라미터 추가
            if page_token:
                search_params["pageToken"] = page_token
                
            if published_after:
                search_params["publishedAfter"] = published_after
            
            # API 호출
            request = youtube.search().list(**search_params)
            return request.execute()
        
        # 비동기로 API 호출
        response = await asyncio.to_thread(api_call)
        
        videos = []
        for item in response.get("items", []):
            # 영상 정보 추출
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            
            # 키워드가 제목에 포함된 영상만 필터링 (대소문자 구분 없이)
            if keyword.lower() in title.lower():
                video = {
                    "title": title,
                    "video_id": item["id"]["videoId"],
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "upload_date": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "is_upcoming": False,  # 기본값
                    "is_live": False,      # 기본값
                    "video_length": "Unknown"  # 기본값
                }
                videos.append(video)
        
        # 페이지네이션 정보
        next_page_token = response.get("nextPageToken")
        
        return videos, next_page_token
    
    except HttpError as e:
        print(f"YouTube API HTTP 오류: {str(e)}")
        return [], None
    except Exception as e:
        print(f"영상 검색 오류: {str(e)}")
        return [], None

async def get_video_details(video_ids: List[str]) -> List[Dict[str, Any]]:
    """
    여러 비디오 ID에 대한 상세 정보를 가져옵니다.
    """
    try:
        # ID가 너무 많은 경우 나눠서 처리 (API 제한: 최대 50개)
        chunk_size = 50
        all_videos = []
        
        for i in range(0, len(video_ids), chunk_size):
            video_ids_chunk = video_ids[i:i+chunk_size]
            
            def api_call():
                youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
                request = youtube.videos().list(
                    part="snippet,contentDetails,statistics,liveStreamingDetails",
                    id=",".join(video_ids_chunk)
                )
                return request.execute()
            
            # 비동기로 API 호출
            response = await asyncio.to_thread(api_call)
            
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                
                # 라이브 상태 확인 - 수정된 로직
                live_details = item.get("liveStreamingDetails", {})
                is_live = False
                is_upcoming = False
                
                # 라이브 상태 판단 로직 수정:
                # 1. actualEndTime이 있으면 이미 종료된 라이브임
                # 2. actualStartTime이 있고 actualEndTime이 없으면 현재 진행중인 라이브임
                # 3. scheduledStartTime만 있고 actualStartTime이 없으면 예정된 라이브임
                if "actualEndTime" not in live_details:
                    if "actualStartTime" in live_details:
                        is_live = True
                    elif "scheduledStartTime" in live_details:
                        is_upcoming = True
                
                # 영상 길이 형식 변환 (ISO 8601 기간 형식)
                duration = content_details.get("duration", "PT0S")  # 기본값: 0초
                video_length = format_duration(duration)
                
                # 영상 길이 초 단위로 계산
                duration_seconds = parse_duration_to_seconds(duration)
                
                # liveBroadcastContent 확인 (추가 검증)
                if snippet.get("liveBroadcastContent") == "live":
                    is_live = True
                elif snippet.get("liveBroadcastContent") == "upcoming":
                    is_upcoming = True
                
                video = {
                    "video_id": item.get("id", ""),
                    "title": snippet.get("title", ""),
                    "url": f"https://www.youtube.com/watch?v={item.get('id', '')}",
                    "upload_date": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "is_live": is_live,
                    "is_upcoming": is_upcoming,
                    "video_length": video_length,
                    "duration_seconds": duration_seconds,
                    "view_count": item.get("statistics", {}).get("viewCount", "0"),
                    "like_count": item.get("statistics", {}).get("likeCount", "0")
                }
                all_videos.append(video)
            
            # API 요청 제한 준수
            if i + chunk_size < len(video_ids):
                await asyncio.sleep(1)
        
        return all_videos
    
    except HttpError as e:
        print(f"YouTube API HTTP 오류: {str(e)}")
        return []
    except Exception as e:
        print(f"비디오 상세 정보 가져오기 오류: {str(e)}")
        return []

async def get_channel_id_from_url(channel_url: str) -> Optional[str]:
    """
    YouTube 채널 URL을 이용해 채널 ID를 검색합니다.
    
    Args:
        channel_url: YouTube 채널 URL (예: https://www.youtube.com/@channelname)
        
    Returns:
        채널 ID 또는 None (추출 실패 시)
    """
    # 먼저 URL 형식 분석으로 채널 ID 추출 시도
    channel_id = extract_channel_id_from_url(channel_url)
    if channel_id:
        return channel_id
        
    # URL에서 추출 실패 시 API 호출로 시도
    try:
        # 채널명/유저명 추출
        channel_handle = None
        if "/@" in channel_url:
            parts = channel_url.split("/@")
            if len(parts) >= 2:
                channel_handle = parts[1].split("/")[0].split("?")[0]
        elif "/user/" in channel_url:
            parts = channel_url.split("/user/")
            if len(parts) >= 2:
                channel_handle = parts[1].split("/")[0].split("?")[0]
                
        if not channel_handle:
            print(f"채널 핸들을 추출할 수 없습니다: {channel_url}")
            return None
            
        # API 호출로 채널 ID 검색
        def api_call():
            youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
            
            # 검색 쿼리 구성
            request = youtube.search().list(
                part="snippet",
                q=channel_handle,
                type="channel",
                maxResults=1
            )
            
            return request.execute()
            
        # 비동기로 API 호출
        response = await asyncio.to_thread(api_call)
        
        # 결과 처리
        if "items" in response and len(response["items"]) > 0:
            item = response["items"][0]
            if "id" in item and "channelId" in item["id"]:
                channel_id = item["id"]["channelId"]
                print(f"API 검색으로 채널 ID 찾음: {channel_url} -> {channel_id}")
                return channel_id
                
        print(f"API 검색으로 채널 ID를 찾을 수 없습니다: {channel_url}")
        return None
        
    except Exception as e:
        print(f"채널 ID 검색 중 오류: {str(e)}")
        return None

def extract_channel_id_from_url(channel_url: str) -> Optional[str]:
    """
    URL 형식 분석을 통해 채널 ID를 추출합니다.
    
    Args:
        channel_url: YouTube 채널 URL
        
    Returns:
        추출된 채널 ID 또는 None
    """
    try:
        # 형식: https://www.youtube.com/channel/[CHANNEL_ID]
        if "/channel/" in channel_url:
            parts = channel_url.split("/channel/")
            if len(parts) >= 2:
                channel_id = parts[1].split("/")[0].split("?")[0]
                return channel_id
    except Exception as e:
        print(f"채널 URL에서 ID 추출 오류: {str(e)}")
    
    return None

def format_duration(iso_duration: str) -> str:
    """
    ISO 8601 기간 형식을 읽기 쉬운 형식으로 변환합니다.
    예: PT1H30M15S -> 1:30:15
    """
    try:
        hours = 0
        minutes = 0
        seconds = 0
        
        # 시간 추출
        hour_match = re.search(r'(\d+)H', iso_duration)
        if hour_match:
            hours = int(hour_match.group(1))
        
        # 분 추출
        minute_match = re.search(r'(\d+)M', iso_duration)
        if minute_match:
            minutes = int(minute_match.group(1))
        
        # 초 추출
        second_match = re.search(r'(\d+)S', iso_duration)
        if second_match:
            seconds = int(second_match.group(1))
        
        # 형식 지정
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    except:
        return "Unknown"

def parse_duration_to_seconds(iso_duration: str) -> int:
    """
    ISO 8601 기간 형식을 초 단위로 변환합니다.
    예: PT1H30M15S -> 5415
    """
    try:
        hours = 0
        minutes = 0
        seconds = 0
        
        # 시간 추출
        hour_match = re.search(r'(\d+)H', iso_duration)
        if hour_match:
            hours = int(hour_match.group(1))
        
        # 분 추출
        minute_match = re.search(r'(\d+)M', iso_duration)
        if minute_match:
            minutes = int(minute_match.group(1))
        
        # 초 추출
        second_match = re.search(r'(\d+)S', iso_duration)
        if second_match:
            seconds = int(second_match.group(1))
        
        # 초 단위로 변환
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    except:
        return 0

async def find_latest_video_for_channel(channel_id: str, keyword: str, max_results: int = 50) -> Optional[Dict[str, Any]]:
    """
    채널에서 키워드가 포함된 최신 영상을 찾습니다.
    
    Args:
        channel_id: YouTube 채널 ID
        keyword: 검색할 키워드
        max_results: 검색할 최대 영상 수
        
    Returns:
        최신 영상 정보 또는 None
    """
    # 1. 채널의 영상 목록 가져오기
    videos, _ = await get_videos_by_channel_id(channel_id, max_results=max_results)
    
    if not videos:
        print(f"채널 {channel_id}에서 영상을 찾을 수 없습니다.")
        return None
    
    # 2. 키워드가 포함된 영상 필터링
    matching_videos = []
    for video in videos:
        if keyword.lower() in video["title"].lower():
            matching_videos.append(video)
    
    if not matching_videos:
        print(f"채널 {channel_id}에서 키워드 '{keyword}'가 포함된 영상을 찾을 수 없습니다.")
        return None
    
    # 3. 영상 세부 정보 가져오기
    video_ids = [video["video_id"] for video in matching_videos]
    videos_with_details = await get_video_details(video_ids)
    
    if not videos_with_details:
        print(f"영상 세부 정보를 가져올 수 없습니다.")
        return None
    
    # 4. 영상 필터링 - 5분(300초) 이내 영상 제외, 라이브/예정 영상 별도 처리
    live_videos = []
    upcoming_videos = []
    normal_videos = []
    
    for video in videos_with_details:
        # 라이브 영상 우선
        if video.get("is_live", False):
            live_videos.append(video)
        # 예정된 영상 다음 순위
        elif video.get("is_upcoming", False):
            upcoming_videos.append(video)
        # 일반 영상은 5분(300초) 이상인 경우만 포함
        elif video.get("duration_seconds", 0) >= 300:
            normal_videos.append(video)
    
    # 5. 우선순위에 따라 영상 반환
    if live_videos:
        print(f"라이브 중인 영상 발견: {live_videos[0]['title']}")
        return live_videos[0]  # 라이브 중인 영상 우선
    
    if upcoming_videos:
        print(f"라이브 예정 영상 발견: {upcoming_videos[0]['title']}")
        return upcoming_videos[0]  # 라이브 예정 영상 다음 우선
    
    if normal_videos:
        # 업로드 날짜 기준 내림차순 정렬
        normal_videos.sort(key=lambda x: x.get("upload_date", ""), reverse=True)
        print(f"일반 영상 발견: {normal_videos[0]['title']}")
        return normal_videos[0]  # 가장 최근 영상 반환
    
    print("적합한 영상을 찾을 수 없습니다 (모두 5분 이하 또는 없음)")
    return None  # 적합한 영상이 없는 경우