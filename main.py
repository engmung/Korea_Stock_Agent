import os
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모듈화된 컴포넌트 임포트
from scheduler import setup_scheduler, simulate_scheduler_at_time, process_channels_by_setting
from notion_utils import (
    query_notion_database,
    REFERENCE_DB_ID, 
    SCRIPT_DB_ID
)
from historical_data_processor import process_all_channels_historical_data

app = FastAPI(title="투자 의사결정 지원 시스템")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NotionSyncRequest(BaseModel):
    pass  # 빈 요청 본문, 단순히 동기화 작업 트리거용

class NotionSyncResponse(BaseModel):
    status: str
    message: str

class SimulateRequest(BaseModel):
    time_setting: int = Field(
        9, 
        description="시뮬레이션할 시간 설정 (0-23 사이의 정수)", 
        ge=0, 
        le=23
    )
    simulate_only: bool = Field(
        True, 
        description="실제 채널 처리 실행 여부 (True: 시뮬레이션만, False: 실제 실행)"
    )

class HistoricalDataRequest(BaseModel):
    videos_per_channel: int = 500
    process_limit_per_channel: int = 20
    target_programs: Optional[List[str]] = None
    concurrent_channels: int = 3

@app.get("/")
async def root():
    return {"message": "투자 의사결정 지원 시스템 API"}

@app.post("/sync-channels", response_model=NotionSyncResponse)
async def sync_channels(background_tasks: BackgroundTasks):
    """모든 채널에 대해 콘텐츠를 추출하고 분석합니다."""
    try:
        # 백그라운드 작업으로 실행
        background_tasks.add_task(process_channels_by_setting)
        return {"status": "processing", "message": "동기화 작업이 시작되었습니다. 완료까지 시간이 걸릴 수 있습니다."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채널 동기화 중 오류가 발생했습니다: {str(e)}")

@app.post("/run-now")
async def run_now():
    """지금 바로 채널 처리 작업을 실행합니다."""
    try:
        await process_channels_by_setting()
        return {"status": "success", "message": "채널 처리 작업이 실행되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"즉시 실행 중 오류가 발생했습니다: {str(e)}")

@app.post("/simulate", description="특정 시간 설정에 대한 작업 시뮬레이션. 시간을 지정하여 해당 시간에 어떤 채널이 처리될지 확인하거나 실제로 처리합니다.")
async def simulate(request: SimulateRequest):
    """특정 시간 설정에 대한 작업 시뮬레이션"""
    try:
        time_setting = request.time_setting
        simulate_only = request.simulate_only
        
        result = await simulate_scheduler_at_time(time_setting, simulate_only)
        return {
            "status": "success", 
            "message": f"시간 {time_setting}시 설정에 대한 {'시뮬레이션' if simulate_only else '실행'} 완료",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시뮬레이션 중 오류가 발생했습니다: {str(e)}")

@app.get("/reference-db")
async def get_reference_db():
    """참고용 DB의 모든 채널 정보 조회"""
    try:
        pages = await query_notion_database(REFERENCE_DB_ID)
        
        channels = []
        for page in pages:
            properties = page.get("properties", {})
            
            # 필요한 속성 추출
            channel_info = {
                "id": page.get("id"),
                "title": "",
                "url": "",
                "channel_name": "",
                "active": False,
                "time": 9,
                "content_type": "",
                "investment_style": []
            }
            
            # 제목
            if "제목" in properties and "title" in properties["제목"] and properties["제목"]["title"]:
                channel_info["title"] = properties["제목"]["title"][0]["plain_text"]
            
            # URL
            if "URL" in properties and "url" in properties["URL"]:
                channel_info["url"] = properties["URL"]["url"]
            
            # 채널명
            if "채널명" in properties and "select" in properties["채널명"] and properties["채널명"]["select"]:
                channel_info["channel_name"] = properties["채널명"]["select"]["name"]
            
            # 활성화
            if "활성화" in properties and "checkbox" in properties["활성화"]:
                channel_info["active"] = properties["활성화"]["checkbox"]
            
            # 시간
            if "시간" in properties and "number" in properties["시간"] and properties["시간"]["number"] is not None:
                channel_info["time"] = properties["시간"]["number"]
            
            # 콘텐츠 유형
            if "콘텐츠 유형" in properties and "select" in properties["콘텐츠 유형"] and properties["콘텐츠 유형"]["select"]:
                channel_info["content_type"] = properties["콘텐츠 유형"]["select"]["name"]
            
            # 투자 스타일
            if "투자 스타일" in properties and "multi_select" in properties["투자 스타일"]:
                channel_info["investment_style"] = [item["name"] for item in properties["투자 스타일"]["multi_select"]]
            
            channels.append(channel_info)
        
        return {"status": "success", "channels": channels, "total": len(channels)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"참고용 DB 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/script-db")
async def get_script_db():
    """스크립트 DB의 모든 분석 보고서 정보 조회"""
    try:
        pages = await query_notion_database(SCRIPT_DB_ID)
        
        scripts = []
        for page in pages:
            properties = page.get("properties", {})
            
            # 필요한 속성 추출
            script_info = {
                "id": page.get("id"),
                "title": "",
                "url": "",
                "video_date": "",
                "channel_name": "",
                "video_length": "",
                "citation_count": 0,
                "presenters": []
            }
            
            # 제목
            if "제목" in properties and "title" in properties["제목"] and properties["제목"]["title"]:
                script_info["title"] = properties["제목"]["title"][0]["plain_text"]
            
            # URL
            if "URL" in properties and "url" in properties["URL"]:
                script_info["url"] = properties["URL"]["url"]
            
            # 영상 날짜
            if "영상 날짜" in properties and "date" in properties["영상 날짜"] and properties["영상 날짜"]["date"]:
                script_info["video_date"] = properties["영상 날짜"]["date"]["start"]
            
            # 채널명
            if "채널명" in properties and "select" in properties["채널명"] and properties["채널명"]["select"]:
                script_info["channel_name"] = properties["채널명"]["select"]["name"]
            
            # 영상 길이
            if "영상 길이" in properties and "rich_text" in properties["영상 길이"] and properties["영상 길이"]["rich_text"]:
                script_info["video_length"] = properties["영상 길이"]["rich_text"][0]["plain_text"]
            
            # 인용 횟수
            if "인용 횟수" in properties and "number" in properties["인용 횟수"]:
                script_info["citation_count"] = properties["인용 횟수"]["number"] or 0
            
            # 출연자
            if "출연자" in properties and "multi_select" in properties["출연자"]:
                script_info["presenters"] = [item["name"] for item in properties["출연자"]["multi_select"]]
            
            scripts.append(script_info)
        
        # 영상 날짜 기준 내림차순 정렬
        scripts.sort(key=lambda x: x["video_date"] or "", reverse=True)
        
        return {"status": "success", "scripts": scripts, "total": len(scripts)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"스크립트 DB 조회 중 오류가 발생했습니다: {str(e)}")

@app.post("/process-historical-data")
async def process_historical_data(background_tasks: BackgroundTasks, request: HistoricalDataRequest):
    """
    과거 데이터를 처리하여 분석 보고서를 Notion DB에 저장합니다
    YouTube Data API를 활용하여 채널의 과거 영상을 검색하고 분석합니다
    
    - videos_per_channel: 각 채널에서 가져올 최대 영상 수 (기본값: 500)
    - process_limit_per_channel: 각 채널에서 실제로 처리할 최대 영상 수 (기본값: 20)
    - target_programs: 특정 프로그램 제목 목록 (지정 시 해당 프로그램만 처리)
    - concurrent_channels: 동시에 처리할 채널 수 (기본값: 3)
    """
    try:
        # 병렬 처리 설정 검증
        concurrent_channels = max(1, min(request.concurrent_channels, 5))  # 1-5 사이로 제한
        
        # 백그라운드 작업으로 실행
        def run_historical_processor():
            import asyncio
            asyncio.run(process_all_channels_historical_data(
                videos_per_channel=request.videos_per_channel,
                process_limit_per_channel=request.process_limit_per_channel,
                target_programs=request.target_programs,
                concurrent_channels=concurrent_channels
            ))
        
        background_tasks.add_task(run_historical_processor)
        
        return {
            "status": "processing", 
            "message": "과거 데이터 처리가 백그라운드에서 시작되었습니다. 완료까지 시간이 걸릴 수 있습니다.",
            "config": {
                "videos_per_channel": request.videos_per_channel,
                "process_limit_per_channel": request.process_limit_per_channel,
                "target_programs": request.target_programs,
                "concurrent_channels": concurrent_channels
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"과거 데이터 처리 시작 중 오류가 발생했습니다: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 스케줄러 설정"""
    setup_scheduler()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)