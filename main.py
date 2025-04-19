import os
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 모듈화된 컴포넌트 임포트
from scheduler import setup_scheduler, simulate_scheduler_at_time, process_channels_by_setting
from notion_utils import (
    query_notion_database, 
    create_investment_agent, 
    create_investment_performance,
    REFERENCE_DB_ID, 
    SCRIPT_DB_ID,
    INVESTMENT_AGENT_DB_ID,
    INVESTMENT_PERFORMANCE_DB_ID
)

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
    time_setting: int
    simulate_only: bool = True

class CreateAgentRequest(BaseModel):
    agent_id: str
    investment_philosophy: str
    status: str = "활성"

class CreatePerformanceRequest(BaseModel):
    title: str
    agent_id: str
    agent_id_relation: str
    start_date: str
    end_date: str
    stocks: list
    weights: str
    total_return: float
    max_drawdown: float
    evaluation: str

@app.get("/")
async def root():
    return {"message": "투자 의사결정 지원 시스템 API"}

@app.post("/sync-channels", response_model=NotionSyncResponse)
async def sync_channels(background_tasks: BackgroundTasks):
    """모든 채널에 대해 콘텐츠를 추출하고 분석합니다."""
    try:
        # 백그라운드 작업으로 실행
        logger.info("채널 동기화 작업 시작")
        background_tasks.add_task(process_channels_by_setting)
        return {"status": "processing", "message": "동기화 작업이 시작되었습니다. 완료까지 시간이 걸릴 수 있습니다."}
    
    except Exception as e:
        logger.error(f"채널 동기화 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"채널 동기화 중 오류가 발생했습니다: {str(e)}")

@app.post("/run-now")
async def run_now():
    """지금 바로 채널 처리 작업을 실행합니다."""
    try:
        logger.info("즉시 채널 처리 작업 실행")
        await process_channels_by_setting()
        return {"status": "success", "message": "채널 처리 작업이 실행되었습니다."}
    except Exception as e:
        logger.error(f"즉시 실행 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"즉시 실행 중 오류가 발생했습니다: {str(e)}")

@app.post("/simulate")
async def simulate(request: Dict[str, Any]):
    """특정 시간 설정에 대한 작업 시뮬레이션"""
    try:
        time_setting = request.get("time_setting", datetime.now().hour)
        simulate_only = request.get("simulate_only", True)
        
        logger.info(f"시뮬레이션 요청: time_setting={time_setting}, simulate_only={simulate_only}")
        
        result = await simulate_scheduler_at_time(time_setting, simulate_only)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"시뮬레이션 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"시뮬레이션 중 오류가 발생했습니다: {str(e)}")

@app.post("/agents")
async def create_agent(request: CreateAgentRequest):
    """새 투자 에이전트 생성"""
    try:
        agent_data = {
            "agent_id": request.agent_id,
            "investment_philosophy": request.investment_philosophy,
            "status": request.status,
            "avg_return": 0,
            "success_rate": 0
        }
        
        result = await create_investment_agent(agent_data)
        
        if not result:
            raise HTTPException(status_code=500, detail="에이전트 생성에 실패했습니다")
        
        return {"status": "success", "message": f"에이전트 {request.agent_id} 생성 완료", "agent_id": result.get("id")}
    
    except Exception as e:
        logger.error(f"에이전트 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에이전트 생성 중 오류가 발생했습니다: {str(e)}")

@app.post("/performances")
async def create_performance(request: CreatePerformanceRequest):
    """새 투자 실적 기록 생성"""
    try:
        # 날짜 문자열을 datetime 객체로 변환
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        performance_data = {
            "title": request.title,
            "agent_id_relation": request.agent_id_relation,
            "start_date": start_date,
            "end_date": end_date,
            "stocks": request.stocks,
            "weights": request.weights,
            "total_return": request.total_return,
            "max_drawdown": request.max_drawdown,
            "evaluation": request.evaluation
        }
        
        result = await create_investment_performance(performance_data)
        
        if not result:
            raise HTTPException(status_code=500, detail="투자 실적 기록 생성에 실패했습니다")
        
        return {"status": "success", "message": f"투자 실적 {request.title} 생성 완료", "performance_id": result.get("id")}
    
    except Exception as e:
        logger.error(f"투자 실적 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"투자 실적 생성 중 오류가 발생했습니다: {str(e)}")

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
        logger.error(f"참고용 DB 조회 중 오류: {str(e)}")
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
        logger.error(f"스크립트 DB 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"스크립트 DB 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/agents")
async def get_agents():
    """투자 에이전트 DB의 모든 에이전트 정보 조회"""
    try:
        pages = await query_notion_database(INVESTMENT_AGENT_DB_ID)
        
        agents = []
        for page in pages:
            properties = page.get("properties", {})
            
            # 필요한 속성 추출
            agent_info = {
                "id": page.get("id"),
                "agent_id": "",
                "investment_philosophy": "",
                "creation_date": "",
                "status": "",
                "avg_return": 0,
                "success_rate": 0
            }
            
            # 에이전트 ID
            if "에이전트 ID" in properties and "title" in properties["에이전트 ID"] and properties["에이전트 ID"]["title"]:
                agent_info["agent_id"] = properties["에이전트 ID"]["title"][0]["plain_text"]
            
            # 투자 철학
            if "투자 철학" in properties and "rich_text" in properties["투자 철학"] and properties["투자 철학"]["rich_text"]:
                agent_info["investment_philosophy"] = properties["투자 철학"]["rich_text"][0]["plain_text"]
            
            # 생성일
            if "생성일" in properties and "date" in properties["생성일"] and properties["생성일"]["date"]:
                agent_info["creation_date"] = properties["생성일"]["date"]["start"]
            
            # 현재 상태
            if "현재 상태" in properties and "select" in properties["현재 상태"] and properties["현재 상태"]["select"]:
                agent_info["status"] = properties["현재 상태"]["select"]["name"]
            
            # 평균 수익률
            if "평균 수익률" in properties and "number" in properties["평균 수익률"]:
                agent_info["avg_return"] = properties["평균 수익률"]["number"] or 0
            
            # 성공률
            if "성공률" in properties and "number" in properties["성공률"]:
                agent_info["success_rate"] = properties["성공률"]["number"] or 0
            
            agents.append(agent_info)
        
        # 평균 수익률 기준 내림차순 정렬
        agents.sort(key=lambda x: x["avg_return"] or 0, reverse=True)
        
        return {"status": "success", "agents": agents, "total": len(agents)}
    
    except Exception as e:
        logger.error(f"에이전트 DB 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에이전트 DB 조회 중 오류가 발생했습니다: {str(e)}")

@app.get("/performances")
async def get_performances():
    """투자 실적 DB의 모든 실적 정보 조회"""
    try:
        pages = await query_notion_database(INVESTMENT_PERFORMANCE_DB_ID)
        
        performances = []
        for page in pages:
            properties = page.get("properties", {})
            
            # 필요한 속성 추출
            performance_info = {
                "id": page.get("id"),
                "title": "",
                "agent_id": [],
                "start_date": "",
                "end_date": "",
                "stocks": [],
                "weights": "",
                "total_return": 0,
                "max_drawdown": 0,
                "evaluation": ""
            }
            
            # 투자 기록
            if "투자 기록" in properties and "title" in properties["투자 기록"] and properties["투자 기록"]["title"]:
                performance_info["title"] = properties["투자 기록"]["title"][0]["plain_text"]
            
            # 에이전트 ID
            if "에이전트 ID" in properties and "relation" in properties["에이전트 ID"]:
                performance_info["agent_id"] = [relation["id"] for relation in properties["에이전트 ID"]["relation"]]
            
            # 시작일
            if "시작일" in properties and "date" in properties["시작일"] and properties["시작일"]["date"]:
                performance_info["start_date"] = properties["시작일"]["date"]["start"]
            
            # 종료일
            if "종료일" in properties and "date" in properties["종료일"] and properties["종료일"]["date"]:
                performance_info["end_date"] = properties["종료일"]["date"]["start"]
            
            # 투자 종목
            if "투자 종목" in properties and "multi_select" in properties["투자 종목"]:
                performance_info["stocks"] = [item["name"] for item in properties["투자 종목"]["multi_select"]]
            
            # 투자 비중
            if "투자 비중" in properties and "rich_text" in properties["투자 비중"] and properties["투자 비중"]["rich_text"]:
                performance_info["weights"] = properties["투자 비중"]["rich_text"][0]["plain_text"]
            
            # 총 수익률
            if "총 수익률" in properties and "number" in properties["총 수익률"]:
                performance_info["total_return"] = properties["총 수익률"]["number"] or 0
            
            # 최대 낙폭
            if "최대 낙폭" in properties and "number" in properties["최대 낙폭"]:
                performance_info["max_drawdown"] = properties["최대 낙폭"]["number"] or 0
            
            # 결과 평가
            if "결과 평가" in properties and "select" in properties["결과 평가"] and properties["결과 평가"]["select"]:
                performance_info["evaluation"] = properties["결과 평가"]["select"]["name"]
            
            performances.append(performance_info)
        
        # 종료일 기준 내림차순 정렬
        performances.sort(key=lambda x: x["end_date"] or "", reverse=True)
        
        return {"status": "success", "performances": performances, "total": len(performances)}
    
    except Exception as e:
        logger.error(f"투자 실적 DB 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"투자 실적 DB 조회 중 오류가 발생했습니다: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 스케줄러 설정"""
    setup_scheduler()
    logger.info("Application started with scheduler configured")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)