import os
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from datetime import datetime

# 환경 변수 로드
load_dotenv()

# 모듈화된 컴포넌트 임포트
from notion_utils import find_agent_by_name, create_recommendation_record
from investment_agent import InvestmentAgent, create_investment_agent
from report_analyzer import analyze_reports, find_relevant_reports
from stock_recommender import recommend_stocks
from performance_evaluator import backtest_recommendation
from backtest_scheduler import start_scheduler

# 로깅 설정 - 간결한 포맷으로 변경
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="투자 에이전트 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 모델 정의
class AgentDefinition(BaseModel):
    agent_name: Optional[str] = None
    investment_philosophy: str
    target_channels: List[str] = []
    target_keywords: List[str] = []
    recommendation_strength: List[str] = ["매수", "적극매수"]
    investment_horizon: List[str] = ["단기", "중기"]
    backtest_schedule: Optional[str] = None

class RecommendationByNameRequest(BaseModel):
    agent_name: str
    max_stocks: int = Field(5, ge=1, le=10)
    investment_period: int = Field(7, description="투자 기간(일)", ge=1, le=30)

class BacktestByNameRequest(BaseModel):
    agent_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    investment_amount: float = 1000000

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트 핸들러"""
    logger.info("투자 에이전트 API 시작")
    
    # 백테스팅 스케줄러 시작
    start_scheduler()
    logger.info("백테스팅 스케줄러 시작됨")

@app.get("/")
async def root():
    return {"message": "투자 에이전트 API"}

@app.post("/agents")
async def create_agent(agent_def: AgentDefinition):
    """새 투자 에이전트를 생성합니다."""
    try:
        agent_data = agent_def.dict()
        
        # 백테스팅 예약 정보가 있으면 추가
        if agent_def.backtest_schedule:
            agent_data["backtest_schedule"] = agent_def.backtest_schedule
        
        # 에이전트 생성
        result = await create_investment_agent(agent_data)
        
        if result and result.get("id"):
            # 페이지 ID 추출
            page_id = result.get("id")
            
            # 에이전트 기본 정보 반환
            return {
                "status": "success", 
                "agent": {
                    "id": page_id,
                    "agent_name": agent_def.agent_name or f"에이전트_{datetime.now().strftime('%Y%m%d%H%M%S')}", 
                    "investment_philosophy": agent_def.investment_philosophy,
                    "backtest_schedule": agent_def.backtest_schedule
                }
            }
        else:
            raise HTTPException(status_code=400, detail="에이전트 생성 실패")
    
    except Exception as e:
        logger.error(f"에이전트 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에이전트 생성 중 오류: {str(e)}")
    
@app.post("/backtest/scheduler/test")
async def test_backtest_scheduler():
    """
    백테스팅 예약 스케줄러를 수동으로 실행합니다.
    이 API는 테스트 용도로, 실제로는 매 시간 30분에 자동으로 실행됩니다.
    """
    try:
        from backtest_scheduler import check_backtest_schedules
        
        # 백테스팅 스케줄러 수동 실행
        await check_backtest_schedules()
        
        return {
            "status": "success",
            "message": "백테스팅 예약 스케줄러가 성공적으로 실행되었습니다.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"백테스팅 스케줄러 테스트 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"백테스팅 스케줄러 테스트 중 오류: {str(e)}")

@app.post("/backtest/scheduler/force-run")
async def force_run_scheduler():
    """
    백테스팅 예약 스케줄러를 강제로 즉시 실행합니다.
    """
    try:
        from backtest_scheduler import run_async_in_thread, check_backtest_schedules
        
        logger.info("API를 통한 백테스팅 스케줄러 강제 실행 시작")
        # API 컨텍스트에서는 직접 비동기 함수를 호출할 수 있음
        await check_backtest_schedules()
        logger.info("API를 통한 백테스팅 스케줄러 강제 실행 완료")
        
        return {
            "status": "success",
            "message": "백테스팅 예약 스케줄러가 강제로 실행되었습니다.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"백테스팅 스케줄러 강제 실행 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"백테스팅 스케줄러 강제 실행 중 오류: {str(e)}")

@app.get("/backtest/scheduler/status")
async def get_scheduler_status():
    """
    백테스팅 예약 스케줄러의 상태를 확인합니다.
    """
    try:
        from backtest_scheduler import scheduler
        
        jobs = scheduler.get_jobs()
        next_runs = []
        
        for job in jobs:
            if job.id == 'backtest_scheduler':
                next_runs.append({
                    "id": job.id,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None
                })
        
        return {
            "status": "success",
            "scheduler_running": scheduler.running,
            "next_scheduled_runs": next_runs,
            "current_time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"스케줄러 상태 확인 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"스케줄러 상태 확인 중 오류: {str(e)}")

@app.get("/agents")
async def list_agents():
    """모든 투자 에이전트 목록을 가져옵니다."""
    try:
        # Notion DB에서 에이전트 페이지 조회
        from notion_utils import query_notion_database, NOTION_AGENT_DB_ID
        agent_pages = await query_notion_database(NOTION_AGENT_DB_ID)
        
        agents = []
        for page in agent_pages:
            properties = page.get("properties", {})
            
            agent_info = {
                "id": page.get("id"),
                "agent_name": "",
                "investment_philosophy": "",
                "status": "활성",
                "created_at": ""
            }
            
            # 에이전트명
            if "에이전트명" in properties and "title" in properties["에이전트명"]:
                title_obj = properties["에이전트명"]["title"]
                if title_obj and len(title_obj) > 0:
                    agent_info["agent_name"] = title_obj[0]["plain_text"]
            
            # 투자 철학
            if "투자 철학" in properties and "rich_text" in properties["투자 철학"]:
                text_obj = properties["투자 철학"]["rich_text"]
                if text_obj and len(text_obj) > 0:
                    agent_info["investment_philosophy"] = text_obj[0]["plain_text"]
            
            # 상태
            if "현재 상태" in properties and "select" in properties["현재 상태"]:
                status_obj = properties["현재 상태"]["select"]
                if status_obj:
                    agent_info["status"] = status_obj["name"]
                
            # 생성일
            if "생성일" in properties and "date" in properties["생성일"]:
                date_obj = properties["생성일"]["date"]
                if date_obj and "start" in date_obj:
                    agent_info["created_at"] = date_obj["start"]
            
            agents.append(agent_info)
        
        return {"status": "success", "agents": agents, "total": len(agents)}
        
    except Exception as e:
        logger.error(f"에이전트 목록 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에이전트 목록 조회 중 오류: {str(e)}")

@app.get("/agents/name/{agent_name}")
async def get_agent_by_name(agent_name: str):
    """에이전트명으로 투자 에이전트의 정보를 가져옵니다."""
    try:
        # 에이전트명으로 페이지 ID 검색
        page_id = await find_agent_by_name(agent_name)
        
        if not page_id:
            raise HTTPException(status_code=404, detail=f"에이전트명 '{agent_name}'에 해당하는 에이전트를 찾을 수 없습니다.")
        
        # 에이전트 로드
        agent = await InvestmentAgent.load_from_notion(page_id)
        
        if agent:
            return {
                "status": "success",
                "agent": {
                    "id": agent.page_id,
                    "agent_name": agent.agent_name,
                    "investment_philosophy": agent.investment_philosophy,
                    "target_channels": agent.target_channels,
                    "target_keywords": agent.target_keywords,
                    "recommendation_strength": agent.recommendation_strength_filter,
                    "investment_horizon": agent.investment_horizon,
                    "status": agent.status
                }
            }
        else:
            raise HTTPException(status_code=404, detail=f"에이전트를 찾을 수 없습니다: {page_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"에이전트 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에이전트 조회 중 오류: {str(e)}")

@app.post("/recommendations/name")
async def get_recommendations_by_name(request: RecommendationByNameRequest):
    """에이전트명 기반으로 투자 에이전트의 종목 추천을 제공합니다."""
    try:
        # 에이전트명으로 페이지 ID 검색
        page_id = await find_agent_by_name(request.agent_name)
        
        if not page_id:
            raise HTTPException(status_code=404, detail=f"에이전트명 '{request.agent_name}'에 해당하는 에이전트를 찾을 수 없습니다.")
        
        # 에이전트 로드
        agent = await InvestmentAgent.load_from_notion(page_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"에이전트를 찾을 수 없습니다: {page_id}")
        
        # 현재 날짜 기준 설정
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # 관련 보고서 검색 - 현재 날짜 이전의 보고서만
        reports = await find_relevant_reports(
            agent=agent,
            backtest_date=current_date,
            max_reports=10
        )
        
        if not reports:
            return {
                "status": "warning",
                "message": "투자 에이전트의 조건에 맞는 보고서를 찾을 수 없습니다.",
                "recommendations": []
            }
        
        # LLM을 사용한 보고서 분석 및 종목 추천 (통합)
        from report_analyzer import analyze_reports_with_llm
        result = await analyze_reports_with_llm(
            reports=reports,
            agent=agent,
            max_stocks=request.max_stocks,
            investment_period=request.investment_period
        )
        
        # 추천 기록 저장 (백테스팅 없이)
        await create_recommendation_record(
            agent_page_id=page_id,
            recommendations=result,
            investment_period=request.investment_period
        )
        
        return {
            "status": "success",
            "agent_name": agent.agent_name,
            "investment_philosophy": agent.investment_philosophy,
            "analyzed_reports_count": result.get("total_reports", 0),
            "recommendations": result.get("recommended_stocks", []),
            "portfolio_logic": result.get("portfolio_logic", ""),
            "recommendation_date": current_date
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"종목 추천 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"종목 추천 중 오류: {str(e)}")

@app.post("/backtest/name")
async def run_backtest_by_name(request: BacktestByNameRequest):
    """에이전트명 기반으로 투자 에이전트의 백테스팅을 실행합니다."""
    try:
        # 에이전트명으로 페이지 ID 검색
        page_id = await find_agent_by_name(request.agent_name)
        
        if not page_id:
            raise HTTPException(status_code=404, detail=f"에이전트명 '{request.agent_name}'에 해당하는 에이전트를 찾을 수 없습니다.")
        
        # 에이전트 로드
        agent = await InvestmentAgent.load_from_notion(page_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"에이전트를 찾을 수 없습니다: {page_id}")
        
        # 백테스팅 시작
        result = await backtest_recommendation(
            page_id=page_id,
            start_date=request.start_date,
            end_date=request.end_date,
            investment_amount=request.investment_amount
        )
        
        return {
            "status": "success",
            "message": "백테스팅이 성공적으로 요청되었습니다. 결과는 자동으로 기록됩니다.",
            "agent_name": agent.agent_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"백테스팅 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"백테스팅 중 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)