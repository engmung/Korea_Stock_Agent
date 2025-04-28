import os
import asyncio
import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import concurrent.futures
from performance_evaluator import backtest_recommendation
from backtest_scheduler import update_schedule_text

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BacktestWorker:
    """독립적인 백테스팅 작업을 처리하는 워커 클래스."""
    
    def __init__(self, worker_id: str, notion_api_manager, gemini_api_manager):
        self.worker_id = worker_id
        self.notion_api_manager = notion_api_manager
        self.gemini_api_manager = gemini_api_manager
        self.is_running = False
        self.current_task = None
        self.event_loop = None
        self.executor = None
        self.task_queue = asyncio.Queue()
        self.dispatcher = None  # 디스패처 참조 (완료 콜백용)
        logger.info(f"백테스팅 워커 {worker_id} 초기화 완료")
    
    def set_dispatcher(self, dispatcher):
        """디스패처 참조를 설정합니다."""
        self.dispatcher = dispatcher
    
    async def start(self):
        """워커를 시작합니다."""
        if self.is_running:
            return
        
        self.is_running = True
        self.current_task = asyncio.create_task(self._process_queue())
        logger.info(f"백테스팅 워커 {self.worker_id} 시작됨")
    
    async def stop(self):
        """워커를 중지합니다."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.current_task:
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
            self.current_task = None
        
        logger.info(f"백테스팅 워커 {self.worker_id} 중지됨")
    
    async def add_task(self, agent_data: Dict[str, Any]):
        """
        백테스팅 작업을 추가합니다.
        
        Args:
            agent_data: 에이전트 데이터 (페이지 ID 포함)
        """
        await self.task_queue.put(agent_data)
        logger.info(f"워커 {self.worker_id}에 에이전트 '{agent_data.get('agent_name', agent_data.get('id', 'Unknown'))}' 작업 추가")
    
    async def _process_queue(self):
        """작업 큐를 처리하는 루프."""
        try:
            while self.is_running:
                # 큐에서 다음 작업 가져오기
                task_data = await self.task_queue.get()
                
                try:
                    # 에이전트 정보 로깅
                    agent_id = task_data.get("id")
                    task_id = task_data.get("task_id")
                    
                    # 개별 백테스팅 작업 여부 확인
                    if "backtest_task" in task_data:
                        from performance_evaluator import backtest_recommendation
                        from backtest_scheduler import update_schedule_text
                        
                        backtest_task = task_data["backtest_task"]
                        agent_name = backtest_task.get("agent_name", "Unknown")
                        date_range = backtest_task.get("date_range", {})
                        
                        logger.info(f"워커 {self.worker_id}: 백테스팅 작업 '{task_id}' 처리 시작 (에이전트: {agent_name})")
                        
                        # 백테스팅 시작 시간 기록
                        start_time = time.time()
                        
                        # 백테스팅 실행
                        result = await backtest_recommendation(
                            page_id=agent_id, 
                            start_date=date_range.get("start_date"),
                            end_date=date_range.get("end_date"),
                            investment_amount=1000000,  # 기본 투자금액
                            worker_id=self.worker_id,
                            notion_api_manager=self.notion_api_manager,
                            gemini_api_manager=self.gemini_api_manager
                        )
                        
                        if result.get("status") == "success":
                            # 성공 시 개별 예약 제거
                            await update_schedule_text(
                                page_id=agent_id,
                                current_schedule=date_range.get("original_text", ""),
                                completed_item=date_range.get("original_text", ""),
                                notion_api_manager=self.notion_api_manager
                            )
                            logger.info(f"워커 {self.worker_id}: 백테스팅 작업 '{task_id}' 성공")
                        else:
                            logger.error(f"워커 {self.worker_id}: 백테스팅 작업 '{task_id}' 실패: {result.get('message')}")
                        
                        # 처리 시간 계산
                        elapsed_time = time.time() - start_time
                        logger.info(f"워커 {self.worker_id}: 백테스팅 작업 '{task_id}' 처리 완료 (소요 시간: {elapsed_time:.2f}초)")
                    
                    else:
                        # 일반 에이전트 작업 처리 (기존 방식)
                        agent_name = None
                        
                        # 에이전트명 추출 시도
                        if "properties" in task_data and "에이전트명" in task_data["properties"]:
                            title_objs = task_data["properties"]["에이전트명"].get("title", [])
                            if title_objs and len(title_objs) > 0:
                                agent_name = title_objs[0].get("plain_text", "Unknown")
                        
                        logger.info(f"워커 {self.worker_id}: 에이전트 '{agent_name or agent_id}' 처리 시작")
                        
                        # 백테스팅 시작 시간 기록
                        start_time = time.time()
                        
                        # 에이전트 처리
                        from backtest_scheduler import process_agent_schedules
                        
                        await process_agent_schedules(
                            task_data,
                            worker_id=self.worker_id,
                            notion_api_manager=self.notion_api_manager,
                            gemini_api_manager=self.gemini_api_manager,
                            dispatcher=self.dispatcher  # 디스패처 참조 전달
                        )
                        
                        # 처리 시간 계산
                        elapsed_time = time.time() - start_time
                        logger.info(f"워커 {self.worker_id}: 에이전트 '{agent_name or agent_id}' 처리 완료 (소요 시간: {elapsed_time:.2f}초)")
                    
                    # 디스패처에 완료 알림
                    if self.dispatcher:
                        if task_id:
                            await self.dispatcher.mark_task_completed(task_id)
                        else:
                            await self.dispatcher.mark_agent_completed(agent_id)
                    
                except Exception as e:
                    logger.error(f"워커 {self.worker_id}: 작업 처리 중 오류 발생: {str(e)}")
                    # 에러가 발생해도 디스패처에 완료 알림
                    if self.dispatcher:
                        if task_id:
                            await self.dispatcher.mark_task_completed(task_id)
                        elif agent_id:
                            await self.dispatcher.mark_agent_completed(agent_id)
                
                finally:
                    # 작업 완료 표시
                    self.task_queue.task_done()
            
        except asyncio.CancelledError:
            logger.info(f"워커 {self.worker_id}: 작업 처리 루프 종료")
            return


class BacktestDispatcher:
    """에이전트를 워커에 효율적으로 분배하는 디스패처 클래스."""
    
    def __init__(self, notion_api_manager, gemini_api_manager, num_workers: int = 3):
        self.notion_api_manager = notion_api_manager
        self.gemini_api_manager = gemini_api_manager
        self.num_workers = num_workers
        self.workers = []
        self.processing_agents = set()  # 현재 처리 중인 에이전트 ID 세트
        self.lock = asyncio.Lock()
        logger.info(f"백테스팅 디스패처 초기화 완료 (워커 {num_workers}개)")
    
    async def start(self):
        """디스패처와 모든 워커를 시작합니다."""
        # 워커 생성 및 시작
        for i in range(self.num_workers):
            worker_id = f"worker-{i+1}"
            worker = BacktestWorker(
                worker_id=worker_id,
                notion_api_manager=self.notion_api_manager,
                gemini_api_manager=self.gemini_api_manager
            )
            worker.set_dispatcher(self)  # 디스패처 참조 설정
            await worker.start()
            self.workers.append(worker)
        
        logger.info(f"백테스팅 디스패처 시작됨 (워커 {len(self.workers)}개)")
    
    async def stop(self):
        """디스패처와 모든 워커를 중지합니다."""
        for worker in self.workers:
            await worker.stop()
        
        self.workers = []
        logger.info("백테스팅 디스패처 중지됨")

    async def mark_task_completed(self, task_id: str):
        """
        백테스팅 작업 처리 완료를 표시합니다.
        
        Args:
            task_id: 완료된 작업 ID
        """
        async with self.lock:
            if task_id in self.processing_agents:
                self.processing_agents.remove(task_id)
                logger.debug(f"백테스팅 작업 {task_id} 처리 완료 표시")

    async def dispatch_backtest_task(self, task: Dict[str, Any]):
        """
        개별 백테스팅 작업을 워커에 분배합니다.
        
        Args:
            task: 백테스팅 작업 정보
        """
        task_id = f"backtest-{uuid.uuid4()}"
        task["task_id"] = task_id
        
        logger.info(f"백테스팅 작업 {task_id} 분배 시작 (에이전트: {task.get('agent_name', task.get('agent_id', 'Unknown'))})")
        
        # 라운드 로빈 방식으로 워커 선택
        worker_index = len(self.processing_agents) % len(self.workers)
        worker = self.workers[worker_index]
        
        # 작업 추가
        await worker.add_task({
            "id": task["agent_id"],
            "task_id": task_id,
            "backtest_task": task  # 작업 정보 전달
        })
        
        # 처리 중 목록에 추가
        async with self.lock:
            self.processing_agents.add(task_id)
        
        logger.info(f"백테스팅 작업 {task_id} 워커 {worker.worker_id}에 할당됨")
    
    async def dispatch_agents(self, agents: List[Dict[str, Any]]):
        """
        에이전트 목록을 워커에 분배합니다.
        
        Args:
            agents: 처리할 에이전트 목록
        """
        if not agents:
            logger.info("분배할 에이전트가 없습니다.")
            return
        
        logger.info(f"총 {len(agents)}개 에이전트 분배 시작")
        
        # 라운드 로빈 방식으로 워커에 작업 분배
        worker_index = 0
        dispatched_count = 0
        
        for agent in agents:
            agent_id = agent.get("id")
            
            # 이미 처리 중인 에이전트인지 확인
            async with self.lock:
                if agent_id in self.processing_agents:
                    logger.info(f"에이전트 {agent_id}는 이미 처리 중입니다. 건너뜁니다.")
                    continue
                
                # 처리 중 목록에 추가
                self.processing_agents.add(agent_id)
            
            # 워커 선택 (라운드 로빈)
            worker = self.workers[worker_index]
            worker_index = (worker_index + 1) % len(self.workers)
            
            # 작업 추가
            await worker.add_task(agent)
            dispatched_count += 1
        
        logger.info(f"총 {dispatched_count}개 에이전트 분배 완료")
    
    async def mark_agent_completed(self, agent_id: str):
        """
        에이전트 처리 완료를 표시합니다.
        
        Args:
            agent_id: 완료된 에이전트 ID
        """
        async with self.lock:
            if agent_id in self.processing_agents:
                self.processing_agents.remove(agent_id)
                logger.debug(f"에이전트 {agent_id} 처리 완료 표시")

    async def get_active_agent_count(self):
        """현재 처리 중인 에이전트 수를 반환합니다."""
        async with self.lock:
            return len(self.processing_agents)