import os
import asyncio
import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 모듈 가져오기
from api_managers import NotionAPIManager, GeminiAPIManager
from workers import BacktestDispatcher

# Notion DB ID 환경 변수
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
NOTION_AGENT_DB_ID = os.environ.get("NOTION_AGENT_DB_ID")


class BacktestParallelSystem:
    """병렬 백테스팅 시스템의 메인 클래스."""
    
    def __init__(self, num_workers: int = 3):
        """
        병렬 백테스팅 시스템을 초기화합니다.
        
        Args:
            num_workers: 사용할 워커 수 (기본값: 3)
        """
        # API 키 확인
        if not NOTION_API_KEY:
            raise ValueError("NOTION_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        if not NOTION_AGENT_DB_ID:
            raise ValueError("NOTION_AGENT_DB_ID 환경 변수가 설정되지 않았습니다.")
        
        # API 관리자 생성
        self.notion_api_manager = NotionAPIManager(api_key=NOTION_API_KEY)
        self.gemini_api_manager = GeminiAPIManager()
        
        # 디스패처 생성
        self.dispatcher = BacktestDispatcher(
            notion_api_manager=self.notion_api_manager,
            gemini_api_manager=self.gemini_api_manager,
            num_workers=num_workers
        )
        
        logger.info(f"병렬 백테스팅 시스템 초기화 완료 (워커 {num_workers}개)")
    
    async def start(self):
        """시스템을 시작합니다."""
        # API 관리자 시작
        await self.notion_api_manager.start()
        
        # 디스패처 시작
        await self.dispatcher.start()
        
        logger.info("병렬 백테스팅 시스템 시작됨")
    
    async def stop(self):
        """시스템을 중지합니다."""
        # 디스패처 중지
        await self.dispatcher.stop()
        
        # API 관리자 중지
        await self.notion_api_manager.stop()
        
        logger.info("병렬 백테스팅 시스템 중지됨")
    
    async def run_backtest_cycle(self):
        """
        백테스팅 사이클을 실행합니다.
        1. 에이전트 목록 조회
        2. 디스패처를 통해 에이전트 분배
        """
        logger.info("백테스팅 사이클 시작")
        
        try:
            # 에이전트 목록 조회
            agents = await self.notion_api_manager.query_notion_database(NOTION_AGENT_DB_ID)
            
            if not agents:
                logger.info("처리할 에이전트가 없습니다.")
                return
            
            logger.info(f"총 {len(agents)}개 에이전트 조회됨")
            
            # 에이전트 분배
            await self.dispatcher.dispatch_agents(agents)
            
        except Exception as e:
            logger.error(f"백테스팅 사이클 실행 중 오류: {str(e)}")
    
    async def wait_for_completion(self, timeout: float = None):
        """
        모든 작업이 완료될 때까지 대기합니다.
        
        Args:
            timeout: 최대 대기 시간 (초), None이면 무제한 대기
        
        Returns:
            bool: 모든 작업이 완료되었으면 True, 타임아웃이면 False
        """
        start_time = datetime.now()
        
        while True:
            # 처리 중인 에이전트 수 확인
            active_count = await self.dispatcher.get_active_agent_count()
            
            if active_count == 0:
                logger.info("모든 작업이 완료되었습니다.")
                return True
            
            # 타임아웃 확인
            if timeout is not None:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > timeout:
                    logger.warning(f"타임아웃: {active_count}개 작업이 아직 처리 중입니다.")
                    return False
            
            # 잠시 대기 후 다시 확인
            await asyncio.sleep(5)


async def main():
    """메인 함수."""
    # 기본 워커 수 설정 (환경 변수에서 가져오거나 기본값 3 사용)
    num_workers = int(os.environ.get("BACKTEST_NUM_WORKERS", 3))
    
    # 병렬 백테스팅 시스템 생성
    system = BacktestParallelSystem(num_workers=num_workers)
    
    try:
        # 시스템 시작
        await system.start()
        
        # 백테스팅 사이클 실행
        await system.run_backtest_cycle()
        
        # 모든 작업 완료 대기 (최대 1시간)
        await system.wait_for_completion(timeout=3600)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
    finally:
        # 시스템 중지
        await system.stop()


# 스크립트가 직접 실행될 때만 메인 함수 호출
if __name__ == "__main__":
    asyncio.run(main())