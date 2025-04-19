import asyncio
import logging
from datetime import datetime
from investment_agent import run_investment_simulation

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"agent_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("투자 에이전트 시뮬레이션 실행 시작")
    
    try:
        result = asyncio.run(run_investment_simulation())
        
        if result["status"] == "success":
            logger.info(f"시뮬레이션 완료: {result['successful_investments']}/{result['agents_count']} 에이전트가 투자 결정 기록 성공")
        else:
            logger.error(f"시뮬레이션 실패: {result.get('message', '알 수 없는 오류')}")
        
    except Exception as e:
        logger.error(f"시뮬레이션 중 예외 발생: {str(e)}")
    
    logger.info("투자 에이전트 시뮬레이션 종료")