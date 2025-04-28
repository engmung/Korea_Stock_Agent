import os
import logging
import asyncio
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

# Notion DB ID 환경 변수
NOTION_AGENT_DB_ID = os.environ.get("NOTION_AGENT_DB_ID")

class InvestmentAgent:
    """투자 에이전트 관리 클래스"""
    
    def __init__(self, agent_name: Optional[str] = None):
        self.agent_name = agent_name or f"에이전트_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.investment_philosophy = ""  # 옵션으로 유지 (하위 호환성)
        self.target_channels = []
        self.target_keywords = []
        self.recommendation_strength_filter = []  # 매수, 적극매수 등
        self.investment_horizon = []  # 단기, 중기, 장기
        self.created_at = datetime.now()
        self.status = "활성"
        self.page_id = None  # Notion 페이지 ID
    
    @classmethod
    async def create_from_definition(cls, definition: Dict[str, Any]) -> 'InvestmentAgent':
        """정의 데이터로부터 에이전트 생성"""
        agent = cls(definition.get("agent_name"))
        agent.investment_philosophy = definition.get("investment_philosophy", "")  # 옵션으로 유지
        agent.target_channels = definition.get("target_channels", [])
        agent.target_keywords = definition.get("target_keywords", [])
        agent.recommendation_strength_filter = definition.get("recommendation_strength", [])
        agent.investment_horizon = definition.get("investment_horizon", [])
        
        # Notion DB에 저장
        await agent.save_to_notion()
        
        return agent
    
    @classmethod
    async def load_from_notion(cls, page_id: str) -> Optional['InvestmentAgent']:
        """Notion 페이지에서 에이전트 로드"""
        from notion_utils import get_notion_page
        
        try:
            # Notion API로 페이지 데이터 조회
            page_data = await get_notion_page(page_id)
            
            if not page_data:
                logger.error(f"에이전트 페이지를 찾을 수 없음: {page_id}")
                return None
                
            agent = cls()
            agent.page_id = page_id
            
            # 페이지 속성에서 에이전트 데이터 추출
            properties = page_data.get("properties", {})
            
            # 에이전트 이름
            if "에이전트명" in properties and "title" in properties["에이전트명"]:
                title_obj = properties["에이전트명"]["title"]
                if title_obj and len(title_obj) > 0:
                    agent.agent_name = title_obj[0]["plain_text"]
            
            # 투자 철학
            if "투자 철학" in properties and "rich_text" in properties["투자 철학"]:
                text_obj = properties["투자 철학"]["rich_text"]
                if text_obj and len(text_obj) > 0:
                    agent.investment_philosophy = text_obj[0]["plain_text"]
            
            # 타겟 채널
            if "타겟 채널" in properties and "multi_select" in properties["타겟 채널"]:
                agent.target_channels = [item["name"] for item in properties["타겟 채널"]["multi_select"]]
            
            # 키워드
            if "키워드" in properties and "multi_select" in properties["키워드"]:
                agent.target_keywords = [item["name"] for item in properties["키워드"]["multi_select"]]
            
            # 추천 강도
            if "추천 강도" in properties and "multi_select" in properties["추천 강도"]:
                agent.recommendation_strength_filter = [item["name"] for item in properties["추천 강도"]["multi_select"]]
            
            # 투자 기간
            if "투자 기간" in properties and "multi_select" in properties["투자 기간"]:
                agent.investment_horizon = [item["name"] for item in properties["투자 기간"]["multi_select"]]
            
            # 성과 지표
            if "평균 수익률" in properties and "number" in properties["평균 수익률"]:
                agent.avg_return = properties["평균 수익률"]["number"] or 0
                
            if "성공률" in properties and "number" in properties["성공률"]:
                agent.success_rate = properties["성공률"]["number"] or 0
                
            # 상태
            if "현재 상태" in properties and "select" in properties["현재 상태"]:
                status_obj = properties["현재 상태"]["select"]
                if status_obj:
                    agent.status = status_obj["name"]
            
            # 생성일
            if "생성일" in properties and "date" in properties["생성일"]:
                date_obj = properties["생성일"]["date"]
                if date_obj and "start" in date_obj:
                    try:
                        agent.created_at = datetime.fromisoformat(date_obj["start"].replace("Z", "+00:00"))
                    except:
                        agent.created_at = date_obj["start"]
            
            return agent
            
        except Exception as e:
            logger.error(f"에이전트 로드 중 오류: {str(e)}")
            return None
    
    @classmethod
    async def load_from_notion_with_manager(cls, page_id: str, notion_api_manager) -> Optional['InvestmentAgent']:
        """
        API 관리자를 통해 Notion 페이지에서 에이전트 로드
        
        Args:
            page_id: Notion 페이지 ID
            notion_api_manager: Notion API 관리자 인스턴스
            
        Returns:
            InvestmentAgent 객체 또는 None
        """
        try:
            # API 관리자를 통해 페이지 데이터 조회
            page_data = await notion_api_manager.get_notion_page(page_id)
            
            if not page_data:
                logger.error(f"에이전트 페이지를 찾을 수 없음: {page_id}")
                return None
                
            agent = cls()
            agent.page_id = page_id
            
            # 페이지 속성에서 에이전트 데이터 추출
            properties = page_data.get("properties", {})
            
            # 에이전트 이름
            if "에이전트명" in properties and "title" in properties["에이전트명"]:
                title_obj = properties["에이전트명"]["title"]
                if title_obj and len(title_obj) > 0:
                    agent.agent_name = title_obj[0]["plain_text"]
            
            # 투자 철학
            if "투자 철학" in properties and "rich_text" in properties["투자 철학"]:
                text_obj = properties["투자 철학"]["rich_text"]
                if text_obj and len(text_obj) > 0:
                    agent.investment_philosophy = text_obj[0]["plain_text"]
            
            # 타겟 채널
            if "타겟 채널" in properties and "multi_select" in properties["타겟 채널"]:
                agent.target_channels = [item["name"] for item in properties["타겟 채널"]["multi_select"]]
            
            # 키워드
            if "키워드" in properties and "multi_select" in properties["키워드"]:
                agent.target_keywords = [item["name"] for item in properties["키워드"]["multi_select"]]
            
            # 추천 강도
            if "추천 강도" in properties and "multi_select" in properties["추천 강도"]:
                agent.recommendation_strength_filter = [item["name"] for item in properties["추천 강도"]["multi_select"]]
            
            # 투자 기간
            if "투자 기간" in properties and "multi_select" in properties["투자 기간"]:
                agent.investment_horizon = [item["name"] for item in properties["투자 기간"]["multi_select"]]
            
            # 성과 지표
            if "평균 수익률" in properties and "number" in properties["평균 수익률"]:
                agent.avg_return = properties["평균 수익률"]["number"] or 0
                
            if "성공률" in properties and "number" in properties["성공률"]:
                agent.success_rate = properties["성공률"]["number"] or 0
                
            # 상태
            if "현재 상태" in properties and "select" in properties["현재 상태"]:
                status_obj = properties["현재 상태"]["select"]
                if status_obj:
                    agent.status = status_obj["name"]
            
            # 생성일
            if "생성일" in properties and "date" in properties["생성일"]:
                date_obj = properties["생성일"]["date"]
                if date_obj and "start" in date_obj:
                    try:
                        agent.created_at = datetime.fromisoformat(date_obj["start"].replace("Z", "+00:00"))
                    except:
                        agent.created_at = date_obj["start"]
            
            return agent
            
        except Exception as e:
            logger.error(f"에이전트 로드 중 오류: {str(e)}")
            return None
    
    async def save_to_notion(self) -> bool:
        """에이전트 정보를 Notion DB에 저장"""
        from notion_utils import create_investment_agent
        
        try:
            agent_data = {
                "agent_name": self.agent_name,
                "investment_philosophy": self.investment_philosophy,
                "target_channels": self.target_channels,
                "target_keywords": self.target_keywords,
                "recommendation_strength": self.recommendation_strength_filter,
                "investment_horizon": self.investment_horizon,
                "status": self.status,
                "avg_return": self.avg_return,
                "success_rate": self.success_rate
            }
            
            result = await create_investment_agent(agent_data)
            
            if result:
                self.page_id = result.get("id")
                logger.info(f"에이전트 '{self.agent_name}' Notion DB에 저장 완료")
                return True
            else:
                logger.error(f"에이전트 '{self.agent_name}' Notion DB 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"에이전트 저장 중 오류: {str(e)}")
            return False
    
    async def save_to_notion_with_manager(self, notion_api_manager) -> bool:
        """
        API 관리자를 통해 에이전트 정보를 Notion DB에 저장
        
        Args:
            notion_api_manager: Notion API 관리자 인스턴스
            
        Returns:
            저장 성공 여부
        """
        # 이 메서드는 향후 구현 예정
        # 현재는 기존 메서드를 호출
        return await self.save_to_notion()
    
    async def update_performance(self, avg_return: float, success_rate: float) -> bool:
        """에이전트의 성과 지표 업데이트"""
        from notion_utils import update_notion_page
        
        if not self.page_id:
            logger.error("페이지 ID가 없어 업데이트할 수 없습니다.")
            return False
            
        try:
            properties = {
                "평균 수익률": {
                    "number": avg_return
                },
                "성공률": {
                    "number": success_rate
                }
            }
            
            result = await update_notion_page(self.page_id, properties)
            
            if result:
                self.avg_return = avg_return
                self.success_rate = success_rate
                logger.info(f"에이전트 '{self.agent_name}' 성과 지표 업데이트 완료")
                return True
            else:
                logger.error(f"에이전트 '{self.agent_name}' 성과 지표 업데이트 실패")
                return False
                
        except Exception as e:
            logger.error(f"성과 업데이트 중 오류: {str(e)}")
            return False

    async def update_performance_with_manager(self, avg_return: float, success_rate: float, notion_api_manager) -> bool:
        """
        API 관리자를 통해 에이전트의 성과 지표 업데이트
        
        Args:
            avg_return: 평균 수익률
            success_rate: 성공률
            notion_api_manager: Notion API 관리자 인스턴스
            
        Returns:
            업데이트 성공 여부
        """
        if not self.page_id:
            logger.error("페이지 ID가 없어 업데이트할 수 없습니다.")
            return False
            
        try:
            properties = {
                "평균 수익률": {
                    "number": avg_return
                },
                "성공률": {
                    "number": success_rate
                }
            }
            
            result = await notion_api_manager.update_notion_page(self.page_id, properties)
            
            if result:
                self.avg_return = avg_return
                self.success_rate = success_rate
                logger.info(f"에이전트 '{self.agent_name}' 성과 지표 업데이트 완료")
                return True
            else:
                logger.error(f"에이전트 '{self.agent_name}' 성과 지표 업데이트 실패")
                return False
                
        except Exception as e:
            logger.error(f"성과 업데이트 중 오류: {str(e)}")
            return False


async def create_investment_agent(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """투자 에이전트를 생성하고 Notion DB에 저장합니다."""
    try:
        # 에이전트 생성
        agent = await InvestmentAgent.create_from_definition(agent_data)
        
        if agent and agent.page_id:
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
                    "status": agent.status,
                    "created_at": agent.created_at.isoformat() if isinstance(agent.created_at, datetime) else str(agent.created_at)
                }
            }
        else:
            return {
                "status": "error",
                "error": "에이전트 생성에 실패했습니다."
            }
    
    except Exception as e:
        logger.error(f"에이전트 생성 중 오류: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


async def create_investment_agent_with_manager(agent_data: Dict[str, Any], notion_api_manager) -> Dict[str, Any]:
    """
    API 관리자를 통해 투자 에이전트를 생성하고 Notion DB에 저장합니다.
    
    Args:
        agent_data: 에이전트 데이터
        notion_api_manager: Notion API 관리자 인스턴스
        
    Returns:
        생성 결과 및 에이전트 정보
    """
    # 이 함수는 향후 구현 예정
    # 현재는 기존 함수를 호출
    return await create_investment_agent(agent_data)