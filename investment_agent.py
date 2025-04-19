import os
import json
import logging
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
# Assuming re is needed based on the code structure, though not explicitly imported at the top.
# Adding import re here for completeness if it was intended.
import re

from notion_utils import (
    query_notion_database,
    create_investment_agent,
    create_investment_performance,
    increment_citation_count,
    SCRIPT_DB_ID,
    INVESTMENT_AGENT_DB_ID
)
from google import genai
from google.genai import types

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# API 접근을 위한 세마포어
ANALYSIS_SEMAPHORE = asyncio.Semaphore(1)

class InvestmentAgent:
    """투자 신념과 전략을 가진 AI 에이전트"""

    def __init__(
        self,
        agent_id: str,
        investment_philosophy: str,
        status: str = "활성"
    ):
        """
        에이전트 초기화

        Args:
            agent_id: 에이전트 식별자
            investment_philosophy: 투자 철학 설명
            status: 에이전트 상태 (활성/비활성)
        """
        self.agent_id = agent_id
        self.investment_philosophy = investment_philosophy
        self.status = status
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")

        # Notion DB에 저장된 ID (생성 후 설정됨)
        self.notion_id = None

        # 성과 기록
        self.current_portfolio = {}
        self.avg_return = 0
        self.success_rate = 0

    async def register(self) -> bool:
        """에이전트를 Notion DB에 등록합니다."""
        try:
            agent_data = {
                "agent_id": self.agent_id,
                "investment_philosophy": self.investment_philosophy,
                "status": self.status,
                "avg_return": self.avg_return,
                "success_rate": self.success_rate
            }

            result = await create_investment_agent(agent_data)

            if result and "id" in result:
                self.notion_id = result["id"]
                logger.info(f"에이전트 {self.agent_id} 등록 완료 (ID: {self.notion_id})")
                return True
            else:
                logger.error(f"에이전트 {self.agent_id} 등록 실패")
                return False

        except Exception as e:
            logger.error(f"에이전트 등록 중 오류: {str(e)}")
            return False

    async def analyze_reports(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        최근 스크립트 분석 보고서들을 분석하여 투자 기회 탐색

        Args:
            days: 최근 몇 일 동안의 보고서를 분석할지 지정

        Returns:
            추천 종목 목록
        """
        # 최근 보고서 가져오기
        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        # 필터를 위한 요청 본문
        request_body = {
            "filter": {
                "property": "영상 날짜",
                "date": {
                    "on_or_after": since_date
                }
            },
            "sorts": [
                {
                    "property": "영상 날짜",
                    "direction": "descending"
                }
            ]
        }

        reports = await query_notion_database(SCRIPT_DB_ID, request_body)
        logger.info(f"최근 {days}일 동안의 보고서 {len(reports)}개를 찾았습니다.")

        if not reports:
            logger.warning("분석할 보고서가 없습니다.")
            return []

        # 보고서 내용 추출
        report_contents = []

        for report in reports:
            properties = report.get("properties", {})
            page_id = report.get("id")

            # 제목 가져오기
            title = ""
            title_property = properties.get("제목", {})
            if "title" in title_property and title_property["title"]:
                title = title_property["title"][0]["plain_text"].strip()

            # 채널명 가져오기
            channel = ""
            channel_property = properties.get("채널명", {})
            if "select" in channel_property and channel_property["select"]:
                channel = channel_property["select"]["name"]

            # URL 가져오기
            url = ""
            url_property = properties.get("URL", {})
            if "url" in url_property:
                url = url_property["url"]

            # 영상 날짜 가져오기
            video_date = ""
            date_property = properties.get("영상 날짜", {})
            if "date" in date_property and date_property["date"]:
                video_date = date_property["date"]["start"]

            # 페이지 본문 가져오기
            # Notion API는 페이지 내용을 별도로 가져와야 함
            # 페이지 내용은 실제로는 분석에 사용하지 않고, 인용 횟수만 증가시킴

            # 인용 횟수 증가
            # await increment_citation_count(page_id) # Uncomment if increment_citation_count is implemented and needed

            report_contents.append({
                "id": page_id,
                "title": title,
                "channel": channel,
                "url": url,
                "video_date": video_date
            })

        # LLM으로 보고서 분석
        return await self._analyze_with_llm(report_contents)

    async def _analyze_with_llm(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        LLM을 사용하여 보고서를 분석하고 투자 결정을 내립니다.
        """
        if not self.gemini_api_key:
            logger.error("GEMINI_API_KEY가 설정되지 않았습니다.")
            return []

        # API 요청 제한 관리
        async with ANALYSIS_SEMAPHORE:
            try:
                # Gemini 클라이언트 설정
                # client = genai.Client(api_key=self.gemini_api_key) # genai.Client is deprecated
                genai.configure(api_key=self.gemini_api_key)
                model = "gemini-2.5-flash-preview-04-17"  # 최신 모델 사용 - Check if this model is available or use a standard one like "gemini-1.5-pro-latest" or "gemini-1.0-pro"

                # 보고서 정보로 프롬프트 생성
                reports_text = "\n\n".join([
                    f"Report #{i+1}:\nTitle: {r['title']}\nChannel: {r['channel']}\nDate: {r['video_date']}\nURL: {r['url']}"
                    for i, r in enumerate(reports[:5])  # 최대 5개 보고서만 사용
                ])

                prompt = f"""# 투자 결정 요청

## 내 투자 철학
{self.investment_philosophy}

## 분석할 최근 보고서들
{reports_text}

위 보고서들에 근거하여, 내 투자 철학에 맞는 종목 5-10개를 추천해주세요.
각 종목에 투자 비중과 투자 이유를 함께 제공해주세요."""

                # 시스템 지시사항 설정
                system_instruction = """당신은 투자 에이전트로서 제공된 투자 철학에 따라 종목을 추천합니다.

다음 형식으로 응답해주세요:
```json
[
{
"ticker": "종목코드",
"name": "종목명",
"weight": 투자비중(숫자),
"reasons": ["투자이유1", "투자이유2", ...],
"investment_period": "단기" 또는 "중기" 또는 "장기"
},
...
]

종목코드는 한국 주식 6자리 코드를 사용하세요 (알 수 없을 경우 빈 문자열)
투자비중은 0-100 사이의 숫자로, 모든 종목의 비중 합은 100이 되어야 합니다
투자이유는 간결하게 3-5개 항목으로 작성하세요
투자기간은 "단기"(1개월 이내), "중기"(1-3개월), "장기"(3개월 이상) 중 하나를 선택하세요

보고서의 내용을 기반으로 최선의 추천을 해주세요. 보고서에 없는 정보는 추가하지 마세요."""
                # API 호출 준비
                # Using genai.GenerativeModel directly as Client is deprecated
                model_instance = genai.GenerativeModel(
                    model_name=model,
                    system_instruction=system_instruction
                )

                # API 호출
                logger.info(f"에이전트 {self.agent_id}의 Gemini API 호출 시작")
                response = await asyncio.to_thread(
                    model_instance.generate_content,
                    contents=prompt,
                    # No generate_content_config needed for response_mime_type="text/plain" and system_instruction here
                )

                if hasattr(response, 'text') and response.text:
                    # JSON 추출
                    try:
                        json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # Attempt to parse the entire response if no code block is found
                             json_str = response.text.strip()

                        # JSON 파싱
                        recommendations = json.loads(json_str)
                        logger.info(f"에이전트 {self.agent_id}가 {len(recommendations)}개 종목 추천")
                        return recommendations
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 파싱 오류: {str(e)}, 응답: {response.text}")
                        return []
                else:
                    logger.error("Gemini API가 빈 응답을 반환했습니다.")
                    return []

            except Exception as e:
                logger.error(f"Gemini API 호출 중 오류: {str(e)}")
                return []

    async def make_investment_decision(self, recommendations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        추천 종목들을 기반으로 최종 투자 결정을 내립니다.
        """
        if not recommendations:
            logger.warning("추천 종목이 없어 투자 결정을 내릴 수 없습니다.")
            return None

        # 간단한 필터링 (예: 너무 작은 비중의 종목 제외)
        filtered_recommendations = [r for r in recommendations if r.get("weight", 0) >= 5]

        if not filtered_recommendations:
             logger.warning("필터링 후 남은 추천 종목이 없습니다.")
             return None

        # 비중 재조정
        total_weight = sum(r.get("weight", 0) for r in filtered_recommendations)
        if total_weight > 0:
            for r in filtered_recommendations:
                r["weight"] = (r.get("weight", 0) / total_weight) * 100
        else:
             logger.warning("필터링 후 남은 추천 종목들의 비중 합이 0입니다.")
             return None


        # 현재 날짜로 포트폴리오 생성
        now = datetime.now()
        start_date = now.isoformat()

        # 투자 기간 설정 (단기/중기/장기에 따라 다름)
        investment_periods_days = {
            "단기": 7,   # 7일
            "중기": 30,  # 30일
            "장기": 90  # 90일
        }

        # 가장 많은 비중을 차지하는, 혹은 가장 자주 나오는 투자 기간 선택
        period_counts = {"단기": 0, "중기": 0, "장기": 0}
        for r in filtered_recommendations:
            period = r.get("investment_period", "중기")
            if period in period_counts:
                period_counts[period] += r.get("weight", 0)

        primary_period = max(period_counts.items(), key=lambda x: x[1])[0]
        days = investment_periods_days.get(primary_period, 30)

        end_date = (now + timedelta(days=days)).isoformat()

        # 투자 종목 및 비중 문자열 생성
        stocks = [f"{r.get('name', '')}({r.get('ticker', '')})" for r in filtered_recommendations]
        weights = ", ".join([f"{r.get('name', '')}: {r.get('weight', 0):.1f}%" for r in filtered_recommendations])

        # 투자 결정 생성
        decision = {
            "title": f"{self.agent_id} - {now.strftime('%Y-%m-%d')} 투자",
            "agent_id": self.agent_id,
            "agent_id_relation": self.notion_id,
            "start_date": datetime.fromisoformat(start_date),
            "end_date": datetime.fromisoformat(end_date),
            "stocks": stocks,
            "weights": weights,
            "recommendations": filtered_recommendations,
            "primary_period": primary_period
        }

        return decision

    async def record_investment(self, decision: Dict[str, Any]) -> bool:
        """
        투자 결정을 Notion DB에 기록합니다.
        """
        try:
            # 임의의 성과 데이터 생성 (실제로는 백테스팅 결과로 대체)
            total_return = random.uniform(-5.0, 15.0)  # -5% ~ +15% 수익률
            max_drawdown = random.uniform(-10.0, 0.0)  # -10% ~ 0% 낙폭

            # 결과 평가 (수익률에 따라)
            if total_return > 10:
                evaluation = "성공"
            elif total_return > 0:
                evaluation = "부분 성공"
            else:
                evaluation = "실패"

            # 성과 데이터 추가
            performance_data = {
                "title": decision["title"],
                "agent_id_relation": decision["agent_id_relation"],
                "start_date": decision["start_date"],
                "end_date": decision["end_date"],
                "stocks": decision["stocks"],
                "weights": decision["weights"],
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "evaluation": evaluation
            }

            # 투자 실적 생성
            result = await create_investment_performance(performance_data)

            return result is not None

        except Exception as e:
            logger.error(f"투자 기록 중 오류: {str(e)}")
            return False

async def create_random_agents(count: int = 5) -> List[InvestmentAgent]:
    """
    다양한 투자 철학을 가진 에이전트들을 무작위로 생성합니다.
    """
    # 투자 철학 템플릿
    philosophies = [
        "나는 가치투자자입니다. 내재가치 대비 저평가된 기업을 선호합니다. PER이 낮고 배당수익률이 높은 기업에 집중 투자합니다. 재무제표가 안정적이고 장기적으로 경쟁력 있는 기업을 선택합니다.",
        "나는 성장투자자입니다. 높은 매출 성장률과 미래 성장 가능성이 큰 기업을 찾습니다. 신기술과 혁신 기업에 투자하며, 수익성보다는 성장성에 중점을 둡니다.",
        "나는 기술적 분석을 중시합니다. 차트 패턴, 거래량, 이동평균선 등 기술적 지표를 활용해 진입 시점을 결정합니다. 추세 추종 전략을 사용하며 모멘텀이 강한 종목을 선호합니다.",
        "나는 섹터 순환 전략을 사용합니다. 경기 사이클에 따라 유망 섹터를 선별하고 해당 섹터 내 우량주에 투자합니다. 경기선행지표와 산업 트렌드를 중요하게 생각합니다.",
        "나는 배당투자자입니다. 안정적인 배당금을 제공하는 기업에 투자합니다. 배당성장률과 배당지속성을 중요시하며, 낮은 변동성과 꾸준한 현금흐름을 추구합니다.",
        "나는 역발상 투자자입니다. 시장의 과도한 반응을 이용해 과매도된 종목에 투자합니다. 단기적 악재로 인해 가격이 하락한 우량 기업을 찾아 저가 매수합니다.",
        "나는 테마 투자자입니다. 특정 산업이나 트렌드에 집중 투자합니다. AI, 반도체, 전기차 등 주요 성장 테마를 주시하며 해당 분야 선도 기업에 투자합니다.",
        "나는 퀀트 투자자입니다. 정량적 지표를 기반으로 종목을 선별합니다. ROE, PBR, FCF 등 재무지표를 분석하여 시스템적으로 투자 결정을 내립니다.",
        "나는 뉴스 기반 투자자입니다. 전문가 의견과 시장 뉴스를 중요시합니다. 기관 투자자의 움직임과 애널리스트 보고서를 참고하여 투자 결정을 내립니다.",
        "나는 단기 트레이더입니다. 시장의 단기적 변동성을 활용합니다. 빠른 매매로 작은 이익을 자주 실현하는 방식을 선호하며 손절과 이익실현 규칙을 엄격히 지킵니다."
    ]

    agents = []

    for i in range(count):
        philosophy = random.choice(philosophies)
        agent_id = f"Agent_{i+1:02d}_{datetime.now().strftime('%m%d')}"

        agent = InvestmentAgent(
            agent_id=agent_id,
            investment_philosophy=philosophy
        )

        # 에이전트 등록
        registration_success = await agent.register()

        if registration_success:
            agents.append(agent)
            logger.info(f"에이전트 {agent_id} 생성 완료")
        else:
            logger.warning(f"에이전트 {agent_id} 생성 실패, 다음 에이전트로 진행")

    return agents

async def run_investment_simulation() -> Dict[str, Any]:
    """
    투자 시뮬레이션을 실행합니다.
    1. 에이전트 생성
    2. 각 에이전트가 최근 보고서 분석
    3. 투자 결정 내리기
    4. 투자 결과 기록
    """
    logger.info("투자 시뮬레이션 시작")
    # 에이전트 생성
    agents = await create_random_agents(5)

    if not agents:
        logger.error("에이전트 생성 실패, 시뮬레이션 종료")
        return {"status": "error", "message": "에이전트 생성 실패"}

    results = []

    # 각 에이전트에 대해 프로세스 실행
    for agent in agents:
        try:
            # 1. 최근 보고서 분석
            recommendations = await agent.analyze_reports(days=30)

            if not recommendations:
                logger.warning(f"에이전트 {agent.agent_id}의 추천 종목이 없습니다.")
                continue

            # 2. 투자 결정
            decision = await agent.make_investment_decision(recommendations)

            if not decision:
                logger.warning(f"에이전트 {agent.agent_id}의 투자 결정 실패")
                continue

            # 3. 투자 결과 기록
            record_success = await agent.record_investment(decision)

            results.append({
                "agent_id": agent.agent_id,
                "notion_id": agent.notion_id,
                "recommendations_count": len(recommendations),
                "primary_period": decision["primary_period"],
                "record_success": record_success
            })

            logger.info(f"에이전트 {agent.agent_id}의 투자 프로세스 완료")

        except Exception as e:
            logger.error(f"에이전트 {agent.agent_id} 처리 중 오류: {str(e)}")

    return {
        "status": "success",
        "agents_count": len(agents),
        "successful_investments": len([r for r in results if r["record_success"]]),
        "results": results
    }

# Example of how to run the simulation (assuming an asyncio event loop)
if __name__ == "__main__":
    async def main():
        simulation_results = await run_investment_simulation()
        print(json.dumps(simulation_results, indent=4, ensure_ascii=False))

    # Ensure the event loop is closed cleanly
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("시뮬레이션 중단됨.")