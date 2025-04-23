import os
import logging
import httpx
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Notion API 설정
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")
NOTION_AGENT_DB_ID = os.environ.get("NOTION_AGENT_DB_ID")
NOTION_SCRIPT_DB_ID = os.environ.get("NOTION_SCRIPT_DB_ID")
NOTION_PERFORMANCE_DB_ID = os.environ.get("NOTION_PERFORMANCE_DB_ID")

# 로깅 설정 - 간결하게 변경
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 환경 변수 검증
if not NOTION_API_KEY:
    logger.error("NOTION_API_KEY 환경 변수가 설정되지 않았습니다.")
    raise ValueError("NOTION_API_KEY 환경 변수가 필요합니다.")

if not NOTION_AGENT_DB_ID:
    logger.error("NOTION_AGENT_DB_ID 환경 변수가 설정되지 않았습니다.")
    raise ValueError("NOTION_AGENT_DB_ID 환경 변수가 필요합니다.")

if not NOTION_SCRIPT_DB_ID:
    logger.error("NOTION_SCRIPT_DB_ID 환경 변수가 설정되지 않았습니다.")
    raise ValueError("NOTION_SCRIPT_DB_ID 환경 변수가 필요합니다.")

async def query_notion_database(database_id: str, request_body: dict = None, max_retries: int = 3, timeout: float = 30.0) -> List[Dict[str, Any]]:
    """
    Notion 데이터베이스를 쿼리합니다. 재시도 및 타임아웃 처리가 포함되어 있습니다.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    if request_body is None:
        request_body = {}
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, 
                    headers=headers, 
                    json=request_body, 
                    timeout=timeout
                )
                
                response.raise_for_status()
                results = response.json().get("results", [])
                logger.info(f"Notion DB 쿼리 성공: {len(results)}개 레코드")
                return results
                
        except httpx.TimeoutException:
            logger.info(f"Notion DB 쿼리 타임아웃 (재시도 {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return []
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 오류: {e.response.status_code}")
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                await asyncio.sleep(retry_after)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Notion DB 쿼리 오류: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return []
    
    return []

async def get_notion_page(page_id: str, max_retries: int = 3, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """
    Notion 페이지를 조회합니다.
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28"
    }
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, 
                    headers=headers, 
                    timeout=timeout
                )
                
                response.raise_for_status()
                page_data = response.json()
                logger.info(f"Notion 페이지 조회 성공: {page_id}")
                return page_data
                
        except Exception as e:
            logger.error(f"Notion 페이지 조회 오류: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return None
    
    return None

async def get_notion_page_content(page_id: str, max_retries: int = 3, timeout: float = 30.0) -> str:
    """
    Notion 페이지 컨텐츠(블록)를 조회합니다. 블록 제한 없음.
    """
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28"
    }
    
    # 쿼리 파라미터 - 페이지 크기를 최대로 설정 (API 제한 100)
    params = {
        "page_size": 100  # Notion API 최대 제한
    }
    
    all_blocks = []
    has_more = True
    next_cursor = None
    
    # 페이지네이션을 사용하여 모든 블록 가져오기
    while has_more:
        # 커서가 있으면 쿼리 파라미터에 추가
        if next_cursor:
            params["start_cursor"] = next_cursor
        
        # 재시도 로직
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url, 
                        headers=headers,
                        params=params,
                        timeout=timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    blocks = result.get("results", [])
                    all_blocks.extend(blocks)
                    
                    # 다음 페이지가 있는지 확인
                    has_more = result.get("has_more", False)
                    next_cursor = result.get("next_cursor")
                    
                    logger.info(f"블록 조회: {len(blocks)}개, 총 {len(all_blocks)}개")
                    break  # 성공적으로 가져왔으므로 재시도 루프 종료
                    
            except Exception as e:
                logger.error(f"블록 조회 오류: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    has_more = False  # 더 이상 시도하지 않음
                    break
    
    # 블록을 마크다운으로 변환
    content = blocks_to_markdown(all_blocks)
    
    return content

def blocks_to_markdown(blocks: List[Dict[str, Any]]) -> str:
    """
    Notion 블록을 마크다운 형식으로 변환합니다.
    """
    markdown = []
    
    for block in blocks:
        block_type = block.get("type")
        
        # 블록 타입에 따라 마크다운 변환
        if block_type == "paragraph":
            text = rich_text_to_plain(block.get("paragraph", {}).get("rich_text", []))
            markdown.append(text)
            markdown.append("")  # 빈 줄 추가
            
        elif block_type == "heading_1":
            text = rich_text_to_plain(block.get("heading_1", {}).get("rich_text", []))
            markdown.append(f"# {text}")
            markdown.append("")
            
        elif block_type == "heading_2":
            text = rich_text_to_plain(block.get("heading_2", {}).get("rich_text", []))
            markdown.append(f"## {text}")
            markdown.append("")
            
        elif block_type == "heading_3":
            text = rich_text_to_plain(block.get("heading_3", {}).get("rich_text", []))
            markdown.append(f"### {text}")
            markdown.append("")
            
        elif block_type == "bulleted_list_item":
            text = rich_text_to_plain(block.get("bulleted_list_item", {}).get("rich_text", []))
            markdown.append(f"- {text}")
            
        elif block_type == "numbered_list_item":
            text = rich_text_to_plain(block.get("numbered_list_item", {}).get("rich_text", []))
            markdown.append(f"1. {text}")
            
        elif block_type == "to_do":
            text = rich_text_to_plain(block.get("to_do", {}).get("rich_text", []))
            checked = block.get("to_do", {}).get("checked", False)
            markdown.append(f"- {'[x]' if checked else '[ ]'} {text}")
            
        elif block_type == "quote":
            text = rich_text_to_plain(block.get("quote", {}).get("rich_text", []))
            markdown.append(f"> {text}")
            markdown.append("")
            
        elif block_type == "code":
            text = rich_text_to_plain(block.get("code", {}).get("rich_text", []))
            language = block.get("code", {}).get("language", "")
            markdown.append(f"```{language}")
            markdown.append(text)
            markdown.append("```")
            markdown.append("")
            
        elif block_type == "divider":
            markdown.append("---")
            markdown.append("")
            
        elif block_type == "callout":
            text = rich_text_to_plain(block.get("callout", {}).get("rich_text", []))
            emoji = block.get("callout", {}).get("icon", {}).get("emoji", "")
            markdown.append(f"> {emoji} {text}")
            markdown.append("")
            
        elif block_type == "toggle":
            text = rich_text_to_plain(block.get("toggle", {}).get("rich_text", []))
            markdown.append(f"<details><summary>{text}</summary>")
            markdown.append("</details>")
            markdown.append("")
    
    return "\n".join(markdown)

def rich_text_to_plain(rich_text: List[Dict[str, Any]]) -> str:
    """
    Notion의 rich_text 객체를 일반 텍스트로 변환합니다.
    """
    if not rich_text:
        return ""
    
    text_parts = []
    
    for part in rich_text:
        text = part.get("plain_text", "")
        annotations = part.get("annotations", {})
        
        # 볼드
        if annotations.get("bold"):
            text = f"**{text}**"
            
        # 이탤릭
        if annotations.get("italic"):
            text = f"*{text}*"
            
        # 코드
        if annotations.get("code"):
            text = f"`{text}`"
            
        # 취소선
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
            
        text_parts.append(text)
    
    return "".join(text_parts)

async def create_investment_agent(agent_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    투자 에이전트 DB에 새 에이전트를 생성합니다.
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 에이전트명 확인 및 기본값 제공
    agent_name = agent_data.get("agent_name", "")
    if not agent_name:
        agent_name = f"에이전트_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # 투자 철학
    investment_philosophy = agent_data.get("investment_philosophy", "")
    if investment_philosophy:
        if len(investment_philosophy) > 2000:  # Notion 제한
            investment_philosophy = investment_philosophy[:1997] + "..."
            
        properties["투자 철학"] = {
            "rich_text": [
                {
                    "text": {
                        "content": investment_philosophy
                    }
                }
            ]
        }
    
    # 페이지 속성 설정
    properties = {
        "에이전트명": {
            "title": [
                {
                    "text": {
                        "content": agent_name
                    }
                }
            ]
        }
    }
    
    # 투자 철학 추가 (빈 문자열이 아닌 경우만)
    if investment_philosophy:
        properties["투자 철학"] = {
            "rich_text": [
                {
                    "text": {
                        "content": investment_philosophy
                    }
                }
            ]
        }
    
    
    # 현재 상태 - 기본 속성으로 가정
    properties["현재 상태"] = {
        "select": {
            "name": agent_data.get("status", "활성")
        }
    }
    
    # 생성일
    properties["생성일"] = {
        "date": {
            "start": datetime.now().isoformat()
        }
    }
    
    # 타겟 채널 - 멀티 셀렉트 객체 생성
    target_channels = []
    for channel in agent_data.get("target_channels", []):
        if channel and isinstance(channel, str):
            target_channels.append({
                "name": channel
            })
    
    if target_channels:
        properties["타겟 채널"] = {
            "multi_select": target_channels
        }
    
    # 키워드 - 멀티 셀렉트 객체 생성
    target_keywords = []
    for keyword in agent_data.get("target_keywords", []):
        if keyword and isinstance(keyword, str):
            target_keywords.append({
                "name": keyword
            })
    
    if target_keywords:
        properties["키워드"] = {
            "multi_select": target_keywords
        }
    
    # 추천 강도 - 멀티 셀렉트 객체 생성
    recommendation_strength = []
    for strength in agent_data.get("recommendation_strength", []):
        if strength and isinstance(strength, str):
            recommendation_strength.append({
                "name": strength
            })
    
    if recommendation_strength:
        properties["추천 강도"] = {
            "multi_select": recommendation_strength
        }
    
    # 투자 기간 - 멀티 셀렉트 객체 생성
    investment_horizon = []
    for horizon in agent_data.get("investment_horizon", []):
        if horizon and isinstance(horizon, str):
            investment_horizon.append({
                "name": horizon
            })
    
    if investment_horizon:
        properties["투자 기간"] = {
            "multi_select": investment_horizon
        }
    
    # 요청 본문 생성
    request_data = {
        "parent": {
            "database_id": NOTION_AGENT_DB_ID
        },
        "properties": properties
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=request_data,
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"투자 에이전트 생성 성공: {agent_name}")
            return result
            
    except Exception as e:
        logger.error(f"투자 에이전트 생성 실패: {str(e)}")
        return None

async def update_notion_page(page_id: str, properties: Dict[str, Any], max_retries: int = 3, timeout: float = 30.0) -> bool:
    """
    Notion 페이지의 속성을 업데이트합니다.
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    request_data = {
        "properties": properties
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                url,
                headers=headers,
                json=request_data,
                timeout=timeout
            )
            
            response.raise_for_status()
            logger.info(f"Notion 페이지 업데이트 성공: {page_id}")
            return True
                
    except Exception as e:
        logger.error(f"Notion 페이지 업데이트 실패: {str(e)}")
        return False

async def add_structured_content_to_notion_page(page_id: str, debug_info: Dict[str, Any], title: str = "백테스팅 결과") -> bool:
    """
    노션 페이지에 구조화된 형식으로 백테스팅 결과를 추가합니다.
    """
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 기본 헤더 블록 추가
    try:
        # 단계별로 블록 추가 - 첫 번째: 헤더와 기본 정보
        basic_blocks = [
            # 헤딩 블록 추가
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": title}}]
                }
            }
        ]
        
        # 투자 기간 정보 추가
        if "start_date" in debug_info and "end_date" in debug_info:
            basic_blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"📅 투자 기간: {debug_info.get('start_date', '')} ~ {debug_info.get('end_date', '')}"}}]
                }
            })
        
        # 먼저 기본 블록 추가
        basic_request = {"children": basic_blocks}
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                url, 
                headers=headers, 
                json=basic_request,
                timeout=30.0
            )
            response.raise_for_status()
            logger.info(f"기본 블록 추가 성공: {page_id}")
        
        # 보고서 선택 전략 정보 추가
        if "report_selection_result" in debug_info:
            selection_info = debug_info["report_selection_result"]
            
            # 헤더 및 선택 전략 추가
            selection_blocks = {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "📊 데이터 선택 전략"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": f"선택 전략: {selection_info.get('selection_strategy', '기본 선택')}"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": f"후보 보고서: {selection_info.get('total_candidates', 0)}개 중 {selection_info.get('selected_count', 0)}개 선택"}}]
                        }
                    }
                ]
            }
            
            # 페이지에 블록 추가
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=selection_blocks,
                    timeout=30.0
                )
                response.raise_for_status()
            
            # 선택된 보고서 상세 정보 추가 (최대 10개만 표시)
            if "selection_details" in selection_info and selection_info["selection_details"]:
                details = selection_info["selection_details"]
                
                # 최대 10개만 표시
                display_details = details[:min(10, len(details))]
                
                details_blocks = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": "선택된 보고서 목록:"}}]
                            }
                        }
                    ]
                }
                
                # 각 보고서별 정보 추가 - 개선된 형식으로 표시
                for idx, detail in enumerate(display_details):
                    report_id = detail.get("id", "")
                    title = detail.get("title", "제목 없음")
                    channel = detail.get("channel", "채널 없음")
                    date = detail.get("date", "날짜 없음")
                    
                    details_blocks["children"].append({
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{"type": "text", "text": {"content": f"{idx+1}. [{channel}] {title} ({date})"}}]
                        }
                    })
                
                # 더 많은 보고서가 있는 경우 메시지 추가
                if len(details) > 10:
                    details_blocks["children"].append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": f"외 {len(details) - 10}개 보고서..."}}]
                        }
                    })
                
                # 선택된 보고서 상세 정보 블록 추가
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=details_blocks,
                        timeout=30.0
                    )
                    response.raise_for_status()
                
                # 구분선 추가
                divider_block = {
                    "children": [
                        {
                            "object": "block",
                            "type": "divider",
                            "divider": {}
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=divider_block,
                        timeout=30.0
                    )
                    response.raise_for_status()
            
            logger.info(f"보고서 선택 정보 블록 추가 성공: {page_id}")
            
        # 이하 코드는 동일하게 유지 (성과 요약, 종목별 성과, 추천 종목, 분석 텍스트 등)
        # 성과 요약 정보 추가
        if "performance_metrics" in debug_info:
            metrics = debug_info["performance_metrics"]
            summary_blocks = [
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": "📊 포트폴리오 성과 요약"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"• 총 수익률: {metrics.get('portfolio_return', 0):.2f}%"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"• 총 수익금: {metrics.get('portfolio_profit', 0):,.0f}원"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"• 승률: {metrics.get('win_rate', 0):.1f}%"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"• 평균 낙폭: {metrics.get('avg_max_drawdown', 0):.2f}%"}}]
                    }
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": f"• 결과 평가: {metrics.get('evaluation', '평가 없음')}"}}]
                    }
                }
            ]
            
            # 성과 요약 블록 추가
            summary_request = {"children": summary_blocks}
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=summary_request,
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"성과 요약 블록 추가 성공: {page_id}")
            
        # 종목별 성과 추가
        if "backtest_result" in debug_info and "stock_results" in debug_info["backtest_result"]:
            backtest = debug_info["backtest_result"]
            stock_results = backtest.get("stock_results", [])
            
            # 종목 매핑 정보
            stock_ticker_mapping = debug_info.get("stock_ticker_mapping", {})
            
            # 헤더 먼저 추가
            header_block = {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "📈 종목별 성과"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": "종목별 수익률 및 가격 정보:"}}]
                        }
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=header_block,
                    timeout=30.0
                )
                response.raise_for_status()
                
            # 각 종목별로 별도 항목 추가
            for item in backtest.get("performance_summary", []):
                # 같은 티커의 stock_results 항목 찾기
                ticker = item.get("ticker", "")
                stock_detail = next((s for s in stock_results if s.get("ticker") == ticker), {})
                
                # 종목명 가져오기 - 매핑, 실제 종목명, 추천정보 순으로 확인
                stock_name = stock_ticker_mapping.get(ticker, "")
                
                if not stock_name:
                    # 결과에 저장된 종목명 확인
                    stock_name = stock_detail.get("name", "")
                    
                # 종목명이 "종목 코드" 형식이면 클린업
                if not stock_name or (stock_name.startswith("종목 ") and ticker in stock_name):
                    # 추천 종목 정보에서 찾기
                    if "recommendations" in debug_info and "recommended_stocks" in debug_info["recommendations"]:
                        recommended_stocks = debug_info["recommendations"]["recommended_stocks"]
                        for rec_stock in recommended_stocks:
                            if rec_stock.get("ticker") == ticker:
                                stock_name = rec_stock.get("name", "")
                                break
                
                # 그래도 없거나 "종목 코드" 형식이면 정리
                if not stock_name or stock_name.startswith("종목 "):
                    stock_name = ticker  # 티커 코드라도 보여주기
                
                # 종목 정보 블록
                stock_blocks = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text", 
                                        "text": {"content": f"🔹 {stock_name} ({ticker})"}, 
                                        "annotations": {"bold": True}
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"수익률: {item.get('profit_percentage', 0):.2f}%"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"구매가격: {stock_detail.get('initial_price', 0):,.0f}원"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"판매가격: {stock_detail.get('final_price', 0):,.0f}원"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"수익금: {item.get('profit', 0):,.0f}원"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"최종 평가액: {item.get('final_value', 0):,.0f}원"}}]
                            }
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=stock_blocks,
                        timeout=30.0
                    )
                    response.raise_for_status()
            
            logger.info(f"종목별 성과 블록 추가 성공: {page_id}")
        
        # 추천 종목 정보 추가
        if "recommendations" in debug_info and "recommended_stocks" in debug_info["recommendations"]:
            recommended_stocks = debug_info["recommendations"]["recommended_stocks"]
            portfolio_logic = debug_info["recommendations"].get("portfolio_logic", "")
            
            # 헤더와 포트폴리오 논리 추가
            recommendation_header = {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "🔍 추천 종목 분석"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": portfolio_logic}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "divider",
                        "divider": {}
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=recommendation_header,
                    timeout=30.0
                )
                response.raise_for_status()
            
            # 각 종목별 정보 추가
            for stock in recommended_stocks:
                # 티커 정보 추가
                ticker_display = f" ({stock.get('ticker', '')})" if "ticker" in stock else ""
                
                stock_info = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text", 
                                        "text": {"content": f"🔹 {stock.get('name', '')}{ticker_display}"}, 
                                        "annotations": {"bold": True}
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"예상 수익률: {stock.get('expected_return', '미제공')}"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"위험도: {stock.get('risk_level', '미제공')}"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": f"추천 이유: {stock.get('reasoning', '미제공')}"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "divider",
                            "divider": {}
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=stock_info,
                        timeout=30.0
                    )
                    response.raise_for_status()
            
            logger.info(f"추천 종목 블록 추가 성공: {page_id}")
        
        # 원본 분석 텍스트 추가 (단순화된 방식으로)
        if "recommendations" in debug_info and "analysis_text" in debug_info["recommendations"]:
            analysis_text = debug_info["recommendations"]["analysis_text"]
            
            # 너무 긴 텍스트는 나누어 처리
            max_length = 1900  # 안전한 길이로 설정
            text_chunks = [analysis_text[i:i+max_length] for i in range(0, len(analysis_text), max_length)]
            
            # 분석 텍스트 헤더 추가
            analysis_header = {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "원본 분석 텍스트"}}]
                        }
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=analysis_header,
                    timeout=30.0
                )
                response.raise_for_status()
            
            # 각 청크별로 단락 추가
            for chunk in text_chunks:
                chunk_block = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": chunk}}]
                            }
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=chunk_block,
                        timeout=30.0
                    )
                    response.raise_for_status()
            
            logger.info(f"분석 텍스트 블록 추가 성공: {page_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Notion 페이지 컨텐츠 추가 실패: {str(e)}")
        return False
    

async def create_recommendation_record(agent_page_id: str, recommendations: Dict[str, Any], investment_period: int) -> bool:
    """
    투자 추천 결과만 Notion DB에 저장합니다 (백테스팅 없이).
    
    Args:
        agent_page_id: 투자 에이전트 페이지 ID
        recommendations: 추천 종목 정보
        investment_period: 투자 기간 (일)
        
    Returns:
        저장 성공 여부
    """
    from datetime import timedelta  # 필요한 임포트 추가
    
    try:
        # 추천 종목 및 비중
        stock_names = []
        if "recommended_stocks" in recommendations:
            for stock in recommendations["recommended_stocks"]:
                if "name" in stock and stock["name"]:
                    stock_names.append(stock["name"])
                    
        # 현재 날짜 및 예상 종료일
        current_date = datetime.now()
        end_date = current_date + timedelta(days=investment_period)
        
        # 투자 비중 텍스트
        weights = "균등 비중"  # 기본값
        
        # 추천 기록 생성
        recommendation_data = {
            "title": f"{current_date.strftime('%Y-%m-%d')} {len(stock_names)}종목 추천",
            "agent_page_id": agent_page_id,
            "start_date": current_date,
            "end_date": end_date,
            "stocks": stock_names,
            "weights": weights,
            "recommendation_type": "신규 추천"  # 새로운 필드 추가
        }
        
        # Notion DB에 저장 - 추천 DB 사용 (성과 DB와 별도로 관리 가능)
        result = await create_investment_recommendation(recommendation_data)
        
        if result and "id" in result:
            page_id = result["id"]
            
            # 추천 내용 상세 정보 추가
            await add_structured_recommendation_content(page_id, recommendations)
            
            return True
        else:
            return False
        
    except Exception as e:
        logger.error(f"추천 기록 저장 중 오류: {str(e)}")
        return False

async def create_investment_recommendation(recommendation_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    추천 기록 DB에 새 추천 기록을 생성합니다. (백테스팅 없이 추천만 기록)
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 추천 종목 - 멀티 셀렉트 객체 생성
    stocks_multi_select = []
    for stock in recommendation_data.get("stocks", []):
        stocks_multi_select.append({
            "name": stock
        })
    
    # 에이전트 ID 관계 설정
    agent_relation = []
    if "agent_page_id" in recommendation_data:
        agent_relation = [{"id": recommendation_data["agent_page_id"]}]
    
    # 날짜 형식 변환
    start_date = recommendation_data.get("start_date")
    end_date = recommendation_data.get("end_date")
    
    if isinstance(start_date, datetime):
        start_date = start_date.isoformat()
    
    if isinstance(end_date, datetime):
        end_date = end_date.isoformat()
    
    # 페이지 속성 설정 - 성과 DB와 동일한 구조 사용 (필요시 수정 가능)
    properties = {
        "투자 기록": {  # 타이틀 필드명은 DB에 맞게 조정 필요
            "title": [
                {
                    "text": {
                        "content": recommendation_data.get("title", f"추천 기록 {datetime.now().strftime('%Y-%m-%d')}")
                    }
                }
            ]
        },
        "에이전트": {
            "relation": agent_relation
        },
        "시작일": {
            "date": {
                "start": start_date
            }
        },
        "종료일": {
            "date": {
                "start": end_date
            }
        },
        "투자 종목": {
            "multi_select": stocks_multi_select
        },
        "투자 비중": {
            "rich_text": [
                {
                    "text": {
                        "content": recommendation_data.get("weights", "균등 비중")
                    }
                }
            ]
        },
        "결과 평가": {
            "select": {
                "name": recommendation_data.get("recommendation_type", "신규 추천")
            }
        }
    }
    
    # 요청 본문 생성 - 성과 DB ID 대신 추천 DB ID 사용
    # 필요시 NOTION_RECOMMENDATION_DB_ID 환경 변수 추가하고 교체 필요
    request_data = {
        "parent": {
            "database_id": NOTION_PERFORMANCE_DB_ID  # 임시로 성과 DB 사용, 필요시 전용 DB로 변경
        },
        "properties": properties
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=request_data,
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"추천 기록 생성 성공: {recommendation_data.get('title')}")
            return result
            
    except Exception as e:
        logger.error(f"추천 기록 생성 실패: {str(e)}")
        return None

async def add_structured_recommendation_content(page_id: str, recommendations: Dict[str, Any], title: str = "추천 종목 정보") -> bool:
    """
    노션 페이지에 구조화된 형식으로 추천 결과를 추가합니다.
    """
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    try:
        # 헤더 블록 추가
        basic_blocks = [
            # 헤딩 블록 추가
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{"type": "text", "text": {"content": title}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": f"추천일: {datetime.now().strftime('%Y-%m-%d')}"}}]
                }
            }
        ]
        
        # 먼저 기본 블록 추가
        basic_request = {"children": basic_blocks}
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                url, 
                headers=headers, 
                json=basic_request,
                timeout=30.0
            )
            response.raise_for_status()
            logger.info(f"기본 블록 추가 성공: {page_id}")
        
        # 추천 종목 정보 추가
        if "recommended_stocks" in recommendations and "portfolio_logic" in recommendations:
            recommended_stocks = recommendations["recommended_stocks"]
            portfolio_logic = recommendations.get("portfolio_logic", "")
            
            # 포트폴리오 논리 추가
            portfolio_blocks = {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "포트폴리오 구성 논리"}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": portfolio_logic}}]
                        }
                    },
                    {
                        "object": "block",
                        "type": "divider",
                        "divider": {}
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=portfolio_blocks,
                    timeout=30.0
                )
                response.raise_for_status()
            
            # 각 종목별 정보 추가
            for stock in recommended_stocks:
                # 티커 정보 추가
                ticker_display = f" ({stock.get('ticker', '')})" if "ticker" in stock else ""
                
                stock_info = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text", 
                                        "text": {"content": f"🔹 {stock.get('name', '')}{ticker_display}"}, 
                                        "annotations": {"bold": True}
                                    }
                                ]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"예상 수익률: {stock.get('expected_return', '미제공')}"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [{"type": "text", "text": {"content": f"위험도: {stock.get('risk_level', '미제공')}"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": f"추천 이유: {stock.get('reasoning', '미제공')}"}}]
                            }
                        },
                        {
                            "object": "block",
                            "type": "divider",
                            "divider": {}
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=stock_info,
                        timeout=30.0
                    )
                    response.raise_for_status()
            
            logger.info(f"추천 종목 블록 추가 성공: {page_id}")
        
        # 원본 분석 텍스트 추가
        if "analysis_text" in recommendations:
            analysis_text = recommendations["analysis_text"]
            
            # 너무 긴 텍스트는 나누어 처리
            max_length = 1900  # 안전한 길이로 설정
            text_chunks = [analysis_text[i:i+max_length] for i in range(0, len(analysis_text), max_length)]
            
            # 분석 텍스트 헤더 추가
            analysis_header = {
                "children": [
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [{"type": "text", "text": {"content": "원본 분석 텍스트"}}]
                        }
                    }
                ]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=analysis_header,
                    timeout=30.0
                )
                response.raise_for_status()
            
            # 각 청크별로 단락 추가
            for chunk in text_chunks:
                chunk_block = {
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": chunk}}]
                            }
                        }
                    ]
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.patch(
                        url, 
                        headers=headers, 
                        json=chunk_block,
                        timeout=30.0
                    )
                    response.raise_for_status()
            
            logger.info(f"분석 텍스트 블록 추가 성공: {page_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Notion 페이지 추천 내용 추가 실패: {str(e)}")
        return False

async def find_agent_by_name(agent_name: str) -> Optional[str]:
    """
    에이전트명으로 Notion DB에서 에이전트 페이지 ID를 찾습니다.
    """
    try:
        # 에이전트명으로 필터링
        filter_condition = {
            "property": "에이전트명",
            "title": {
                "equals": agent_name
            }
        }
        
        request_body = {
            "filter": filter_condition
        }
        
        # Notion DB 쿼리
        agent_pages = await query_notion_database(NOTION_AGENT_DB_ID, request_body)
        
        if not agent_pages:
            logger.info(f"에이전트 '{agent_name}'를 찾을 수 없습니다.")
            return None
        
        # 첫 번째 일치하는 에이전트의 페이지 ID 반환
        page_id = agent_pages[0].get("id")
        logger.info(f"에이전트 '{agent_name}'의 페이지 ID: {page_id}")
        return page_id
    
    except Exception as e:
        logger.error(f"에이전트 검색 실패: {str(e)}")
        return None

async def create_investment_performance(performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    투자 실적 DB에 새 투자 실적을 생성합니다. 디버깅 정보 추가 기능 포함.
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 투자 종목 - 멀티 셀렉트 객체 생성
    stocks_multi_select = []
    for stock in performance_data.get("stocks", []):
        stocks_multi_select.append({
            "name": stock
        })
    
    # 에이전트 ID 관계 설정
    agent_relation = []
    if "agent_page_id" in performance_data:
        agent_relation = [{"id": performance_data["agent_page_id"]}]
    
    # 날짜 형식 변환
    start_date = performance_data.get("start_date")
    end_date = performance_data.get("end_date")
    
    if isinstance(start_date, datetime):
        start_date = start_date.isoformat()
    
    if isinstance(end_date, datetime):
        end_date = end_date.isoformat()
    
    # 페이지 속성 설정
    properties = {
        "투자 기록": {
            "title": [
                {
                    "text": {
                        "content": performance_data.get("title", f"투자 기록 {datetime.now().strftime('%Y-%m-%d')}")
                    }
                }
            ]
        },
        "에이전트": {
            "relation": agent_relation
        },
        "시작일": {
            "date": {
                "start": start_date
            }
        },
        "종료일": {
            "date": {
                "start": end_date
            }
        },
        "투자 종목": {
            "multi_select": stocks_multi_select
        },
        "투자 비중": {
            "rich_text": [
                {
                    "text": {
                        "content": performance_data.get("weights", "균등 비중")
                    }
                }
            ]
        },
        "총 수익률": {
            "number": performance_data.get("total_return", 0)
        },
        "최대 낙폭": {
            "number": performance_data.get("max_drawdown", 0)
        },
        "결과 평가": {
            "select": {
                "name": performance_data.get("evaluation", "부분 성공")
            }
        }
    }
    
    # 요청 본문 생성
    request_data = {
        "parent": {
            "database_id": NOTION_PERFORMANCE_DB_ID
        },
        "properties": properties
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=request_data,
                timeout=30.0
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"투자 성과 기록 생성 성공: {performance_data.get('title')}")
            
            # 페이지가 성공적으로 생성되면 디버깅 정보 추가
            if result and "id" in result:
                page_id = result["id"]
                
                # 디버깅 정보 추출
                debug_info = performance_data.get("debug_info", {})
                if debug_info:
                    # 각 디버깅 정보 섹션을 별도로 추가
                    for title, content in debug_info.items():
                        content_str = (
                            json.dumps(content, ensure_ascii=False, indent=2) 
                            if isinstance(content, (dict, list)) 
                            else str(content)
                        )
                        await add_content_to_notion_page(page_id, content_str, title)
            
            return result
            
    except Exception as e:
        logger.error(f"투자 성과 기록 생성 실패: {str(e)}")
        return None