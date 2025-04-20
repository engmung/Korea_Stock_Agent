import os
import json
import logging
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from notion_markdown import create_markdown_blocks

# 환경 변수 로드
load_dotenv()

# Notion API 설정
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
REFERENCE_DB_ID = os.getenv("REFERENCE_DB_ID")
SCRIPT_DB_ID = os.getenv("SCRIPT_DB_ID")

logger = logging.getLogger(__name__)

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
                logger.info(f"Querying Notion database (attempt {attempt+1}/{max_retries})")
                response = await client.post(
                    url, 
                    headers=headers, 
                    json=request_body, 
                    timeout=timeout
                )
                
                response.raise_for_status()
                results = response.json().get("results", [])
                logger.info(f"Successfully retrieved {len(results)} records from Notion database")
                return results
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout when querying Notion database (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                # 재시도 전 잠시 대기 (지수 백오프)
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(f"Max retries reached when querying Notion database {database_id}")
                return []
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            # 429 (Rate Limit) 오류인 경우 더 오래 대기
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                logger.warning(f"Rate limited. Waiting for {retry_after}s before retry")
                await asyncio.sleep(retry_after)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error querying Notion database: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return []
    
    return []


async def create_script_report_page(database_id: str, properties: Dict[str, Any], content: str, max_retries: int = 3, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
    """
    Notion에 새 페이지를 생성합니다. 마크다운 형식의 콘텐츠를 적절한 Notion 블록으로 변환합니다.
    재시도 및 타임아웃 처리가 포함되어 있습니다.
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 페이지 내용 설정 - 개선된 마크다운 처리 사용
    data = {
        "parent": {"database_id": database_id},
        "properties": properties,
        "children": create_markdown_blocks(content)
    }
    
    # Notion API는 한 번에 100개의 블록까지만 허용
    # 블록이 100개 이상이면 나눠서 요청
    MAX_BLOCKS_PER_REQUEST = 90  # 안전하게 90개로 제한
    
    if len(data["children"]) > MAX_BLOCKS_PER_REQUEST:
        logger.info(f"블록이 너무 많아 여러 요청으로 나누어 처리합니다. 총 {len(data['children'])}개 블록")
        
        # 첫 번째 요청: 속성과 첫 90개 블록
        first_request_data = {
            "parent": data["parent"],
            "properties": data["properties"],
            "children": data["children"][:MAX_BLOCKS_PER_REQUEST]
        }
        
        # 첫 번째 페이지 생성
        page_response = None
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Creating Notion page - first part (attempt {attempt+1}/{max_retries})")
                    response = await client.post(
                        url, 
                        headers=headers, 
                        json=first_request_data,
                        timeout=timeout
                    )
                    
                    response.raise_for_status()
                    page_response = response.json()
                    logger.info(f"First part created successfully")
                    break
                    
            except Exception as e:
                logger.error(f"Error creating first part: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        if not page_response:
            return None
            
        # 남은 블록을 90개씩 나눠 추가 요청
        page_id = page_response["id"]
        remaining_blocks = data["children"][MAX_BLOCKS_PER_REQUEST:]
        
        for i in range(0, len(remaining_blocks), MAX_BLOCKS_PER_REQUEST):
            append_blocks = remaining_blocks[i:i + MAX_BLOCKS_PER_REQUEST]
            
            # 블록 추가 요청
            append_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
            append_data = {"children": append_blocks}
            
            success = False
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient() as client:
                        logger.info(f"Appending blocks part {i//MAX_BLOCKS_PER_REQUEST + 2} (attempt {attempt+1}/{max_retries})")
                        response = await client.patch(
                            append_url, 
                            headers=headers, 
                            json=append_data,
                            timeout=timeout
                        )
                        
                        response.raise_for_status()
                        logger.info(f"Part {i//MAX_BLOCKS_PER_REQUEST + 2} appended successfully")
                        success = True
                        # API 제한 준수를 위한 딜레이
                        await asyncio.sleep(0.5)  # 0.5초 대기
                        break
                        
                except Exception as e:
                    logger.error(f"Error appending part {i//MAX_BLOCKS_PER_REQUEST + 2}: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        # 실패해도 계속 진행, 일부 콘텐츠라도 저장
                        logger.warning(f"Failed to append part {i//MAX_BLOCKS_PER_REQUEST + 2}, but continuing")
            
            if not success:
                logger.warning(f"Could not append all blocks to page")
                
        return page_response
    
    # 블록이 적은 경우 단일 요청으로 처리
    else:
        # 재시도 로직
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Creating Notion page (attempt {attempt+1}/{max_retries})")
                    response = await client.post(
                        url, 
                        headers=headers, 
                        json=data,
                        timeout=timeout
                    )
                    
                    # 디버깅을 위한 상세 오류 로깅
                    if response.status_code != 200:
                        logger.error(f"Notion API 오류: {response.status_code}")
                        logger.error(f"응답 내용: {response.text}")
                        
                        # 오류 내용 상세 분석
                        try:
                            error_json = response.json()
                            if "message" in error_json:
                                logger.error(f"API 오류 메시지: {error_json['message']}")
                            if "code" in error_json:
                                logger.error(f"API 오류 코드: {error_json['code']}")
                        except:
                            pass
                    
                    response.raise_for_status()
                    logger.info(f"Successfully created Notion page")
                    return response.json()
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout when creating Notion page (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached when creating Notion page")
                    return None
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                # 429 (Rate Limit) 오류인 경우 더 오래 대기
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    logger.warning(f"Rate limited. Waiting for {retry_after}s before retry")
                    await asyncio.sleep(retry_after)
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Error creating Notion page: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error("Max retries reached")
                    return None
        
        return None
    
# 원래 함수명을 유지하면서 기능을 개선합니다
async def check_recent_scripts_for_title(program_name: str, video_url: str = None, since_date: str = None) -> bool:
    """
    최근 특정 날짜 이후의 스크립트 중에서 동일한 프로그램의 동일한 영상이 이미 처리되었는지 확인합니다.
    
    Args:
        program_name: 프로그램 키워드(예: '김정수', '모닝 스탠바이')
        video_url: 확인할 영상 URL (None이면 프로그램명만 체크)
        since_date: ISO 형식의 날짜 문자열(예: 2024-04-15T00:00:00Z)
        
    Returns:
        동일한 영상이 이미 처리되었으면 True, 아니면 False
    """
    # since_date가 없으면 모든 기간 검색
    filter_conditions = []
    
    if since_date:
        filter_conditions.append({
            "property": "영상 날짜",
            "date": {
                "on_or_after": since_date
            }
        })
    
    # 프로그램명으로 필터링
    filter_conditions.append({
        "property": "제목",
        "title": {
            "equals": program_name
        }
    })
    
    # 필터 구성
    request_body = {
        "filter": {
            "and": filter_conditions
        }
    }
    
    # 스크립트 조회
    scripts = await query_notion_database(SCRIPT_DB_ID, request_body)
    
    # URL이 제공되지 않았으면 프로그램명만으로 존재 여부 확인
    if not video_url:
        return len(scripts) > 0
    
    # URL까지 비교
    for script in scripts:
        properties = script.get("properties", {})
        
        # URL 속성 확인
        if "URL" in properties and "url" in properties["URL"]:
            script_url = properties["URL"]["url"]
            
            # 동일한 URL이면 이미 처리된 영상
            if script_url == video_url:
                return True
    
    return False

async def update_notion_page(page_id: str, properties: Dict[str, Any], max_retries: int = 3, timeout: float = 30.0) -> bool:
    """
    Notion 페이지의 속성을 업데이트합니다.
    재시도 및 타임아웃 처리가 포함되어 있습니다.
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    data = {
        "properties": properties
    }
    
    # 재시도 로직
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                logger.info(f"Updating Notion page (attempt {attempt+1}/{max_retries})")
                response = await client.patch(
                    url, 
                    headers=headers, 
                    json=data,
                    timeout=timeout
                )
                
                response.raise_for_status()
                logger.info(f"Successfully updated Notion page")
                return True
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout when updating Notion page (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error("Max retries reached when updating Notion page")
                return False
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429 and attempt < max_retries - 1:
                retry_after = int(e.response.headers.get("Retry-After", 5))
                await asyncio.sleep(retry_after)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error updating Notion page: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return False
    
    return False

async def check_script_exists(video_url: str) -> bool:
    """스크립트 DB에 이미 해당 영상의 스크립트가 있는지 확인합니다."""
    script_pages = await query_notion_database(SCRIPT_DB_ID)
    
    for page in script_pages:
        properties = page.get("properties", {})
        url_property = properties.get("URL", {})
        
        if "url" in url_property and url_property["url"] == video_url:
            return True
    
    return False

async def reset_all_channels() -> bool:
    """참고용 DB의 모든 채널을 활성화 상태로 변경합니다."""
    reference_pages = await query_notion_database(REFERENCE_DB_ID)
    logger.info(f"Resetting {len(reference_pages)} channels to active state")
    
    success_count = 0
    
    for page in reference_pages:
        page_id = page.get("id")
        properties = {
            "활성화": {
                "checkbox": True
            }
        }
        
        success = await update_notion_page(page_id, properties)
        if success:
            success_count += 1
    
    logger.info(f"Successfully reset {success_count}/{len(reference_pages)} channels")
    return success_count > 0

async def create_investment_agent(agent_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """투자 에이전트 DB에 새 에이전트를 생성합니다."""
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    properties = {
        "에이전트 ID": {
            "title": [
                {
                    "text": {
                        "content": agent_data["agent_id"]
                    }
                }
            ]
        },
        "투자 철학": {
            "rich_text": [
                {
                    "text": {
                        "content": agent_data["investment_philosophy"]
                    }
                }
            ]
        },
        "생성일": {
            "date": {
                "start": datetime.now().isoformat()
            }
        },
        "현재 상태": {
            "select": {
                "name": agent_data.get("status", "활성")
            }
        },
        "평균 수익률": {
            "number": agent_data.get("avg_return", 0)
        },
        "성공률": {
            "number": agent_data.get("success_rate", 0)
        }
    }
    
    data = {
        "parent": {"database_id": INVESTMENT_AGENT_DB_ID},
        "properties": properties
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                headers=headers, 
                json=data,
                timeout=30.0
            )
            
            response.raise_for_status()
            logger.info(f"Successfully created investment agent: {agent_data['agent_id']}")
            return response.json()
            
    except Exception as e:
        logger.error(f"Error creating investment agent: {str(e)}")
        return None

async def create_investment_performance(performance_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """투자 실적 DB에 새 투자 실적을 생성합니다."""
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    # 에이전트 ID 관계 설정
    agent_relation = []
    if "agent_id_relation" in performance_data and performance_data["agent_id_relation"]:
        agent_relation = [{"id": performance_data["agent_id_relation"]}]
    
    properties = {
        "투자 기록": {
            "title": [
                {
                    "text": {
                        "content": performance_data["title"]
                    }
                }
            ]
        },
        "에이전트 ID": {
            "relation": agent_relation
        },
        "시작일": {
            "date": {
                "start": performance_data["start_date"].isoformat()
            }
        },
        "종료일": {
            "date": {
                "start": performance_data["end_date"].isoformat()
            }
        },
        "투자 종목": {
            "multi_select": [{"name": stock} for stock in performance_data.get("stocks", [])]
        },
        "투자 비중": {
            "rich_text": [
                {
                    "text": {
                        "content": performance_data.get("weights", "")
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
    
    data = {
        "parent": {"database_id": INVESTMENT_PERFORMANCE_DB_ID},
        "properties": properties
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                headers=headers, 
                json=data,
                timeout=30.0
            )
            
            response.raise_for_status()
            logger.info(f"Successfully created investment performance: {performance_data['title']}")
            return response.json()
            
    except Exception as e:
        logger.error(f"Error creating investment performance: {str(e)}")
        return None

async def increment_citation_count(script_page_id: str) -> bool:
    """스크립트의 인용 횟수를 1 증가시킵니다."""
    # 현재 페이지 정보 가져오기
    url = f"https://api.notion.com/v1/pages/{script_page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, 
                headers=headers
            )
            
            response.raise_for_status()
            page_data = response.json()
            
            # 현재 인용 횟수 가져오기
            current_count = 0
            if "properties" in page_data:
                if "인용 횟수" in page_data["properties"]:
                    current_count = page_data["properties"]["인용 횟수"].get("number", 0) or 0
            
            # 인용 횟수 증가
            new_count = current_count + 1
            
            # 페이지 업데이트
            properties = {
                "인용 횟수": {
                    "number": new_count
                }
            }
            
            return await update_notion_page(script_page_id, properties)
            
    except Exception as e:
        logger.error(f"Error incrementing citation count: {str(e)}")
        return False