import os
import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from google import genai
from google.genai import types
import httpx

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class NotionAPIManager:
    """노션 API 요청을 관리하는 클래스. 토큰 버킷 알고리즘을 사용해 요청 제한 준수."""
    
    def __init__(self, api_key: str, requests_per_second: int = 3):
        self.api_key = api_key
        self.requests_per_second = requests_per_second
        self.token_bucket = requests_per_second  # 초기 토큰 수
        self.last_refill_time = time.time()
        self.request_queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self._worker_task = None
        
    async def start(self):
        """요청 처리 워커를 시작합니다."""
        self._worker_task = asyncio.create_task(self._process_queue())
        logger.info("노션 API 관리자 시작됨")
        
    async def stop(self):
        """요청 처리 워커를 중지합니다."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("노션 API 관리자 중지됨")
    
    async def execute_request(self, request_func: Callable, *args, **kwargs):
        """
        노션 API 요청을 큐에 추가하고 결과를 기다립니다.
        
        Args:
            request_func: 실행할 HTTP 요청 함수
            *args, **kwargs: request_func에 전달할 인자들
            
        Returns:
            요청 결과
        """
        future = asyncio.Future()
        await self.request_queue.put((future, request_func, args, kwargs))
        return await future
    
    async def _refill_tokens(self):
        """토큰 버킷을 리필합니다."""
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_refill_time
            tokens_to_add = time_passed * self.requests_per_second
            self.token_bucket = min(self.token_bucket + tokens_to_add, self.requests_per_second)
            self.last_refill_time = current_time
    
    async def _process_queue(self):
        """요청 큐를 처리하는 워커 태스크."""
        try:
            while True:
                future, request_func, args, kwargs = await self.request_queue.get()
                try:
                    # 토큰 버킷 리필
                    await self._refill_tokens()
                    
                    # 토큰 사용 가능 여부 확인
                    async with self.lock:
                        if self.token_bucket < 1:
                            # 토큰이 부족하면 대기
                            sleep_time = (1 - self.token_bucket) / self.requests_per_second
                            # DEBUG 레벨로 변경
                            logger.debug(f"API 제한에 도달했습니다. {sleep_time:.2f}초 대기")
                            await asyncio.sleep(sleep_time)
                            # 대기 후 토큰 다시 계산
                            await self._refill_tokens()
                        
                        # 토큰 사용
                        self.token_bucket -= 1
                    
                    # 요청 실행
                    result = await request_func(*args, **kwargs)
                    future.set_result(result)
                    
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.request_queue.task_done()
        except asyncio.CancelledError:
            # 워커가 취소되면 종료
            return

    async def query_notion_database(self, database_id: str, request_body: dict = None, max_retries: int = 3, timeout: float = 30.0):
        """Notion 데이터베이스를 쿼리합니다."""
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        if request_body is None:
            request_body = {}
        
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, 
                    headers=headers, 
                    json=request_body, 
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json().get("results", [])
        
        # 재시도 로직을 포함하여 요청 실행
        for attempt in range(max_retries):
            try:
                return await self.execute_request(make_request)
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return []
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                else:
                    return []
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return []
        
        return []

    async def get_notion_page(self, page_id: str, max_retries: int = 3, timeout: float = 30.0):
        """Notion 페이지를 조회합니다."""
        url = f"https://api.notion.com/v1/pages/{page_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28"
        }
        
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, 
                    headers=headers, 
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()
        
        for attempt in range(max_retries):
            try:
                return await self.execute_request(make_request)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
        
        return None

    async def update_notion_page(self, page_id: str, properties: Dict[str, Any], max_retries: int = 3, timeout: float = 30.0):
        """Notion 페이지를 업데이트합니다."""
        url = f"https://api.notion.com/v1/pages/{page_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        request_data = {
            "properties": properties
        }
        
        async def make_request():
            async with httpx.AsyncClient() as client:
                response = await client.patch(
                    url,
                    headers=headers,
                    json=request_data,
                    timeout=timeout
                )
                response.raise_for_status()
                return True
                
        for attempt in range(max_retries):
            try:
                return await self.execute_request(make_request)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return False
        
        return False


class GeminiAPIManager:
    """Gemini API 키를 관리하고 워커에 할당하는 클래스."""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.key_usage = {key: 0 for key in self.api_keys}
        self.lock = asyncio.Lock()
        logger.info(f"Gemini API 관리자 초기화 완료 (총 {len(self.api_keys)}개 키)")
    
    def _load_api_keys(self) -> List[str]:
        """환경 변수에서 Gemini API 키를 로드합니다."""
        keys = []
        
        # 기본 키 확인
        main_key = os.environ.get("GEMINI_API_KEY")
        if main_key:
            keys.append(main_key)
        
        # 추가 키 확인 (GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...)
        i = 1
        while True:
            key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        if not keys:
            logger.warning("Gemini API 키가 설정되지 않았습니다.")
        
        return keys
    
    async def get_api_key(self, worker_id: str) -> Optional[str]:
        """
        워커에 Gemini API 키를 할당합니다.
        
        Args:
            worker_id: 워커 ID
            
        Returns:
            Gemini API 키 또는 키가 없으면 None
        """
        if not self.api_keys:
            return None
        
        async with self.lock:
            # 가장 적게 사용된 키 선택
            selected_key = min(self.key_usage, key=self.key_usage.get)
            # 사용 횟수 증가
            self.key_usage[selected_key] += 1
            
            logger.debug(f"워커 {worker_id}에 Gemini API 키 할당 (사용 횟수: {self.key_usage[selected_key]})")
            return selected_key

    async def get_client(self, worker_id: str) -> Optional[genai.Client]:
        """
        워커에 Gemini API 클라이언트를 할당합니다.
        
        Args:
            worker_id: 워커 ID
            
        Returns:
            Gemini API 클라이언트 또는 키가 없으면 None
        """
        api_key = await self.get_api_key(worker_id)
        if not api_key:
            return None
        
        return genai.Client(api_key=api_key)

    async def execute_gemini_request(self, worker_id: str, request_func: Callable, *args, **kwargs):
        """
        Gemini API 요청을 실행합니다.
        
        Args:
            worker_id: 워커 ID
            request_func: 실행할 Gemini API 요청 함수
            *args, **kwargs: request_func에 전달할 인자들
            
        Returns:
            요청 결과
        """
        client = await self.get_client(worker_id)
        if not client:
            raise ValueError("Gemini API 클라이언트를 생성할 수 없습니다. API 키 확인 필요.")
        
        try:
            # kwargs에 클라이언트 전달
            kwargs['client'] = client
            return await request_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Gemini API 요청 실패: {str(e)}")
            raise