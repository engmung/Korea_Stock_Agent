import os
import json
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Gemini API 키
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

async def select_reports_by_agent_preference(
    agent, 
    candidate_reports_metadata: List[Dict[str, Any]], 
    backtest_date: str,
    debug_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    에이전트의 선호도에 맞게 보고서를 선택합니다.
    
    Args:
        agent: 투자 에이전트 객체
        candidate_reports_metadata: 선택 대상 보고서 메타데이터 목록
        backtest_date: 백테스팅 시작일
        debug_info: 디버깅 정보 저장용 딕셔너리
        
    Returns:
        선택된 보고서 ID 목록 및 선택 이유를 포함한 딕셔너리
    """
    try:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY가 설정되지 않았습니다. 기본 보고서 선택 방식을 사용합니다.")
            return {
                "selected_report_ids": [report["page_id"] for report in candidate_reports_metadata[:10]],
                "selection_info": {
                    "strategy": "기본 선택 (최신 10개)",
                    "details": [{"id": report["page_id"], "reason": "API 키 없음"} for report in candidate_reports_metadata[:10]]
                }
            }
        
        # 에이전트 프롬프트 가져오기
        agent_prompt = await get_agent_prompt(agent.page_id)
        if not agent_prompt:
            logger.warning(f"에이전트 '{agent.agent_name}'의 프롬프트를 찾을 수 없습니다. 기본 선택 방식을 사용합니다.")
            return {
                "selected_report_ids": [report["page_id"] for report in candidate_reports_metadata[:10]],
                "selection_info": {
                    "strategy": "기본 선택 (최신 10개)",
                    "details": [{"id": report["page_id"], "reason": "에이전트 프롬프트 없음"} for report in candidate_reports_metadata[:10]]
                }
            }
        
        # 보고서 메타데이터 개선 - 더 자세한 정보 포함
        formatted_reports = []
        for idx, report in enumerate(candidate_reports_metadata):
            # 일자 형식 변환 시도 (읽기 쉬운 형식으로)
            date_display = report["published_date"]
            try:
                if "T" in report["published_date"]:
                    # ISO 형식 날짜 파싱
                    dt = datetime.fromisoformat(report["published_date"].replace("Z", "+00:00"))
                    date_display = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
                
            formatted_reports.append({
                "id": report["page_id"],
                "index": idx + 1,
                "title": report["title"],
                "channel": report["channel"],
                "date": date_display,
                "raw_date": report["published_date"]  # 원본 날짜도 유지
            })
        
        # 선택 프롬프트 구성 - 더 명확한 지시 추가
        selection_prompt = f"""
        당신은 투자 데이터 분석가입니다. 다음 투자 에이전트의 전략과 선호도에 가장 적합한 보고서를 선택해주세요.

        ## 투자 에이전트 프롬프트
        {agent_prompt}

        ## 보고서 후보 목록 (총 {len(formatted_reports)}개)
        {json.dumps(formatted_reports, indent=2, ensure_ascii=False)}

        ## 요청
        위 투자 에이전트에게 가장 적합한 보고서 5-15개를 선택해주세요.
        백테스팅 시작일({backtest_date})을 고려해주세요.

        응답은 다음 JSON 형식으로 제공해주세요:
        {{
            "selected_reports": [
                {{
                    "id": "보고서ID", 
                    "title": "보고서 제목",
                    "channel": "채널명",
                    "date": "발행일자"
                }}
            ],
            "selection_strategy": "전반적인 선택 전략 설명"
        }}
        """
        
        # Gemini API 호출
        response_text = await call_gemini_api(selection_prompt)
        
        # 응답 파싱
        selection_result = parse_selection_response(response_text)
        
        # 디버깅 정보 저장
        if debug_info is not None:
            debug_info["report_selection"] = {
                "prompt": selection_prompt,
                "response": response_text,
                "parsed_result": selection_result
            }
        
        # 선택된 보고서 정보 추출 - 로깅과 디버깅을 위해 상세 정보 유지
        selected_reports_info = selection_result.get("selected_reports", [])
        
        # 선택된 보고서 ID 추출
        selected_report_ids = [item["id"] for item in selected_reports_info]
        
        # 상세 정보 구성
        selection_details = []
        for report in selected_reports_info:
            report_id = report.get("id", "")
            # 원본 메타데이터에서 추가 정보 찾기
            original_meta = next((r for r in candidate_reports_metadata if r["page_id"] == report_id), None)
            
            # 기본 정보
            detail = {
                "id": report_id,
                "title": report.get("title", original_meta["title"] if original_meta else "제목 없음"),
                "channel": report.get("channel", original_meta["channel"] if original_meta else "채널 없음"),
                "date": report.get("date", original_meta["published_date"] if original_meta else "날짜 없음")
            }
            selection_details.append(detail)
        
        # 기본값 처리: 선택된 보고서가 없으면 기본 10개 반환
        if not selected_report_ids:
            logger.warning(f"선택된 보고서가 없습니다. 기본 10개 보고서를 사용합니다.")
            selected_report_ids = [report["page_id"] for report in candidate_reports_metadata[:min(10, len(candidate_reports_metadata))]]
            selection_details = [{
                "id": report["page_id"], 
                "title": report["title"],
                "channel": report["channel"],
                "date": report["published_date"],
                "reason": "기본 선택 (JSON 파싱 실패)"
            } for report in candidate_reports_metadata[:min(10, len(candidate_reports_metadata))]]
            
            # 선택 전략도 업데이트
            if "selection_strategy" not in selection_result:
                selection_result["selection_strategy"] = "기본 선택 (최신 10개, JSON 파싱 실패)"
            
        # 선택된 보고서 로깅 - 더 상세한 정보 표시
        for detail in selection_details[:5]:  # 처음 5개만 로깅
            logger.info(f"선택된 보고서: '{detail['title']}' ({detail['channel']}, {detail['date']})")
        
        if len(selection_details) > 5:
            logger.info(f"그 외 {len(selection_details) - 5}개 보고서 선택됨")
        
        return {
            "selected_report_ids": selected_report_ids,
            "selection_info": {
                "strategy": selection_result.get("selection_strategy", ""),
                "details": selection_details
            }
        }
        
    except Exception as e:
        logger.error(f"보고서 선택 중 오류 발생: {str(e)}")
        # 오류 발생 시 기본 10개 반환
        default_count = min(10, len(candidate_reports_metadata))
        return {
            "selected_report_ids": [report["page_id"] for report in candidate_reports_metadata[:default_count]],
            "selection_info": {
                "strategy": "기본 선택 (최신 10개, 오류 발생)",
                "details": [{
                    "id": report["page_id"], 
                    "title": report["title"],
                    "channel": report["channel"],
                    "date": report["published_date"],
                    "reason": f"오류 발생: {str(e)}"
                } for report in candidate_reports_metadata[:default_count]]
            }
        }

async def get_agent_prompt(page_id: str) -> Optional[str]:
    """
    에이전트 페이지에서 시스템 프롬프트를 추출합니다.
    """
    from notion_utils import get_notion_page_content
    
    try:
        content = await get_notion_page_content(page_id)
        
        # '## 시스템 프롬프트' 섹션 이후의 내용 추출
        pattern = r'## 시스템 프롬프트\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        logger.warning(f"페이지 {page_id}에서 시스템 프롬프트를 찾을 수 없습니다.")
        return None
        
    except Exception as e:
        logger.error(f"에이전트 프롬프트 로드 중 오류: {str(e)}")
        return None

async def call_gemini_api(prompt: str) -> str:
    """
    Gemini API를 호출하여 텍스트 응답을 받습니다.
    """
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        model = "gemini-2.5-flash-preview-04-17"  # 최신 모델 사용
        
        # 요청 구성
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]
        
        # 시스템 지시사항
        system_instruction = """
        당신은 투자 데이터 선별 전문가입니다. 투자 에이전트의 특성과 전략에 가장 적합한 보고서를 선택하는 임무를 맡았습니다.
        요청받은 형식과 지침을 정확히 따라 JSON 형식으로 응답해주세요.
        응답은 반드시 JSON 파싱이 가능해야 합니다.
        특별히 에이전트가 선호하는 특정 전문가, 채널, 투자 스타일 등을 분석하여 보고서를 선택하세요.
        """
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.2,  # 낮은 온도로 일관성 확보
            response_mime_type="text/plain",
            system_instruction=[types.Part.from_text(text=system_instruction)]
        )
        
        # 비동기 처리를 위한 루프
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=generate_content_config
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini API 호출 중 오류: {str(e)}")
        return f"{{\"error\": \"{str(e)}\"}}"

def parse_selection_response(response_text: str) -> Dict[str, Any]:
    """
    Gemini API 응답에서 JSON 형식의 선택 결과를 추출합니다.
    """
    try:
        # JSON 블록 추출 시도
        json_match = re.search(r'({[\s\S]*})', response_text)
        
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # JSON 블록이 명확하지 않은 경우 전체 텍스트로 시도
        return json.loads(response_text)
        
    except Exception as e:
        logger.error(f"선택 응답 파싱 중 오류: {str(e)}")
        return {
            "selected_reports": [],
            "selection_strategy": f"파싱 오류: {str(e)}"
        }