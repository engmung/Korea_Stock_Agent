import os
import asyncio
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# 요청 제한 관리를 위한 세마포어
# 1분당 최대 2개의 요청 허용
API_SEMAPHORE = asyncio.Semaphore(2)
API_RATE_LIMIT_SECONDS = 60  # 1분 딜레이

async def analyze_script_with_gemini(script: str, video_title: str, channel_name: str) -> str:
    """
    Gemini API를 사용하여 스크립트를 분석하고 마크다운 보고서를 생성합니다.
    
    Args:
        script: 분석할 유튜브 스크립트
        video_title: 영상 제목
        channel_name: 채널명
    
    Returns:
        마크다운 형식의 분석 보고서 또는 오류 시 오류 메시지
    """
    try:
        # API 키 설정
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
            return "# AI 분석 보고서\n\n## 분석 오류\n\nGEMINI_API_KEY 환경 변수가 설정되지 않았습니다."
        
        # 프롬프트 작성 - 주식 종목 분석에 특화된 프롬프트
        prompt = f"""# 주식 종목 분석 요청

제목: {video_title}
채널: {channel_name}

내가 주는 스크립트의 내용을 필요한 내용만 정리한 보고서로 만들어줘. 내가 주식종목채널의 콘텐츠를 주면, 너는 거기서 언급되는 종목들과 그 내용을 상세하게 기록해줘. 언급된 이유, 추천정도, 투자관점 등등 자세하게. 다른 내용은 필요 없고, 오로지 종목과 그에 관한 정보만 줘. 시장 전반에 대한 언급같은 건 필요 없어."""

        # 비동기적으로 Gemini API 호출 (API 제한 고려)
        async with API_SEMAPHORE:
            logger.info(f"Gemini API 호출 시작: {video_title}")
            
            # API 호출 전 타임스탬프 기록
            start_time = asyncio.get_event_loop().time()
            
            # 프로세스 시작
            def call_gemini():
                try:
                    client = genai.Client(api_key=api_key)
                    model = "gemini-2.5-flash-preview-04-17"  # 최신 모델 사용
                    
                    # Content 객체 생성
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=prompt)],
                        ),
                    ]
                    
                    # 시스템 지시사항 설정 - 종목 분석에 특화
                    system_instruction = """당신은 투자 전문가로서 주식 종목 분석을 담당합니다. 주어진 스크립트에서 언급된 주식 종목과 관련 정보만을 추출하여 정리해주세요. 

다음 지침을 반드시 따르세요:

1. 종목 중심으로 정보를 정리하세요. 시장 전반에 대한 일반적인 내용은 제외합니다.
2. 각 종목에 대해 다음 정보를 포함하세요:
   - 언급 이유
   - 추천 정도 (적극 매수/매수/중립/매도 등)
   - 투자 관점 (단기/중기/장기)
   - 주요 내용

3. 정보가 없는 경우 "언급되지 않음"으로 표시하세요.
4. 마크다운 형식으로 작성하고, 각 종목은 제목(##)으로 구분하세요.
5. 종목명은 정확하게 작성하세요. 오타가 있을 경우 올바른 종목명을 사용하세요."""
                    
                    generate_content_config = types.GenerateContentConfig(
                        response_mime_type="text/plain",
                        system_instruction=[types.Part.from_text(text=system_instruction)],
                    )
                    
                    # 스트리밍 응답 수집
                    response_text = ""
                    
                    # 스크립트를 분석에 사용
                    full_prompt = f"{prompt}\n\n스크립트 내용:\n{script}"
                    contents[0].parts[0].text = full_prompt
                    
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=contents,
                        config=generate_content_config,
                    ):
                        if chunk.text:
                            response_text += chunk.text
                    
                    return response_text
                except Exception as e:
                    logger.error(f"Gemini 함수 내 오류: {str(e)}")
                    return f"# AI 분석 보고서\n\n## Gemini API 오류\n\n{str(e)}"
            
            # 비동기적으로 API 호출 실행
            try:
                response_text = await asyncio.to_thread(call_gemini)
            except Exception as e:
                logger.error(f"asyncio.to_thread 오류: {str(e)}")
                return f"# AI 분석 보고서\n\n## 분석 오류\n\nasyncio.to_thread 실행 중 오류가 발생했습니다: {str(e)}"
            
            # API 호출 후 경과 시간 계산
            elapsed_time = asyncio.get_event_loop().time() - start_time
            # 1분에서 경과 시간을 뺀 만큼 대기 (최소 0초)
            wait_time = max(0, API_RATE_LIMIT_SECONDS - elapsed_time)
            
            if wait_time > 0:
                logger.info(f"API 제한 준수를 위해 {wait_time:.1f}초 대기")
                await asyncio.sleep(wait_time)
            
            if response_text:
                logger.info("Gemini 분석 완료")
                
                # 응답이 마크다운 형식인지 확인하고 수정
                if not response_text.startswith("# "):
                    response_text = "# 주식 종목 분석 보고서\n\n" + response_text
                
                # 마크다운 형식 일관성 개선
                response_text = clean_markdown_format(response_text)
                
                return response_text
            else:
                logger.error("Gemini가 빈 응답을 반환했습니다.")
                return "# 주식 종목 분석 보고서\n\n## 분석 오류\n\nGemini API가 응답을 생성하지 못했습니다."
            
    except Exception as e:
        logger.error(f"Gemini API 호출 중 오류 발생: {str(e)}")
        return f"# 주식 종목 분석 보고서\n\n## 분석 오류\n\nGemini API 호출 중 오류가 발생했습니다: {str(e)}"


def clean_markdown_format(text: str) -> str:
    """마크다운 형식을 정리하고 일관성을 높입니다."""
    lines = text.split('\n')
    result_lines = []
    
    # 불릿 포인트 형식 일관화 (* -> -)
    for i, line in enumerate(lines):
        # 불릿 포인트 일관화
        if line.strip().startswith('* '):
            line = line.replace('* ', '- ', 1)
        
        # 줄바꿈 개선: 제목 앞에는 빈 줄 추가
        if line.startswith('#') and i > 0 and lines[i-1].strip():
            result_lines.append('')
        
        # 현재 줄 추가
        result_lines.append(line)
        
        # 줄바꿈 개선: 제목 뒤에는 빈 줄 추가
        if line.startswith('#') and i < len(lines) - 1 and lines[i+1].strip() and not lines[i+1].startswith('#'):
            result_lines.append('')
    
    return '\n'.join(result_lines)