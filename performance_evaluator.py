import os
import json
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# FinanceDataReader 임포트
import FinanceDataReader as fdr

# 환경 변수 로드
load_dotenv()

# 로깅 설정 - 간결하게 변경
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Gemini API 키
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 종목 검색기 임포트
from stock_searcher import StockSearcher

# NumPy 데이터 타입을 Python 기본 타입으로 변환하는 함수
def convert_numpy_types(obj):
    """NumPy 데이터 타입을 Python 기본 타입으로 변환합니다."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    return obj

async def backtest_recommendation(
    page_id: str, 
    recommendations: Dict[str, Any] = None,
    investment_period: int = 7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    investment_amount: float = 1000000
) -> Dict[str, Any]:
    """
    추천 종목에 대한 백테스팅을 수행합니다.
    """
    try:
        # 날짜 설정
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if not start_date:
            start_date = (datetime.now() - timedelta(days=investment_period)).strftime("%Y-%m-%d")
        
        logger.info(f"백테스팅 시작: 기간 {start_date} ~ {end_date}")
        
        # 에이전트 로드
        from investment_agent import InvestmentAgent
        agent = await InvestmentAgent.load_from_notion(page_id)
        
        if not agent:
            logger.error(f"에이전트를 찾을 수 없습니다: {page_id}")
            return {
                "status": "error", 
                "message": f"에이전트를 찾을 수 없습니다: {page_id}"
            }
        
        # 디버깅 정보 저장용 딕셔너리
        debug_info = {
            "agent_name": agent.agent_name,
            "start_date": start_date,
            "end_date": end_date,
            "investment_amount": investment_amount
        }
        
        # 종목 추천이 제공되지 않은 경우 새로 생성
        if not recommendations:
            # 보고서 분석 - 백테스팅 시작일 기준 이전 데이터만 사용
            from report_analyzer import find_relevant_reports, analyze_reports
            from stock_recommender import recommend_stocks
            
            # 백테스팅 시작일 이전의 관련 보고서 검색 - 디버깅 정보 전달
            reports = await find_relevant_reports(
                agent=agent,
                backtest_date=start_date,
                max_reports=40,
                debug_info=debug_info
            )
            
            if not reports:
                logger.info(f"백테스팅 날짜 {start_date} 이전에 관련 보고서가 없습니다.")
                return {
                    "status": "warning",
                    "message": f"백테스팅 날짜 {start_date} 이전에 관련 보고서가 없습니다."
                }
            
            # 보고서 분석
            analyzed_reports = await analyze_reports(reports)
            
            # Gemini를 사용한 종목 추천
            recommendations = await recommend_stocks(
                agent=agent,
                analyzed_reports=analyzed_reports,
                max_stocks=5,
                investment_period=investment_period
            )
        
        # 추천 종목 리스트 추출
        if not recommendations or "recommended_stocks" not in recommendations:
            logger.error("올바른 추천 데이터가 제공되지 않았습니다.")
            return {
                "status": "error",
                "message": "추천 종목 데이터가 없습니다."
            }
        
        stocks = recommendations["recommended_stocks"]
        
        if not stocks:
            logger.info("추천 종목이 없습니다.")
            return {
                "status": "warning",
                "message": "추천 종목이 없습니다."
            }
        
        # 종목명 리스트 추출
        stock_names = []
        for stock in stocks:
            if stock.get("name"):
                stock_names.append(stock.get("name"))
        
        if not stock_names:
            logger.error("유효한 종목명이 없습니다.")
            return {
                "status": "error",
                "message": "유효한 종목명이 없습니다."
            }
        
        logger.info(f"백테스팅할 종목명: {', '.join(stock_names)}")
        
        # 종목명을 종목코드로 변환 - Gemini의 Google Search 기능 활용
        from stock_searcher import StockSearcher
        stock_searcher = StockSearcher(api_key=GEMINI_API_KEY)
        
        # 디버깅 정보 업데이트
        debug_info["recommendations"] = recommendations
        debug_info["stock_names"] = stock_names
        
        # === 1단계: 각 종목명을 개별적으로 처리 ===
        all_ticker_symbols = []
        stock_search_results = []
        
        # 종목명-티커 매핑 정보를 저장할 딕셔너리
        stock_name_to_ticker = {}
        
        # 티커->종목명 매핑
        stock_ticker_mapping = recommendations.get("stock_tickers", {})
        
        for stock in stocks:
            stock_name = stock.get("name")
            
            # 이미 티커 코드가 포함된 경우
            if stock.get("ticker"):
                ticker = stock.get("ticker")
                all_ticker_symbols.append(ticker)
                stock_name_to_ticker[stock_name] = ticker
                stock_ticker_mapping[ticker] = stock_name
                continue
                
            # 주식 종목 검색 프롬프트 생성
            search_prompt = f"{stock_name} 주식 종목코드 찾아줘"
            extraction_result = await stock_searcher.extract_stock_codes(search_prompt)
            
            stock_search_results.append({
                "stock_name": stock_name,
                "prompt": search_prompt,
                "result": extraction_result
            })
            
            # 응답에서 종목 코드 추출 (6자리 숫자)
            response_text = extraction_result.get("response", "")
            ticker_match = re.search(r'(\d{6})', response_text)
            
            if ticker_match:
                ticker = ticker_match.group(1)
                logger.info(f"종목 '{stock_name}'에 대한 코드 '{ticker}' 찾음")
                all_ticker_symbols.append(ticker)
                stock_name_to_ticker[stock_name] = ticker
                stock_ticker_mapping[ticker] = stock_name
            else:
                logger.info(f"종목 '{stock_name}'에 대한 코드를 찾을 수 없음, KRX 검색 시도")
                
                # KRX 직접 검색 시도
                try:
                    krx_listing = await asyncio.to_thread(fdr.StockListing, 'KRX')
                    # 이름 기반 검색
                    matches = krx_listing[krx_listing['Name'].str.contains(stock_name, case=False)]
                    if not matches.empty:
                        ticker = matches.iloc[0]['Symbol']
                        exact_name = matches.iloc[0]['Name']
                        logger.info(f"KRX 검색으로 종목 '{stock_name}'에 대한 코드 '{ticker}' 찾음")
                        all_ticker_symbols.append(ticker)
                        stock_name_to_ticker[stock_name] = ticker
                        stock_ticker_mapping[ticker] = exact_name
                except Exception as e:
                    logger.info(f"KRX 검색 중 오류, 건너뜀: {str(e)}")
        
        # 디버깅 정보 추가
        debug_info["stock_search_results"] = stock_search_results
        debug_info["all_ticker_symbols"] = all_ticker_symbols
        debug_info["stock_ticker_mapping"] = stock_ticker_mapping
        
        # 검색 결과가 없으면 오류 반환
        if not all_ticker_symbols:
            logger.error("유효한 종목 코드를 하나도 찾을 수 없습니다.")
            return {
                "status": "error",
                "message": "유효한 종목 코드를 하나도 찾을 수 없습니다.",
                "debug_info": debug_info
            }
        
        # 중복 제거
        all_ticker_symbols = list(set(all_ticker_symbols))
        logger.info(f"최종 백테스팅 종목 코드: {', '.join(all_ticker_symbols)}")
        
        # 포트폴리오 백테스팅 실행
        backtest_result = await run_portfolio_backtest(
            all_ticker_symbols, 
            start_date, 
            end_date, 
            investment_amount,
            stock_names=stock_ticker_mapping  # 티커->종목명 매핑 전달
        )
        
        # 디버깅 정보에 백테스팅 결과 추가
        debug_info["backtest_result"] = backtest_result
        
        if backtest_result.get("status") != "success":
            logger.error(f"백테스팅 실패: {backtest_result.get('error')}")
            return {
                "status": "error",
                "message": f"백테스팅 중 오류가 발생했습니다: {backtest_result.get('error')}",
                "debug_info": debug_info
            }
        
        # 성과 지표 계산
        performance_metrics = calculate_performance_metrics(backtest_result)
        
        # 디버깅 정보에 성과 지표 추가
        debug_info["performance_metrics"] = performance_metrics
        
        # 투자 실적 기록 (Notion DB에 저장) - 디버깅 정보 포함
        await save_performance_record(
            agent_page_id=page_id,
            recommendations=recommendations,
            backtest_result=backtest_result,
            performance_metrics=performance_metrics,
            start_date=start_date,
            end_date=end_date,
            debug_info=debug_info
        )
        
        return {
            "status": "success",
            "agent_page_id": page_id,
            "agent_name": agent.agent_name,
            "start_date": start_date,
            "end_date": end_date,
            "metrics": performance_metrics,
            "backtest_details": backtest_result
        }
    
    except Exception as e:
        logger.error(f"성과 평가 중 오류: {str(e)}")
        return {
            "status": "error",
            "message": f"성과 평가 중 오류가 발생했습니다: {str(e)}"
        }

async def run_portfolio_backtest(
    ticker_symbols: List[str], 
    start_date: str, 
    end_date: str, 
    investment_amount: float = 1000000,
    stock_names: Dict[str, str] = None  # 티커 -> 종목명 매핑 추가
) -> Dict[str, Any]:
    """
    포트폴리오 백테스팅을 실행합니다.
    """
    try:
        results = []
        total_investment = investment_amount * len(ticker_symbols)
        portfolio_value = 0
        
        # 종목명 매핑이 없으면 빈 딕셔너리 생성
        if stock_names is None:
            stock_names = {}
        
        # 각 종목에 대해 백테스팅 수행
        for ticker in ticker_symbols:
            # 종목명 정보 전달
            result = await backtest_stock(
                ticker, 
                start_date, 
                end_date, 
                investment_amount,
                stock_name=stock_names.get(ticker)  # 매핑에서 종목명 가져오기
            )
            
            if result["status"] == "success":
                results.append(result)
                portfolio_value += result["final_value"]
            else:
                logger.info(f"종목 '{ticker}' 백테스팅 실패: {result.get('error')}, 건너뜀")
        
        if not results:
            logger.error("어떤 종목도 백테스팅할 수 없습니다.")
            return {
                "status": "error",
                "error": "어떤 종목도 백테스팅할 수 없습니다."
            }
        
        # 포트폴리오 성과 계산
        portfolio_profit = portfolio_value - total_investment
        portfolio_return = (portfolio_profit / total_investment) * 100
        
        # 종목별 성과 랭킹
        ranked_results = sorted(results, key=lambda x: x["profit_percentage"], reverse=True)
        
        # 포트폴리오 성과 요약
        performance_summary = []
        for result in ranked_results:
            performance_summary.append({
                "ticker": result["ticker"],
                "name": result["name"],
                "profit_percentage": result["profit_percentage"],
                "profit": result["profit"],
                "final_value": result["final_value"]
            })
        
        # 포트폴리오 결과 생성
        result = {
            "status": "success",
            "portfolio_size": len(results),
            "total_investment": total_investment,
            "portfolio_value": round(portfolio_value, 2),
            "portfolio_profit": round(portfolio_profit, 2),
            "portfolio_return": round(portfolio_return, 2),
            "start_date": start_date,
            "end_date": end_date,
            "best_performer": {
                "ticker": ranked_results[0]["ticker"],
                "name": ranked_results[0]["name"],
                "return": ranked_results[0]["profit_percentage"]
            } if ranked_results else None,
            "worst_performer": {
                "ticker": ranked_results[-1]["ticker"],
                "name": ranked_results[-1]["name"],
                "return": ranked_results[-1]["profit_percentage"]
            } if ranked_results else None,
            "performance_summary": performance_summary,
            "stock_results": results
        }
        
        # NumPy 타입 변환
        return convert_numpy_types(result)
    
    except Exception as e:
        logger.error(f"포트폴리오 백테스팅 중 오류: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


async def backtest_stock(
    ticker_symbol: str,
    start_date: str,
    end_date: str,
    investment_amount: float = 1000000,
    stock_name: str = None  # 종목명 파라미터 추가
) -> Dict[str, Any]:
    """
    단일 종목에 대한 백테스팅을 수행합니다.
    """
    try:
        logger.info(f"종목 '{ticker_symbol}' 백테스팅 시작 ({start_date} ~ {end_date})")
        
        # 티커가 문자열인지 확인
        if not isinstance(ticker_symbol, str):
            ticker_symbol = str(ticker_symbol)
        
        # 종목 데이터 가져오기
        try:
            data = await asyncio.to_thread(
                fdr.DataReader, 
                ticker_symbol, 
                start_date, 
                end_date
            )
            
            if data.empty:
                logger.info(f"종목 '{ticker_symbol}'의 데이터를 찾을 수 없습니다.")
                return {
                    "status": "error",
                    "error": f"종목 '{ticker_symbol}'의 데이터를 찾을 수 없습니다."
                }
        except Exception as e:
            logger.info(f"종목 '{ticker_symbol}' 데이터 조회 실패: {str(e)}")
            return {
                "status": "error",
                "error": f"종목 데이터 조회 실패: {str(e)}"
            }
        
        # 종목명 조회 또는 사용
        if not stock_name:
            try:
                krx_listing = fdr.StockListing('KRX')
                stock_info = krx_listing[krx_listing['Symbol'] == ticker_symbol]
                if not stock_info.empty:
                    stock_name = stock_info.iloc[0]['Name']
                else:
                    stock_name = f"종목 {ticker_symbol}"
            except:
                stock_name = f"종목 {ticker_symbol}"
        
        # 주식 매수 시뮬레이션
        initial_price = data.iloc[0]["Close"]
        final_price = data.iloc[-1]["Close"]
        
        shares_bought = investment_amount / initial_price
        final_value = shares_bought * final_price
        
        profit = final_value - investment_amount
        profit_percentage = (profit / investment_amount) * 100
        
        # 일간 수익률 계산
        data["Daily_Return"] = data["Close"].pct_change()
        
        # 누적 수익률 계산
        data["Cumulative_Return"] = (1 + data["Daily_Return"]).cumprod() - 1
        
        # 최대 낙폭(MDD) 계산
        data["Cumulative_Max"] = data["Close"].cummax()
        data["Drawdown"] = (data["Close"] - data["Cumulative_Max"]) / data["Cumulative_Max"]
        mdd = data["Drawdown"].min() * 100
        
        # 수익률 통계
        daily_returns = data["Daily_Return"].dropna()
        volatility = daily_returns.std() * (252 ** 0.5) * 100  # 연간 변동성
        sharpe_ratio = (profit_percentage / 365 * len(data)) / volatility if volatility != 0 else 0
        
        # 거래 시뮬레이션 (단순 매수-보유 전략)
        trade_history = [{
            "date": data.index[0].strftime("%Y-%m-%d"),
            "action": "매수",
            "price": round(initial_price, 2),
            "shares": round(shares_bought, 4),
            "value": round(investment_amount, 2),
        }, {
            "date": data.index[-1].strftime("%Y-%m-%d"),
            "action": "평가",
            "price": round(final_price, 2),
            "shares": round(shares_bought, 4),
            "value": round(final_value, 2),
        }]
        
        # 결과 생성
        result = {
            "status": "success",
            "ticker": ticker_symbol,
            "name": stock_name,  # 실제 종목명 저장
            "start_date": data.index[0].strftime("%Y-%m-%d"),
            "end_date": data.index[-1].strftime("%Y-%m-%d"),
            "initial_investment": investment_amount,
            "initial_price": round(initial_price, 2),
            "final_price": round(final_price, 2),
            "shares_bought": round(shares_bought, 4),
            "final_value": round(final_value, 2),
            "profit": round(profit, 2),
            "profit_percentage": round(profit_percentage, 2),
            "max_drawdown": round(mdd, 2),
            "volatility": round(volatility, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "trade_history": trade_history
        }
        
        return convert_numpy_types(result)
    
    except Exception as e:
        logger.error(f"종목 '{ticker_symbol}' 백테스팅 중 오류: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


def calculate_performance_metrics(backtest_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    백테스팅 결과에서 성과 지표를 계산합니다.
    
    Args:
        backtest_result: 백테스팅 결과 데이터
        
    Returns:
        성과 지표
    """
    try:
        # 성과 데이터 추출
        portfolio_return = backtest_result.get("portfolio_return", 0)
        portfolio_profit = backtest_result.get("portfolio_profit", 0)
        stock_results = backtest_result.get("stock_results", [])
        
        # 승률 계산 (수익이 발생한 종목 비율)
        profitable_stocks = sum(1 for stock in stock_results if stock.get("profit_percentage", 0) > 0)
        total_stocks = len(stock_results)
        win_rate = (profitable_stocks / total_stocks) * 100 if total_stocks > 0 else 0
        
        # 최대 낙폭 평균
        avg_max_drawdown = sum(stock.get("max_drawdown", 0) for stock in stock_results) / total_stocks if total_stocks > 0 else 0
        
        # 일관성 점수 (수익률의 표준편차 역수, 높을수록 일관적)
        returns = [stock.get("profit_percentage", 0) for stock in stock_results]
        returns_std = np.std(returns) if returns else 0
        consistency_score = 100 / (1 + returns_std) if returns_std > 0 else 100
        
        # 종목별 최고/최저 성과
        best_stock = max(stock_results, key=lambda x: x.get("profit_percentage", 0)) if stock_results else {}
        worst_stock = min(stock_results, key=lambda x: x.get("profit_percentage", 0)) if stock_results else {}
        
        # 종합 신뢰도 점수 계산
        # - 수익률 30% (높을수록 좋음)
        # - 승률 30% (높을수록 좋음)
        # - 일관성 20% (높을수록 좋음)
        # - 낙폭 20% (낮을수록 좋음, 역수로 계산)
        
        # 각 지표 정규화 (0~100 범위)
        return_score = min(max(portfolio_return * 5, 0), 100)  # 20% 수익이면 100점
        win_rate_score = win_rate
        consistency_score = min(consistency_score, 100)
        drawdown_score = min(max(100 - abs(avg_max_drawdown) * 10, 0), 100)  # 10% 낙폭이면 0점
        
        # 가중 합계
        trust_score = (
            return_score * 0.3 +
            win_rate_score * 0.3 +
            consistency_score * 0.2 +
            drawdown_score * 0.2
        )
        
        # 성과 평가 결과 (텍스트)
        evaluation = "성공"
        if portfolio_return > 5:
            evaluation = "큰 성공"
        elif portfolio_return > 0:
            evaluation = "부분 성공"
        elif portfolio_return > -5:
            evaluation = "소폭 손실"
        else:
            evaluation = "손실"
        
        return {
            "portfolio_return": portfolio_return,
            "portfolio_profit": portfolio_profit,
            "win_rate": win_rate,
            "profitable_stocks": profitable_stocks,
            "total_stocks": total_stocks,
            "avg_max_drawdown": avg_max_drawdown,
            "consistency_score": consistency_score,
            "trust_score": trust_score,
            "best_stock": {
                "ticker": best_stock.get("ticker", ""),
                "name": best_stock.get("name", ""),
                "return": best_stock.get("profit_percentage", 0)
            } if best_stock else None,
            "worst_stock": {
                "ticker": worst_stock.get("ticker", ""),
                "name": worst_stock.get("name", ""),
                "return": worst_stock.get("profit_percentage", 0)
            } if worst_stock else None,
            "evaluation": evaluation
        }
    
    except Exception as e:
        logger.error(f"성과 지표 계산 중 오류: {str(e)}")
        return {
            "portfolio_return": 0,
            "win_rate": 0,
            "trust_score": 0,
            "evaluation": "평가 오류"
        }


async def save_performance_record(
    agent_page_id: str,
    recommendations: Dict[str, Any],
    backtest_result: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    start_date: str,
    end_date: str,
    debug_info: Dict[str, Any] = None
) -> bool:
    """
    투자 성과 결과를 Notion DB에 저장합니다.
    
    Args:
        agent_page_id: 투자 에이전트 페이지 ID
        recommendations: 추천 종목 정보
        backtest_result: 백테스팅 결과
        performance_metrics: 성과 지표
        start_date: 백테스팅 시작일
        end_date: 백테스팅 종료일
        debug_info: 디버깅 정보 (선택사항)
        
    Returns:
        저장 성공 여부
    """
    from notion_utils import create_investment_performance, add_structured_content_to_notion_page
    
    try:
        # 추천 종목 및 비중
        stock_names = []
        if "recommended_stocks" in recommendations:
            for stock in recommendations["recommended_stocks"]:
                if "name" in stock and stock["name"]:
                    stock_names.append(stock["name"])
        
        # 투자 비중 텍스트
        weights = "균등 비중"  # 기본값
        
        # 총 수익률 (백테스팅 결과에서 가져오기)
        portfolio_return = performance_metrics.get("portfolio_return", 0)
        
        # 페이지 타이틀 형식 변경 - 총수익률(종목수)
        page_title = f"{portfolio_return:.1f}%({len(stock_names)}종목)"
        
        # 성과 기록 생성
        performance_data = {
            "title": page_title,
            "agent_page_id": agent_page_id,
            "start_date": datetime.fromisoformat(start_date.replace('Z', '+00:00')) if isinstance(start_date, str) else start_date,
            "end_date": datetime.fromisoformat(end_date.replace('Z', '+00:00')) if isinstance(end_date, str) else end_date,
            "stocks": stock_names,
            "weights": weights,
            "total_return": portfolio_return,
            "max_drawdown": performance_metrics.get("avg_max_drawdown", 0),
            "evaluation": performance_metrics.get("evaluation", "부분 성공"),
            "debug_info": debug_info  # 디버깅 정보 전달
        }
        
        # Notion DB에 저장
        result = await create_investment_performance(performance_data)
        
        if result and "id" in result:
            page_id = result["id"]
            
            # 디버깅 정보가 있으면 구조화된 형식으로 추가
            if debug_info:
                await add_structured_content_to_notion_page(page_id, debug_info, "백테스팅 상세 결과")
            
            return True
        else:
            return False
        
    except Exception as e:
        logger.error(f"성과 기록 저장 중 오류: {str(e)}")
        return False