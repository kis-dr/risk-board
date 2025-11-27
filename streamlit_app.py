import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import altair as alt
import warnings
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="우체국보험 리스크 스코어보드",
    page_icon="📊",
    layout="wide"
)

# --- [설정] 툴팁 데이터 (지표 설명) ---
tooltip_data = {
    "Citi 매크로 리스크": {"desc": "금융시장의 위험을 반영하는 글로벌 신용위험지표. 주식 시장의 기대 변동성(VIX)과 환율 변동성, 금리 변동성스프레드, 회사채 신용부도스왑, EMBI + 스프레드를 종합하여 산출한 리스크 지표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "KOSPI200 기대변동성(VKOSPI)": {"desc": "VKOSPI 지수가 높으면 투자심리가 불안하여 앞으로 주식시장의 변동성이 높아질 것이라는 예상이 많다는 뜻. 높은 시장 변동성이 예상될 때에는 투자자들이 주식매수를 기피하여, 하락장으로 이어질 개연성이 증가.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "S&P500 기대변동성(VIX)": {"desc": "VIX 지수가 높으면 투자심리가 불안하여 앞으로 주식시장의 변동성이 높아질 것이라는 예상이 많다는 뜻. 높은 시장 변동성이 예상될 때에는 투자자들이 주식매수를 기피하여, 하락장으로 이어질 개연성이 증가.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 정책불확실성 지수": {"desc": "신문기사 등 대중매체에서 경제 정책과 관련된 불확실성을 나타내는 용어의 빈도를 분석하여, 경제 전반의 불안정성과 경기를 예측하는 데 사용되는 지표. 텍스트 분석을 통해 경기 및 금융 시장의 변동성을 파악하는 실시간 바로미터 역할을 함.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "국내 CDS 프리미엄": {"desc": "Credit default Swap(신용부도스왑)은 채권의 신용위험을 전가하고자 하는 매입자가 일정한 수수료를 지급하는 대가로  기초자산의 채무불이행 등의 사건 발생시 매도자로부터 손실액 또는 일정 금액을 보전 받기로 약정하는 거래. 채무불이행에  대비한 보험과 유사. 한국 국채 5년물의 CDS 프리미엄이 상승한다는 것은 해외 투자자들이 곧 한국 경제가 좋지 않을 것으로 예상한다는 의미.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "외환변동성지수(JPM)": {"desc": "JP Morgan의 환율 변동성 지수. VIX가 S&P 500 시장에서의 기대변동성을 측정하는 것처럼 외환변동성 지수는 환율 시장의 기대변동성을 측정한 지표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "채권 기대변동성(MOVE)": {"desc": "ICE BofA MOVE 지수. 채권시장에 예상되는 가격의 변동성. 미국 금리 스왑에 대한 OTC 옵션 바스켓을 추적하여 측정. 옵션 가격에 내재된 미국 국채 금리의 변동성을 산출한 값.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "글로벌 금융 스트레스(BofA)": {"desc": "뱅크오브아메리카 메릴린치에서 집계하는 글로벌 금융 스트레스 지수. 메릴린치에서 집계하는 3개의 세부 지표(Risk Index, Flow Index, Skew Index)를 종합하여 산출.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "신흥국채권 스프레드(JPM EMBI)": {"desc": "EMBI 스프레드는 신흥시장의 채권 수익률과 미국 국채 수익률 간의 차를 의미. 스프레드가 확대된다는 것은 위험 자산인 신흥국 채권을 투자자가 불안하게 여겨 매수하지 않고 안전자산인 미국 국채의 매력이 상승함을 의미. 즉 두 국채 수익률의 차가 벌어진다는 것은 투자자들이 경제를 불안하게 여긴다는 지표로 해석 가능.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 하이일드 스프레드": {"desc": "Bloomberg US Corporate High Yield Average OAS. 미국 하이일드 회사채와 미국채 10년물의 금리 차이. 시장에 유동성이 귀해질수록 신용등급이 낮은 회사는 자금 조달이 어려워지고 하이일드 스프레드는 상승하므로, 하이일드 스프레드를 통해 시장의 유동성 레벨을 측정 가능. 하이일드 스프레드가 낮을수록 시장에 유동성이 풍부한 것으로 해석.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "美주간경제인덱스(WEI)": {"desc": "뉴욕 연은이 코로나 팬데믹 이후 개발한 주간 경제지수. WEI는 소비와 고용, 산업생산과 관련한 10개의 핵심 경제 자료에서 공통된 요인을 추출해 지수화 한 것으로, GDP의 전년동기대비 수치와 유사하도록 단위를 조정.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "글로벌경기선행지수": {"desc": "평균 주당노동시간, 신규 수주, 소비자 예상, 주택허가건수, 주식가격, 금리 스프레드 등을 포함한 10개의 선행지표를 조합하여 산출. 경제활동의 팽창과 둔화 사이의 전환점에 대한 조기 신호를 제공. OECD에서 월간 단위로 발표.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "국내 경기선행지수 순환변동치": {"desc": "향후 경기의 국면 및 전환점을 단기 예측하는데 활용. 통상 지표가 현재까지와 반대방향으로 2분기 이상 연속하여 움직이면 이 시점을 경기전환점 신호로 간주. 여기에 과거의 평균 선행시차를 더하면 향후 국면전환이 발생할 시점을 대략 추정 가능.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "미국 ISM제조업지수": {"desc": "미국 공급관리자협회가 발간하는 제조업 구매관리자 지수. 매달 400개 이상의 기업 구매/공급 관련 중역 대상으로 실시하는 설문조사 결과를 토대로 산출. 신규수주, 생산, 고용, 공급자 운송시간, 재고 등 5개의 지표에 가중치를 두어 산출하는 종합지수.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "미국 경기서프라이즈 지수": {"desc": "실제 발표된 경제지표가 시장 전망치에 부합한 정도를 나타낸 지수. 씨티그룹에서 산출.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "국내 수출증가율": {"desc": "수출이 증가하면 외환의 공급이 증가하므로 수출의 증감은 외환 수급에 직접적 영향을 주는 요소. 환율이 상승하면 국내 생산품의 수출 경쟁력이 높아져 수출 증가로 이어지기도 하므로, 환율과 수출에는 상관 고리가 존재.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "건화물 운임지수(BDI)": {"desc": "Baltic Dry Index. 발틱해운거래소에서 발표하는 건화물 운임 지수. 건화물은 생산에 활용하는 원자재로 철광석, 석탄, 곡물 등이 대표적. BDI의 상승은 원자재의 물동량 증가, 원자재 물동량 증가는 원자재 수요 상승이 원인. 따라서 BDI는 국제 건설 경기, 제조업 경기의 선행지표로 해석.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "글로벌주식모멘텀": {"desc": "MSCI ACWI 지수의 일별 로그수익률을 계산한 후, 이를 최근 250 거래일 합산하여 산출.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "외국인 순매수(60일 누계)": {"desc": "60일 누적 외국인 총합계 순매수 대금을 지표로 사용.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "국고채 장단기금리차(10Y-3Y)": {"desc": "한국 국고채 10년물과 3년물의 금리차.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "미 장단기금리차(10년-2년)": {"desc": "경기 선행지수를 구성하는 요소로서 향후 경기의 판단에 유용한 지표. 장단기 금리의 역전은 유동성 프리미엄의 소멸을 의미하며, 일반적으로 경기침체의 전조로 해석. 미국채 10년물 – 2년물 금리.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "원달러 환율": {"desc": "원달러 환율이 오를 경우 달러 대비 원화 가치는 절하, 달러 가치는 절상됨을 의미. 일반적으로 안전자산에 대한 선호가 증가할 때 달러 가치 절상이 발생하며, 이는 원화자산 및 위험자산 투자에는 리스크로 작용 가능.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "달러인덱스(DXY)": {"desc": "미국과 거래를 많이 하는 주요 통화인 유로화, 엔화, 파운드,  캐나다 달러, 스웨덴 크로네, 스위스 프랑과 비교한 달러의 상대적 가치를 가중 평균하여 측정. 일반적으로 달러는 안전 자산으로 간주되어 달러 강세일 경우 위험 자산 회피 및 안전 자산 선호 현상으로 해석.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국채 10년물 금리": {"desc": "미국채 금리 중 장기금리를 대표하여 차용되는 금리.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "국내 기준금리": {"desc": "한국은행 기준금리. 국내 금리 체계의 기준이 되는 금리이며 한국은행 소속 기관인 금융통화위원회에서 1년에 8번 결정. 중앙은행이 통화량을 조절하는 대표적인 방법으로 시중 유동성에 직결되는 요소.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 기대인플레이션(5년)": {"desc": "시장이 기대하고 있는 인플레이션 전망. 기대인플레이션은 5년물 BEI Rate(5년물 미국채 금리 – 5년물 물가연동국채 금리)으로 계산. 시장참여자들이 강한 인플레이션을 예상하는 경우, 물가연동국채를 적극 매입하며 물가연동국채의 금리는 하락하고 BEI Rate은 상승. 기대 인플레이션의 상승은 일반적으로 채권의 수요 감소로 연결.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "유가(WTI 최근월물)": {"desc": "유가의 상승은 공산품의 원재료비 상승 및 에너지 가격의 상승으로 연결. 이는 대체 에너지 사업에는 긍정적으로 작용할 수 있으나, 기타 원재료비가 중요한 공사 등의 사업에는 부정적. 사업의 종류에 따라 다른 방향으로 작용할 수 있으나, 어떤 방향으로든 유가의 상승은 대체투자 사업의 변동성 확대로 해석 가능.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 부동산담보대출 연체율": {"desc": "부동산담보대출 연체율의 증가는 임대 경기의 침체 또는 부동산 가격의 하락이 원인. 따라서 부동산담보대출 연체율의 증가는 부동산 시장의 침체 가능성 증가로 해석. 연방준비제도 발표 데이터, 전 미국 상업은행 대상 조사.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 모기지 금리(30년)": {"desc": "미국 주택 구매자들 사이에서 가장 인기 있는 주택담보대출 상품. 모기지 30년 금리는 향후 금리 전망과 함께 오르고 내리는 경향. 모기지 금리의 상승은 주택시장 침체 우려로 연결.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 주택가격 지수": {"desc": "케이스-쉴러 미국 주택가격 지수. S&P가 발표하는 대표적인 주택 가격지수. 미국의 주택가격은 고금리에도 상승하며 둔화하는 주택 건설 경기와 상반된 모습 관찰. 주택가격의 유지 또는 상승이 주택 건설 경기 반전으로 이어질 수 있을지 주의 필요.", "interpretation": "수치가 낮을수록, 리스크지수를 높게 환산했습니다."},
    "미국 상업용 부동산 공실률": {"desc": "미국 전체 도심 지역을 대상으로 조사한 상업용 부동산 공실률. 공실률은 건설업 밸류체인에서 최종 실수요를 보여주는 지표로, 향후 건설업의 사업성 판단에 큰 영향. 무디스 자회사 Reis에서 집계 및 발표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 상업용 부동산 공실률 (LA)": {"desc": "LA를 대상으로 조사한 상업용 부동산 공실률. 공실률은 건설업 밸류체인에서 최종 실수요를 보여주는 지표로, 향후 건설업의 사업성 판단에 큰 영향. 무디스 자회사 Reis에서 집계 및 발표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 상업용 부동산 공실률 (보스턴)": {"desc": "보스턴을 대상으로 조사한 상업용 부동산 공실률. 공실률은 건설업 밸류체인에서 최종 실수요를 보여주는 지표로, 향후 건설업의 사업성 판단에 큰 영향. 무디스 자회사 Reis에서 집계 및 발표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 상업용 부동산 공실률 (시카고)": {"desc": "시카고를 대상으로 조사한 상업용 부동산 공실률. 공실률은 건설업 밸류체인에서 최종 실수요를 보여주는 지표로, 향후 건설업의 사업성 판단에 큰 영향. 무디스 자회사 Reis에서 집계 및 발표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "미국 상업용 부동산 공실률 (애틀랜타)": {"desc": "애틀랜타를 대상으로 조사한 상업용 부동산 공실률. 공실률은 건설업 밸류체인에서 최종 실수요를 보여주는 지표로, 향후 건설업의 사업성 판단에 큰 영향. 무디스 자회사 Reis에서 집계 및 발표.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (뉴욕)": {"desc": "뉴욕 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (샌프란시스코)": {"desc": "샌프란시스코 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (파리)": {"desc": "파리 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (런던)": {"desc": "런던 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (베를린)": {"desc": "베를린 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (마드리드)": {"desc": "마드리드 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},
    "오피스 공실률 (멜버른)": {"desc": "멜버른 오피스 빌딩의 공실률.", "interpretation": "수치가 높을수록, 리스크지수를 높게 환산했습니다."},

    "국내 리스크종합지수": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 종합적 국내 리스크가 높습니다."},
    "글로벌 리스크종합지수": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 종합적 글로벌 리스크가 높습니다."},
    "국내주식리스크": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 국내 주식 리스크가 높습니다."},
    "글로벌주식리스크": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 글로벌 주식 리스크가 높습니다."},
    "채권리스크": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 채권 리스크가 높습니다."},
    "외환리스크": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 외환 리스크가 높습니다."},
    "크레딧/유동성리스크": {"desc": "세부 리스크 지표들의 산술평균.", "interpretation": "수치가 높을수록, 크레딧/유동성 리스크가 높습니다."}

}

# --- [설정] 국면별 툴팁 매핑 (라벨, CSS클래스, 설명) ---
state_map = {
    "Low": ("안정", "badge-low", "위험자산 확대 📈"),
    "Mid": ("중립", "badge-mid", "위험자산 유지 ⚖️"),
    "High": ("위험", "badge-high", "위험자산 축소 📉")
}

st.markdown("""
<style>
    .block-container {
        max-width: 1600px;
        padding-top: 1rem;
        padding-right: 2rem;
        padding-left: 2rem;
    }
    div[data-testid="stActionButton"] button[kind="header"]:nth-child(1) { display: none; }
    div[data-testid="stActionButton"] button[kind="header"]:nth-child(2) { display: none; }
    span[data-testid="stMainMenu"] { display: block; }
    
    /* 국면 배지 스타일 */
    .badge-low { background: #E8F5E9; color: #2E7D32; padding: 2px 6px; border-radius: 4px; cursor: help; }
    .badge-mid { background: #FFFDE7; color: #F9A825; padding: 2px 6px; border-radius: 4px; cursor: help; }
    .badge-high{ background: #FFEBEE; color: #C62828; padding: 2px 6px; border-radius: 4px; cursor: help; }

    /* 테이블 스타일 */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        font-family: "Source Sans Pro", sans-serif;
    }
    .custom-table th {
        background-color: #f0f2f6;
        color: #31333F;
        font-weight: 600;
        padding: 8px 12px;
        text-align: left;
        border-bottom: 2px solid #e6e9ef;
    }
    .custom-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #e6e9ef;
        color: #31333F;
    }
    .custom-table tr:hover {
        background-color: #f9f9f9;
    }
    
    /* 지표 툴팁 스타일 */
    .tooltip-text {
        text-decoration: underline dotted #888; 
        cursor: help;
    }
    
    /* 슬라이더와 차트 간격 줄이기 */
    div[data-testid="stSlider"] {
        margin-top: -15px;
        padding-top: 0px;
    }
    div[data-testid="stAltairChart"] {
        margin-bottom: -15px;
    }
    
    /* st.data_editor의 선택 열 헤더 이모티콘 제거 (숨기기) */
    /* data-testid="stWidgetLabel" 안에 있는 span 태그 중 이모티콘 역할을 하는 첫 번째 span을 숨김 */
    .st-emotion-cache-1wv939k > span:first-child { /* 이 클래스는 Streamlit 버전에 따라 변경될 수 있으나, 현재 구조를 기반으로 선택 */
        display: none !important;
    }

</style>
""", unsafe_allow_html=True)

rename_dict = {
    'Date.1': 'Date',
    '시티 매크로 리스크 지수': 'Citi 매크로 리스크',
    '국고채장단기금리차(10Y-3Y).1': '국고채 장단기금리차(10Y-3Y)',
    'VKOSPI.1': 'KOSPI200 기대변동성(VKOSPI)',
    '외국인 순매수 혹은 지분율': '외국인 순매수(60일 누계)',
    '美주간경제인덱스(WEI).1': '美주간경제인덱스(WEI)',
    'ISM제조업지수.1': '미국 ISM제조업지수',
    'VIX Index.1': 'S&P500 기대변동성(VIX)',
    'CBOE Put/Call Ratio(센티멘트).1': '글로벌경기선행지수',
    '시티 경기서프라이즈 지수.1': '미국 경기서프라이즈 지수',
    '미국 정책불확실성지수.1': '미국 정책불확실성 지수',
    '글로벌주식모멘텀.1': '글로벌주식모멘텀',
    '국내주식': '국내주식리스크',
    '해외주식': '글로벌주식리스크',
    '국내 BEI Rate.1': '국내 BEI Rate',
    'JPM EMBI Global Spread.1': '신흥국채권 스프레드(JPM EMBI)',
    'MOVE.1': '채권 기대변동성(MOVE)',
    '미국 기대물가(BEI 5년).1': '미국 기대인플레이션(5년)',
    '미국채 10년물 금리.1': '미국채 10년물 금리',
    '채권지수 ': '채권리스크',
    '수출증가율 .1': '국내 수출증가율',
    '한국 CDS 프리미엄.2': '국내 CDS 프리미엄',
    '한국1Y-미국1Y 금리차.1': '한국1Y-미국1Y 금리차',
    '외환변동성지수(JPM).1': '외환변동성지수(JPM)',
    '달러인덱스(DXY).1': '달러인덱스(DXY)',
    'FX 지수 ': '외환리스크',
    'CD(91일).1': '국내 기준금리',
    '한국 CDS 프리미엄.3': '국내 CDS 프리미엄',
    '미국 하이일드 스프레드(OAS).1': '미국 하이일드 스프레드',
    '장단기금리차(미국채 10년물-2년물).1': '미 장단기금리차(10년-2년)',
    'BofA 메릴린치 글로벌 금융 스트레스.1': '글로벌 금융 스트레스(BofA)',
    '크레딧 지수 ': '크레딧/유동성리스크',
    '한국 기준금리.1': '국내 기준금리',
    '유가(WTI 최근월물 CL1).1': '유가(WTI 최근월물)',
    '글로벌 운임지수.1': '건화물 운임지수(BDI)',
    '상업용 부동산 공실률(CRE Vacancy rate)-cbre.1': '미국 상업용 부동산 공실률',
    '공실률 LA': '미국 상업용 부동산 공실률 (LA)',
    '공실률 보스턴': '미국 상업용 부동산 공실률 (보스턴)',
    '공실률 시카고': '미국 상업용 부동산 공실률 (시카고)',
    '애틀랜타 공실률.1': '미국 상업용 부동산 공실률 (애틀랜타)',
    '뉴욕 공실률.1': '오피스 공실률 (뉴욕)',
    '샌프란시스코 공실률.1': '오피스 공실률 (샌프란시스코)',
    '파리 오피스 공실률': '오피스 공실률 (파리)',
    '런던 오피스 공실률': '오피스 공실률 (런던)',
    '베를린 오피스 공실률': '오피스 공실률 (베를린)',
    '마드리드 오피스 공실률': '오피스 공실률 (마드리드)',
    '멜버른 오피스 공실률': '오피스 공실률 (멜버른)',
    'Fed Delinquency rate on loians secured by RE all commercial banks': '미국 부동산담보대출 연체율',
    '미국 모기지 금리(30년).1': '미국 모기지 금리(30년)',
    'S&P Case-Shiller 주택가격 지수.1': '미국 주택가격 지수',
    '국내 리스크종합지수.1': '국내 리스크종합지수',
    '글로벌 리스크종합지수.1': '글로벌 리스크종합지수',
    '원달러환율.1': '원달러 환율',
    '국내 경기선행지수 순환변동치': '국내 경기선행지수 순환변동치_절대수치',
    '국내 경기선행지수 순환변동치.1': '국내 경기선행지수 순환변동치'
}


def fit_hmm_posterior(series: pd.Series, n_states: int = 3, random_state: int = 100):
    s = series.dropna()
    X = (s.values.reshape(-1, 1).astype(float) * 100.0)
    idx = s.index

    model = GaussianHMM(n_components=n_states, covariance_type="diag", random_state=random_state, n_iter=200, tol=1e-6, init_params="stmcw")
    model.fit(X)

    _, post = model.score_samples(X) 
    means = model.means_.flatten()
    order = np.argsort(means)
    label_map = {order[0]: "Low", order[1]: "Mid", order[2]: "High"}

    post_df = pd.DataFrame(post, index=idx, columns=[f"st{k}" for k in range(n_states)])
    post_df = post_df.rename(columns={f"st{k}": label_map[k] for k in range(n_states)})
    post_df = post_df[["Low","Mid","High"]]

    states = model.predict(X)
    state_labels = pd.Series([label_map[s] for s in states], index=idx, name="state")

    return post_df, state_labels

def run_and_export(risk_df: pd.DataFrame, target_col: str, random_state: int = 100):
    s = risk_df[target_col].astype(float)
    post, state_labels = fit_hmm_posterior(s, n_states=3, random_state=random_state)

    out = risk_df.copy()
    out = out.join(post, how="left")
    out.insert(0, "state", state_labels)
    out = out.reset_index().rename(columns={"Date": "Date"})
    # Keep only necessary state info to return
    out = out[["Date", target_col, "state", "Low", "Mid", "High"]]
    out.columns = ["Date", target_col, target_col+"_state", target_col+"_Low", target_col+"_Mid", target_col+"_High"]
    return out


@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('data/리스크보드New_v5_rawdata.csv', usecols=list(rename_dict.keys()))
    df = df[pd.to_datetime(df['Date.1']) < datetime.today() - timedelta(days=1)]
    
    composite = df[['Date.1', '국내주식', '해외주식', '채권지수 ', 'FX 지수 ', '크레딧 지수 ', '국내 리스크종합지수.1', '글로벌 리스크종합지수.1']]
    composite.columns = ['Date', 'K_EQUITY', 'G_EQUITY', 'FI', 'FX', 'CREDIT', 'KRCI', 'GRCI']
    
    for c in ["K_EQUITY", "G_EQUITY", "FI", "FX", "CREDIT", "KRCI", "GRCI"]:
        composite[c] = pd.to_numeric(composite[c], errors="coerce")
    
    # HMM 분석 (KRCI, GRCI)
    grci_out = run_and_export(composite, "GRCI", random_state=100)
    krci_out = run_and_export(composite, "KRCI", random_state=100)
    
    # [수정] 기존 composite 데이터프레임에 HMM 결과(상태정보)를 병합하여 모든 세부 지표 유지
    risk_df = composite.copy()
    
    # GRCI 정보 병합 (GRCI 값은 이미 composite에 있으므로 제외하고 상태 정보만 merge)
    risk_df = pd.merge(risk_df, grci_out[['Date', 'GRCI_state', 'GRCI_Low', 'GRCI_Mid', 'GRCI_High']], on='Date', how='left')
    
    # KRCI 정보 병합
    risk_df = pd.merge(risk_df, krci_out[['Date', 'KRCI_state', 'KRCI_Low', 'KRCI_Mid', 'KRCI_High']], on='Date', how='left')
    
    # 정리
    risk_df = risk_df.dropna(subset=['KRCI'])
    risk_df['Date'] = pd.to_datetime(risk_df['Date'])
    risk_df = risk_df.sort_values('Date', ascending=False).reset_index(drop=True)
    
    df.rename(columns=rename_dict, inplace=True)

    econ_df = df.dropna(subset=['국내 리스크종합지수'])
    econ_df['Date'] = pd.to_datetime(econ_df['Date'])
    econ_df = econ_df.sort_values('Date', ascending=False).reset_index(drop=True)
    econ_df = econ_df.loc[:, ~econ_df.columns.duplicated()]
    
    for col in risk_df.columns:
        # 숫자 변환 (Date와 state 컬럼 제외)
        if col not in ['Date', 'GRCI_state', 'KRCI_state']:
            risk_df[col] = pd.to_numeric(risk_df[col].astype(str).str.strip(), errors='coerce')
    
    for col in econ_df.columns:
        if col != 'Date':
            econ_df[col] = pd.to_numeric(econ_df[col].astype(str).str.strip(), errors='coerce')
    
    return risk_df, econ_df

def get_change_symbol(change):
    change = float(change)
    if np.isnan(change): return "-"
    if abs(round(change,2)) < 0.01: return "-"
    if change > 0: return f"▲ {abs(change):.2f}"
    elif change < 0: return f"▽ {abs(change):.2f}"
    else: return "-"

def color_change(val):
    if '▲' in str(val): return 'color: #D32F2F; font-weight: bold;'
    elif '▽' in str(val): return 'color: #1976D2; font-weight: bold;'
    else: return 'color: #9E9E9E'

def bytes_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# Indicators List
k_indicators = ["Citi 매크로 리스크", "국고채 장단기금리차(10Y-3Y)", "KOSPI200 기대변동성(VKOSPI)", "외국인 순매수(60일 누계)", "미국 정책불확실성 지수", "국내 CDS 프리미엄", "원달러 환율","국내 리스크종합지수"]
k_categories = ["국내·외 주식 등", "국내·외 주식 등", "국내주식", "국내주식", "국내·외 주식 등", "외환", "외환","합계"]
g_indicators = ["Citi 매크로 리스크", "美주간경제인덱스(WEI)", "S&P500 기대변동성(VIX)", "미국 정책불확실성 지수", "글로벌주식모멘텀", "신흥국채권 스프레드(JPM EMBI)", "외환변동성지수(JPM)", "달러인덱스(DXY)", "미 장단기금리차(10년-2년)","글로벌 리스크종합지수"]
g_categories = ["국내·외 주식 등", "국내·외 주식 등", "해외주식", "국내·외 주식 등", "국내·외 주식 등", "해외채권", "외환", "외환", "크레딧","합계"]
k_equity_indicators = ["Citi 매크로 리스크", "美주간경제인덱스(WEI)", "미국 정책불확실성 지수", "글로벌경기선행지수", "KOSPI200 기대변동성(VKOSPI)", "국고채 장단기금리차(10Y-3Y)", "외국인 순매수(60일 누계)", "국내 경기선행지수 순환변동치","국내주식리스크"]
k_equity_categories = ["공통요인", "공통요인", "공통요인", "공통요인", "국내 주식R 요인", "국내 주식R 요인", "국내 주식R 요인", "국내 주식R 요인","합계"]
g_equity_indicators = ["Citi 매크로 리스크", "美주간경제인덱스(WEI)", "미국 정책불확실성 지수", "글로벌경기선행지수", "미국 ISM제조업지수", "S&P500 기대변동성(VIX)", "미국 경기서프라이즈 지수", "글로벌주식모멘텀","글로벌주식리스크"]
g_equity_categories = ["공통요인", "공통요인", "공통요인", "공통요인", "글로벌 주식R 요인", "글로벌 주식R 요인", "글로벌 주식R 요인", "글로벌 주식R 요인","합계"]
fi_indicators = ['신흥국채권 스프레드(JPM EMBI)', '채권 기대변동성(MOVE)', '미국 기대인플레이션(5년)', '미국채 10년물 금리',"채권리스크"]
fi_categories = ["채권R 요인", "채권R 요인", "채권R 요인", "채권R 요인", "합계"]
fx_indicators = ['국내 수출증가율', '국내 CDS 프리미엄', '외환변동성지수(JPM)', '달러인덱스(DXY)',"외환리스크"]
fx_categories = ["외환R 요인", "외환R 요인", "외환R 요인", "외환R 요인", "합계"]
cr_indicators = ['국내 CDS 프리미엄', '미국 하이일드 스프레드', '미 장단기금리차(10년-2년)', '글로벌 금융 스트레스(BofA)',"크레딧/유동성리스크"]
cr_categories = ["크레딧/유동성R 요인", "크레딧/유동성R 요인", "크레딧/유동성R 요인", "크레딧/유동성R 요인", "합계"]
ai_indicators = ['국내 기준금리', '유가(WTI 최근월물)', '건화물 운임지수(BDI)', '미국 상업용 부동산 공실률', '미국 상업용 부동산 공실률 (LA)', '미국 상업용 부동산 공실률 (보스턴)', '미국 상업용 부동산 공실률 (시카고)', '미국 상업용 부동산 공실률 (애틀랜타)', '오피스 공실률 (뉴욕)', '오피스 공실률 (샌프란시스코)', '오피스 공실률 (파리)', '런던 오피스 공실률', '베를린 오피스 공실률', '마드리드 오피스 공실률', '멜버른 오피스 공실률', '미국 부동산담보대출 연체율', '미국 모기지 금리(30년)', '미국 주택가격 지수']
ai_categories = ["국내 대체투자R 요인", "공통 요인", "공통 요인", "해외 대체투자R 요인"] + ["해외 대체투자R 요인"] * 14

RCI_map = {"KRCI": "국내 리스크 종합지수", "GRCI": "글로벌 리스크 종합지수"}
RCI_IMJ_map = {"KRCI": 'https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f1f0-1f1f7.svg',
               "GRCI": 'https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f30f.svg'}

def main():
    st.write("")
    st.title("우체국보험 리스크 스코어보드")
    st.markdown("<div style='text-align: right; color: #909090;'>한국투자증권 리서치본부</div>", unsafe_allow_html=True)
    
    risk_df, econ_df = load_and_preprocess_data()
    available_dates = risk_df['Date'].dt.strftime('%Y-%m-%d').unique()
    default_chart_years = 10
    st.divider()
    st.write("")
  
    with st.sidebar:
        st.markdown("### 기준일자 선택")
        selected_date_str = st.selectbox("", available_dates)
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')
        st.divider()
        
        st.markdown("### 리스크 섹션 바로가기")
        sections = [
            ("국내 리스크 종합지수 (KRCI)", "krci-section"),
            ("글로벌 리스크 종합지수 (GRCI)", "grci-section"),
            ("국내주식리스크", "k-equity-section"),
            ("글로벌주식리스크", "g-equity-section"),
            ("채권리스크", "fi-section"),
            ("외환리스크", "fx-section"),
            ("크레딧/유동성리스크", "cr-section"),
            ("대체투자리스크", "ai-section"),
            ("지표별 종합 비교 분석", "idx-comparison-section")
        ]
        for section_name, section_id in sections:
            st.markdown(f"<a href='#{section_id}' onclick=\"document.getElementById('{section_id}').scrollIntoView({{behavior: 'smooth'}}); return false;\">{section_name}</a>", unsafe_allow_html=True)
        st.divider()

        st.markdown("### 데이터(.CSV) 다운로드")
        st.download_button("리스크 지표 DATA", data=bytes_csv(risk_df), file_name=f"risk_index_data_{selected_date.strftime('%Y%m%d')}.csv", mime="text/csv")
        st.download_button("경제 지표 DATA", data=bytes_csv(econ_df), file_name=f"economic_index_data_{selected_date.strftime('%Y%m%d')}.csv", mime="text/csv")
        st.divider()
        st.markdown("<p style='font-size: 10px; color:grey'> · 본 자료는 고객의 증권투자를 돕기 위하여 작성된 당사의 저작물로서 모든 저작권은 당사에게 있으며, 당사의 동의 없이 어떤 형태로든 복제, 배포, 전송, 변형할 수 없습니다.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 10px; color:grey'> · 본 자료는 리서치센터에서 수집한 자료 및 정보를 기초로 작성된 것이나 당사가 그 자료 및 정보의 정확성이나 완전성을 보장할 수는 없으므로 당사는 본 자료로써 고객의 투자 결과에 대한 어떠한 보장도 행하는 것이 아닙니다. 최종적 투자 결정은 고객의 판단에 기초한 것이며 본 자료는 투자 결과와 관련한 법적 분쟁에서 증거로 사용될 수 없습니다.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 10px; color:grey'> · 본 자료에 제시된 종목들은 리서치센터에서 수집한 자료 및 정보 또는 계량화된 모델을 기초로 작성된 것이나, 당사의 공식적인 의견과는 다를 수 있습니다.</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 10px; color:grey'> · 이 자료에 게재된 내용들은 작성자의 의견을 정확하게 반영하고 있으며, 외부의 부당한 압력이나 간섭 없이 작성되었음을 확인합니다.</p>", unsafe_allow_html=True)

    idx = risk_df[risk_df['Date'] == selected_date].index[0]
    previous_data = {}
    if idx + 1 < len(risk_df): previous_data['next'] = {'risk': risk_df.iloc[idx + 1], 'econ': econ_df.iloc[idx + 1]}
    else: previous_data['next'] = None
    if idx + 4 < len(risk_df): previous_data['month'] = {'risk': risk_df.iloc[idx + 4], 'econ': econ_df.iloc[idx + 4]}
    else: previous_data['month'] = None
    if idx + 52 < len(risk_df): previous_data['year'] = {'risk': risk_df.iloc[idx + 52], 'econ': econ_df.iloc[idx + 52]}
    else: previous_data['year'] = None

    current_risk = risk_df[risk_df['Date'] == selected_date].iloc[0]
    current_econ = econ_df[econ_df['Date'] == selected_date].iloc[0]
    previous_risk = previous_data['next']['risk'] if previous_data['next'] else None
    previous_econ = previous_data['next']['econ'] if previous_data['next'] else None
    previous_month_risk = previous_data['month']['risk'] if previous_data['month'] else None
    previous_month_econ = previous_data['month']['econ'] if previous_data['month'] else None
    previous_year_risk = previous_data['year']['risk'] if previous_data['year'] else None
    previous_year_econ = previous_data['year']['econ'] if previous_data['year'] else None

    if not previous_data['next']:
        st.warning("이전 데이터가 없습니다.")
        return

    def format_tooltip_html(val):
        if val in tooltip_data:
            info = tooltip_data[val]
            tooltip_text = f"{info['desc']} &#10;[{info['interpretation']}]"
            return f'<span title="{tooltip_text}" class="tooltip-text">{val}</span>'
        return val

    def RISK_SECTION(name, indicators, categories):
        is_composite = name in ("KRCI", "GRCI")
        is_alt = (name == "대체투자리스크")

        if is_composite:
            base_series_now = current_risk
            base_series_prev = previous_risk
            base_series_prev_m = previous_month_risk
            base_series_prev_y = previous_year_risk
        else:
            base_series_now = current_econ
            base_series_prev = previous_econ
            base_series_prev_m = previous_month_econ
            base_series_prev_y = previous_year_econ

        wow_change = "-"
        mom_change = "-"
        yoy_change = "-"
        wow_color = "color: black;"
        mom_color = "color: black;"
        yoy_color = "color: black;"

        if not is_alt:
            wow_change = get_change_symbol(base_series_now[name] - base_series_prev[name])
            mom_change = get_change_symbol(base_series_now[name] - base_series_prev_m[name])
            yoy_change = get_change_symbol(base_series_now[name] - base_series_prev_y[name])
            wow_color = color_change(wow_change)
            mom_color = color_change(mom_change)
            yoy_color = color_change(yoy_change)

        current_vals = [current_econ.get(col, np.nan) for col in indicators]
        previous_vals = [previous_econ.get(col, np.nan) for col in indicators]
        table_data = {
            "지표": indicators,
            "세부 리스크": categories,
            "이전": [f"{p:.2f}" if not np.isnan(p) else "-" for p in previous_vals],
            "현재": [f"{c:.2f}" if not np.isnan(c) else "-" for c in current_vals],
            "변화": [get_change_symbol(c - p if not np.isnan(c) and not np.isnan(p) else np.nan) for c, p in zip(current_vals, previous_vals)],
        }
        df_table = pd.DataFrame(table_data)

        if is_composite: chart_base = risk_df
        else: chart_base = econ_df

        min_ts = chart_base["Date"].min()
        max_ts = chart_base["Date"].max()
        min_date = min_ts.to_pydatetime() if isinstance(min_ts, pd.Timestamp) else min_ts
        max_date = max_ts.to_pydatetime() if isinstance(max_ts, pd.Timestamp) else max_ts
        
        default_start = max(min_date, selected_date - timedelta(days=365 * default_chart_years))
        default_end = selected_date

        anchor_id = f"{name.lower()}-section" if is_composite else next((sec_id for sec_name, sec_id in sections if sec_name == name), None)
        if anchor_id: st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)
                
        with st.container(border=True):
            if is_composite:
                state_label, badge_class, state_desc = state_map.get(current_risk[f"{name}_state"], ("Unknown", "badge-mid", "정보 없음"))
                st.markdown(f"""
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <img src='{RCI_IMJ_map[name]}' style='height: 32px; margin-right: 6px;'>
                        <h3 style='margin:0; padding:0;'> {RCI_map[name]} : {current_risk[name]:.2f}</h3>
                        <span class='{badge_class}' title='{state_desc}' style='margin-left:10px; font-weight: bold; font-size: 1.6em;'>{state_label}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='margin-bottom: 15px;'>
                    <span style='color: gray;'>· 국면별 확률 : </span>
                    <span style='color: #2E7D32;'>안정({(current_risk[f'{name}_Low']*100):.2f}%)</span>, 
                    <span style='color: #F9A825;'>중립({(current_risk[f'{name}_Mid']*100):.2f}%)</span>, 
                    <span style='color: #C62828;'>위험({(current_risk[f'{name}_High']*100):.2f}%)</span>
                </div>
                """, unsafe_allow_html=True)

            elif is_alt:
                st.markdown(f"<h3 style='margin-bottom: 15px;'> {name}</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='margin-bottom: 15px;'> {name} : {base_series_now[name]:.2f}</h3>", unsafe_allow_html=True)
            
            if not is_alt:
                st.markdown(f"""
                <div style='margin-bottom: 10px; font-size: 0.9em;'>
                    · 전주대비 : <span style='{wow_color}'>{wow_change}</span> &nbsp;
                    · 전월대비 : <span style='{mom_color}'>{mom_change}</span> &nbsp;
                    · 전년대비 : <span style='{yoy_color}'>{yoy_change}</span>
                </div>
                """, unsafe_allow_html=True)
                st.divider()

            if is_alt:
                # 대체투자리스크: 기존 스타일 유지 (선택 기능 없음)
                styler = df_table.style.format({'지표': format_tooltip_html}).applymap(color_change, subset=["변화"])
                st.markdown(styler.hide(axis="index").set_table_attributes('class="custom-table"').to_html(escape=False), unsafe_allow_html=True)
            else:
                # 대체투자 아님: st.data_editor 사용 (선택 기능 포함)
                desc_placeholder = st.empty()
                chart_placeholder = st.empty()
                slider_placeholder = st.empty()

                # [State 관리] 현재 선택된 인덱스 (디폴트: 마지막 행)
                ss_key = f"selected_idx_{name}"
                if ss_key not in st.session_state:
                    st.session_state[ss_key] = len(df_table) - 1

                # 데이터 준비: '선택' 컬럼 추가 및 초기화
                df_table["선택"] = False
                current_idx = st.session_state[ss_key]
                # 인덱스 유효성 체크
                if current_idx >= len(df_table):
                    current_idx = len(df_table) - 1
                    st.session_state[ss_key] = current_idx
                
                df_table.at[current_idx, "선택"] = True
                
                # 컬럼 순서 변경: '선택'을 맨 앞으로
                cols = ["선택"] + [c for c in df_table.columns if c != "선택"]
                df_table = df_table[cols]

                # Config: 체크박스 설정 및 다른 컬럼 수정 불가 처리
                column_config = {
                    # 선택 열 너비 최소화
                    "선택": st.column_config.CheckboxColumn(
                        "선택",
                        width="small", 
                        default=False
                    )
                }
                disabled_cols = [c for c in df_table.columns if c != "선택"]

                # Data Editor 표시
                edited_df = st.data_editor(
                    df_table,
                    column_config=column_config,
                    disabled=disabled_cols,
                    hide_index=True,
                    use_container_width=True,
                    key=f"editor_{name}"
                )

                # 변경 감지 및 단일 선택 로직
                selected_rows = edited_df[edited_df["선택"] == True].index.tolist()
                
                new_selection = current_idx
                
                if len(selected_rows) == 0:
                    # 다 해제됨 -> 강제로 이전 선택 유지 (Rerun으로 복구)
                    st.session_state[ss_key] = current_idx
                    st.rerun()
                elif len(selected_rows) > 1:
                    # 2개 이상 선택됨 -> 새로 체크된 것을 찾음
                    for idx in selected_rows:
                        if idx != current_idx:
                            new_selection = idx
                            break
                    st.session_state[ss_key] = new_selection
                    st.rerun()
                else:
                    # 1개만 선택됨 (정상)
                    if selected_rows[0] != current_idx:
                        st.session_state[ss_key] = selected_rows[0]
                        st.rerun()
                
                # 차트 및 설명 업데이트 (현재 선택된 target_col 기준)
                target_col = df_table.iloc[current_idx]["지표"] 
                
                target_desc = ""
                target_interpret = ""
                
                if target_col in tooltip_data:
                    target_desc = tooltip_data[target_col]['desc']
                    target_interpret = tooltip_data[target_col]['interpretation']

                if target_desc:
                    desc_placeholder.info(f"**{target_col}**: {target_desc} \n\n {target_interpret}", icon="💡")

                if target_col in risk_df.columns:
                    chart_base = risk_df
                else:
                    chart_base = econ_df

                with slider_placeholder.container():
                    range_key = f"{name}_range_{target_col}" 
                    start_date, end_date = st.slider(
                        "조회 기간",
                        min_value=min_date,
                        max_value=max_date,
                        value=(default_start, default_end),
                        format="YYYY-MM-DD",
                        key=range_key,
                        label_visibility="collapsed"
                    )

                chart_df = chart_base[
                    (chart_base["Date"] >= start_date) & 
                    (chart_base["Date"] <= end_date)
                ][["Date", target_col]]

                chart = alt.Chart(chart_df).mark_line(color="grey").encode(
                    x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y-%m")),
                    y=alt.Y(f"{target_col}:Q", scale=alt.Scale(zero=False), title=None),
                    tooltip=["Date", alt.Tooltip(target_col, format=".2f")]
                ).properties(height=200)

                chart_placeholder.altair_chart(chart, use_container_width=True)

    st.subheader("종합 리스크지표", divider="grey")
    col1, col2 = st.columns(2, gap="large")
    with col1: RISK_SECTION("KRCI", k_indicators, k_categories)
    with col2: RISK_SECTION("GRCI", g_indicators, g_categories)
    
    st.write("")
    st.write("")

    st.subheader("세부 리스크지표", divider="grey")
    col3, col4 = st.columns(2, gap="large")
    with col3: RISK_SECTION("국내주식리스크",k_equity_indicators,k_equity_categories)
    with col4: RISK_SECTION("글로벌주식리스크",g_equity_indicators,g_equity_categories)
    
    col5, col6 = st.columns(2, gap="large")
    with col5: RISK_SECTION("채권리스크",fi_indicators,fi_categories)
    with col6: RISK_SECTION("외환리스크",fx_indicators,fx_categories)
    
    col7, col8 = st.columns(2, gap="large")
    with col7: RISK_SECTION("크레딧/유동성리스크",cr_indicators,cr_categories)
    with col8: RISK_SECTION("대체투자리스크",ai_indicators,ai_categories)

    # --- [추가] 하단 종합 비교 차트 섹션 ---
    st.markdown(f'<div id="idx-comparison-section"></div>', unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.subheader("지표별 종합 비교 분석", divider="grey")

    # 비교할 지표 매핑
    # KRCI -> 국내 리스크 종합지수 (KRCI)
    # G_EQUITY -> 글로벌주식리스크
    comp_map = {
        "국내 리스크 종합지수 (KRCI)": "KRCI",
        "글로벌 리스크 종합지수 (GRCI)": "GRCI",
        "국내주식리스크": "K_EQUITY",
        "글로벌주식리스크": "G_EQUITY",
        "채권리스크": "FI",
        "외환리스크": "FX",
        "크레딧/유동성리스크": "CREDIT"
    }
    
    # 영문 코드 -> 한글 이름으로의 역 매핑 딕셔너리 생성
    # {"KRCI": "국내 리스크 종합지수 (KRCI)", "G_EQUITY": "글로벌주식리스크", ...}
    reverse_comp_map = {v: k for k, v in comp_map.items()}

    # 차트 컨트롤 UI
    with st.container(border=True):
        # 다중 선택 박스
        selected_names = st.multiselect(
            "비교할 지표를 선택하세요",
            options=list(comp_map.keys()),
            default=["국내 리스크 종합지수 (KRCI)", "글로벌 리스크 종합지수 (GRCI)"] # 기본 선택값
        )
        
        if selected_names:
            # 선택된 컬럼명 매핑 (한글 이름 -> 영문 코드)
            selected_cols = [comp_map[name] for name in selected_names]
            
            # 차트용 데이터 준비 (전체 기간)
            min_ts = risk_df["Date"].min()
            max_ts = risk_df["Date"].max()
            min_date = min_ts.to_pydatetime() if isinstance(min_ts, pd.Timestamp) else min_ts
            max_date = max_ts.to_pydatetime() if isinstance(max_ts, pd.Timestamp) else max_ts

            # [수정] 차트가 그려질 공간을 먼저 확보 (위쪽)
            chart_placeholder = st.empty()

            # [수정] 슬라이더 생성 (아래쪽)

            start_date, end_date = st.slider(
                "조회 기간 설정",
                min_value=min_date,
                max_value=max_date,
                # 'default_chart_years' 변수가 정의되어 있지 않으므로 3년으로 고정
                value=(max(min_date, selected_date - timedelta(days=365*default_chart_years)), selected_date),
                format="YYYY-MM-DD",
                key="comp_chart_slider",
                label_visibility="collapsed"
            )
            
            # [수정] 슬라이더 값을 이용한 데이터 필터링 로직
            chart_data = risk_df[
                (risk_df["Date"] >= start_date) & 
                (risk_df["Date"] <= end_date)
            ][["Date"] + selected_cols]
            
            melted_df = chart_data.melt("Date", var_name="지표_코드", value_name="값")
            
            # 컬럼 이름이 '지표_코드'이므로, '지표' 컬럼을 새로 만들어 한글 이름으로 매핑
            melted_df['지표'] = melted_df['지표_코드'].map(reverse_comp_map)
            
            # Altair 멀티라인 차트
            chart = alt.Chart(melted_df).mark_line().encode(
                x=alt.X("Date:T", axis=alt.Axis(format="%Y-%m"), title=None),
                y=alt.Y("값:Q", scale=alt.Scale(zero=False), title=None),
                # 여기서 '지표:N'을 사용하여 한글 이름을 범례에 표시
                color=alt.Color("지표:N", legend=alt.Legend(title="", orient="top")),
                tooltip=["Date", "지표", alt.Tooltip("값", format=".2f")]
            ).properties(height=400)
            
            # [수정] 확보해둔 공간에 차트 그리기
            chart_placeholder.altair_chart(chart, use_container_width=True)
            
        else:
            st.info("비교할 지표를 하나 이상 선택해주세요.")

if __name__ == "__main__":
    main()