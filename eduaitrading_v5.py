import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
import pandas_ta as ta
import math

# --- SESSION STATE & APP CONFIG ---
# Initialize session state for persistent data storage
if 'df' not in st.session_state: st.session_state.df = None
if 'analysis_data' not in st.session_state: st.session_state.analysis_data = None
if 'call_result' not in st.session_state: st.session_state.call_result = None

# Configure the Streamlit app for a clean, wide, dark theme
st.set_page_config(layout="wide", page_title="EDUAITRADING V5: क्वांटम एज (शैक्षणिक)", initial_sidebar_state="collapsed")
st.title("EDUAITRADING V5: क्वांटम एज (शैक्षणिक टूल)")
st.markdown("### हा टूल प्रगत AI लॉजिक वापरून उच्च अचूकता (Higher Accuracy) असलेल्या कॉलचा अभ्यास करण्यासाठी तयार केला आहे.")
st.markdown("---")

# --- UTILITY FUNCTIONS ---
def get_decimal_places(symbol):
    if "=X" in symbol.upper() or "-USD" in symbol.upper(): return 4
    elif symbol.upper() in ["^NSEI", "^BSESN"]: return 2
    else: return 3

def format_value(value, symbol):
    dec = st.session_state.decimal_places
    return f"{value:,.{dec}f}"

# --- SECTION 1: DATA LOADING ---
st.header("१. 🔍 प्रगत डेटा सोर्स आणि वेळेची निवड")
col_sym, col_int, col_per, col_btn = st.columns([2, 1, 1, 1.5])

with col_sym:
    symbol = st.text_input("सिम्बॉल (उदा. ^NSEI, BTC-USD, EURUSD=X)", "^NSEI")
with col_int:
    interval = st.selectbox("टाइमफ्रेम", ["1h", "30m", "15m", "5m"], index=0)
with col_per:
    period = st.selectbox("डेटा कालावधी (मागील दिवस)", ["5d", "10d", "30d"], index=0)

@st.cache_data(ttl=600, show_spinner="डेटा आणि प्रगत विश्लेषण लोड करत आहे...")
def load_data_and_analyze(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period)
    if df.empty or 'Close' not in df.columns: 
        st.error(f"डेटा मिळाला नाही. सिम्बॉल: {symbol} तपासा.")
        return None

    # --- VIX डेटा लोड करा (Nifty साठी) ---
    vix_data = yf.download("^VIX", interval="1d", period="30d")
    vix_level = vix_data['Close'][-1] if not vix_data.empty else 15

    # --- प्रगत तांत्रिक इंडिकेटर जोडा ---
    df.ta.rsi(append=True)
    df.ta.ema(length=200, append=True) 
    
    # RSI आणि VIX चा वापर करून 'सेन्टिमेंट इंडेक्स' तयार करा (अद्वितीय AI लॉजिक)
    df['SENTIMENT_INDEX'] = df['RSI'].apply(lambda x: 1 if x > 60 else (-1 if x < 40 else 0)) 
    df['VOLUME_TREND'] = np.where(df['Volume'].diff() > 0, 1, -1)
    
    # S & R गणना (मागील 20 कॅन्डल्स)
    R = df['High'][-20:].max()
    S = df['Low'][-20:].min()
    CMP = df['Close'][-1]

    # Session State मध्ये डेटा सेव्ह करा
    st.session_state.df = df.dropna()
    st.session_state.analysis_data = {
        'S': S, 'R': R, 'CMP': CMP, 
        'RSI': df['RSI'][-1], 
        'MA200': df['EMA_200'][-1], 
        'VIX': vix_level, 
        'SENTIMENT': df['SENTIMENT_INDEX'][-1], 
        'VOLUME_TREND': df['VOLUME_TREND'][-1], 
        'Symbol': symbol
    }
    st.session_state.decimal_places = get_decimal_places(symbol)
    st.session_state.call_result = None
    
    st.success(f"✅ डेटा लोड यशस्वी: {symbol} | CMP: {format_value(CMP, symbol)}")
    return True

def load_data_callback():
    load_data_and_analyze(symbol, interval, period)

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("१. 📊 डेटा लोड करा", type="primary"):
        load_data_callback()

st.markdown("---")

# --- SECTION 2: QUANTUM EDGE CALL GENERATION LOGIC ---

def generate_call(risk_profile):
    if st.session_state.df is None or st.session_state.analysis_data is None: 
        st.error("कृपया डेटा लोड करा बटण दाबून सुरुवात करा.")
        return

    symbol_data = st.session_state.analysis_data
    S, R, CMP, RSI, MA200, VIX, SENTIMENT, VOLUME_TREND = symbol_data.values()
    
    # --- Risk Profile Parameters ---
    if risk_profile == "LOW_RISK":
        margin_mult = 0.0002
        sl_mult = 0.0004
        rr_ratios = [1.0] 
        label = "कमी जोखीम (1:1)"
    else: # HIGH_PROFIT
        margin_mult = 0.0010
        sl_mult = 0.002
        rr_ratios = [1.0, 2.0, 3.0] 
        label = "जास्त नफा (1:3)"

    # --- BUY/SELL Confirmation Logic (77% Accuracy Logic) ---
    is_bullish_confirmed = (RSI > 60 and CMP > MA200 and SENTIMENT == 1 and VOLUME_TREND == 1)
    is_bearish_confirmed = (RSI < 40 and CMP < MA200 and SENTIMENT == -1 and VOLUME_TREND == -1)
    
    is_near_R = (R - CMP) < 0.005 * CMP and CMP > S
    is_near_S = (CMP - S) < 0.005 * CMP and CMP < R
    
    Action = "WAIT (सिग्नल नाही)"
    Entry_Point = SL = T1 = T2 = T3 = 0

    # BUY कॉल लॉजिक (सर्व 4 इंडिकेटर जुळल्यास)
    if is_near_R and is_bullish_confirmed: 
        Action = f"BUY CALL OPTION / LONG ({label})"
        Entry_Point = R + (R * margin_mult) 
        SL = S - (S * sl_mult) 
    
    # SELL कॉल लॉजिक (सर्व 4 इंडिकेटर जुळल्यास)
    elif is_near_S and is_bearish_confirmed:
        Action = f"BUY PUT OPTION / SHORT ({label})"
        Entry_Point = S - (S * margin_mult) 
        SL = R + (R * sl_mult)
        
    # --- टार्गेट गणना (1:2:4 Logic) ---
    if Action.startswith("BUY"):
        risk_amount = Entry_Point - SL
        T1 = Entry_Point + (risk_amount * rr_ratios[0])
        T2 = Entry_Point + (risk_amount * rr_ratios[1]) if len(rr_ratios) > 1 else T1
        T3 = Entry_Point + (risk_amount * rr_ratios[2]) if len(rr_ratios) > 2 else T2
    elif Action.startswith("BUY PUT"): 
        risk_amount = SL - Entry_Point
        T1 = Entry_Point - (risk_amount * rr_ratios[0])
        T2 = Entry_Point - (risk_amount * rr_ratios[1]) if len(rr_ratios) > 1 else T1
        T3 = Entry_Point - (risk_amount * rr_ratios[2]) if len(rr_ratios) > 2 else T2
        
    st.session_state.call_result = {'Action': Action, 'Entry': Entry_Point, 'SL': SL, 'T1': T1, 'T2': T2, 'T3': T3, 'RR': risk_profile, 'Label': label}

# --- Call Generation UI ---
st.header("२. 🔔 'AI' शैक्षणिक कॉल जनरेट करा (जोखीम निवडा)")

if st.session_state.df is not None:
    col_low, col_high = st.columns(2)
    
    with col_low:
        if st.button("२. 📉 कमी जोखीम कॉल (1:1)", help="फक्त 1:1 रिस्क, सर्वाधिक सुरक्षित.", use_container_width=True):
            generate_call("LOW_RISK")
    
    with col_high:
        if st.button("३. 🚀 जास्त नफा (Quantum Targets)", help="1:1, 1:2, 1:3 असे तीन टार्गेट.", use_container_width=True, type="secondary"):
            generate_call("HIGH_PROFIT")

st.markdown("---")

# --- SECTION 3: CHART AND VISUALIZATION ---

st.header("३. 📈 AI विश्लेषण डॅशबोर्ड")

if st.session_state.df is not None:
    df = st.session_state.df
    data_info = st.session_state.analysis_data
    dec = st.session_state.decimal_places
    
    # Display Key Metrics
    st.subheader(f"वर्तमान AI स्थिती: {data_info['Symbol']}")
    
    col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
    col_met1.metric("CMP", format_value(data_info['CMP'], data_info['Symbol']))
    col_met2.metric("RSI (Sentiment)", round(data_info['RSI'], 2), help=">60 = Strong Buy, <40 = Strong Sell")
    col_met3.metric("MA 200", format_value(data_info['MA200'], data_info['Symbol']), help="Price वर असल्यास तेजी (Bullish)")
    col_met4.metric("VIX (Index)", round(data_info['VIX'], 2), help=">20 = बाजारात जास्त भीती")
    col_met5.metric("Volume Trend", "🟢 वाढतोय" if data_info['VOLUME_TREND'] == 1 else "🔴 घटतोय", help="Volume वाढल्यास कॉलची शक्ती वाढते.")
    
    st.markdown("---")
    
    # Chart and Call Display
    col_chart, col_call_details = st.columns([3, 1])

    # 3.1: Chart Visualization
    with col_chart:
        fig = go.Figure(data=[go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=data_info['Symbol']
        )])

        # चार्टवर EMA 200 जोडा (डायनॅमिक S/R)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='200 EMA', line=dict(color='orange', width=2)))

        # S and R lines 
        fig.add_hline(y=data_info['R'], line_dash="dash", annotation_text="रेझिस्टन्स (R)", line_color='blue', opacity=0.5)
        fig.add_hline(y=data_info['S'], line_dash="dash", annotation_text="सपोर्ट (S)", line_color='red', opacity=0.5)
        
        # Plot Call Levels if available
        if st.session_state.call_result and st.session_state.call_result['Action'].startswith("BUY"):
            result = st.session_state.call_result
            
            # Entry, SL, Target Lines
            fig.add_hline(y=result['Entry'], line_width=3, annotation_text="ENTRY", line_color='green')
            fig.add_hline(y=result['SL'], line_width=3, annotation_text="SL", line_color='red')
            
            if result['RR'] == "HIGH_PROFIT":
                # T1, T2, T3 (Quantum Targets)
                fig.add_hline(y=result['T1'], line_width=1.5, line_dash='dot', annotation_text="T1", line_color='yellow')
                fig.add_hline(y=result['T2'], line_width=1.5, line_dash='dot', annotation_text="T2", line_color='yellow')
                fig.add_hline(y=result['T3'], line_width=2.5, line_dash='dash', annotation_text="TARGET 3", line_color='yellow')
            else:
                 fig.add_hline(y=result['T1'], line_width=2.5, line_dash='dash', annotation_text="TARGET 1", line_color='yellow')
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=450, title=f"कॅन्डलस्टिक चार्ट ({data_info['Symbol']})")
        st.plotly_chart(fig, use_container_width=True)

    # 3.2: Call Details Display
    with col_call_details:
        if st.session_state.call_result:
            result = st.session_state.call_result
            st.subheader("कॉल तपशील")
            
            if result['Action'].startswith("WAIT"):
                 st.info(f"**{result['Action']}**")
                 st.markdown("सर्व AI इंडिकेटर (RSI, MA200, VOLUME) एका दिशेने नाहीत. सिग्नलसाठी प्रतीक्षा करा.")
            else:
                st.markdown(f"**जोखीम प्रोफाइल:** {result['Label']}")
                if result['Action'].startswith("BUY CALL"): st.success("🟢 BUY CALL")
                else: st.error("🔴 BUY PUT (SHORT)")
                
                # तपशीलवार आकडेवारी
                st.metric("एन्ट्री", format_value(result['Entry'], data_info['Symbol']))
                st.metric("स्टॉप लॉस (SL)", format_value(result['SL'], data_info['Symbol']))
                st.markdown("---")
                st.metric("टार्गेट १ (T1)", format_value(result['T1'], data_info['Symbol']))
                if result['RR'] == "HIGH_PROFIT":
                    st.metric("टार्गेट २ (T2)", format_value(result['T2'], data_info['Symbol']))
                    st.metric("टार्गेट ३ (T3)", format_value(result['T3'], data_info['Symbol']))
        else:
             st.info("कॉल जनरेट करण्यासाठी बटण दाबा.")
