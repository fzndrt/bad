
import io, time, math, warnings, requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="BEI Rally Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HF_API = "https://datasets-server.huggingface.co/rows?dataset=kjhq/Indonesia-Stock-Symbols-and-Metadata&config=default&split=train&offset=0&length=1000"

# ============================================================
# UNIVERSE
# ============================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_universe():
    try:
        r = requests.get(HF_API, timeout=20)
        r.raise_for_status()
        rows = r.json()["rows"]
        data = pd.DataFrame([x["row"] for x in rows])
        if "ticker" in data.columns:
            data["ticker"] = (
                data["ticker"].astype(str).str.upper().str.replace(".JK", "", regex=False)
            )
            data = data[data["ticker"].str.fullmatch(r"[A-Z]{4}")]
            return data.drop_duplicates("ticker").reset_index(drop=True)
    except Exception:
        pass

    # Fallback for core IDX names if the public universe source is unavailable.
    fallback = """
    AADI ABBA ABDA ABMM ACES ACRO ACST ADCP ADES ADHI ADMF ADMG ADMR ADRO AGII AGRO
    AHAP AIMS AISA AKKU AKPI AKRA AKSI ALDO ALKA ALMI ALTO AMAR AMFG AMIN AMMN AMRT
    ANJT ANTM APEX APIC APII APLI APLN ARCI ARII ARNA ARTA ASBI ASDM ASGR ASII ASLC
    ASRI ASSA ATAP ATIC AUTO AVIA AWAN AYAM BALI BANK BBCA BBNI BBRI BBTN BBYB BCAP
    BDMN BEST BFIN BGTG BHIT BIKA BIRD BISI BJBR BJTM BKSL BMRI BMTR BNBA BNGA BNII
    BNLI BRIS BRMS BRPT BSDE BSIM BSSR BTEK BTEL BTPS BUDI BUVA BYAN CAKK CAMP CASS
    CBPE CCSI CITA CTRA CYBR DATA DEWA DGIK DMAS DOID DPUM DSNG DSSA DUTI DVLA DYOB
    ELSA ELTY EMDE EMTK ENRG ERAA ESIP ESSA EXCL FAST FILM FIRE FREN GIAA GJTL GOTO
    GPRA GREN GGRM HEAL HERO HEXA HMSP HOME HRME HRUM ICBP ICON IDPR IFSH IGAR IMAS
    INAF INCO INDF INDS INET INKP INOV INPC INTP IPCC IPPE IPOL IPTV IRRA ISAT ISSP
    ITMG JGLE JMAS JPFA JRPT JSMR KBAG KBLI KBLM KDSI KEEN KETR KICI KIJA KLBF KLIN
    KMDS KOBX KOIN KPIG KRAS LABA LAPI LCKM LEAD LINK LION LIVE LMAS LPCK LPKR LPPF
    LPPS LSIP LTLS MABA MAIN MAPA MAPI MARK MASA MAYA MBAP MBMA MBSS MCAS MCOR MDKA
    MDLN MEDC MEGA MENN MERK META MIKA MINA MLBI MLIA MLPL MMIX MNCN MPPA MPMX MRAT
    MTEL MTLA MTDL MTEL MYOH MYOR NANO NASA NASI NCKL NISP NKEN NOBU NRCA OBMD OILS
    OMED OMRE PADI PALM PAMG PANR PANS PBID PBSA PBRX PDPP PEGE PGAS PGEO PGJO PKPK
    PLIN PNBN PNLF PNSE POLA POLI POLL PORT POWR PPGL PPRE PPRO PTPP PTRO PTPW PTBA
    PTIS PWON PYFA RAAM RALS RATU RBMS RDTX REAL RELI RGAS RIMO RISE RLCM RMKE RMKO
    ROTI RUIS SAFE SAMF SAME SIDO SILO SIMP SIPD SKBM SKRN SMAR SMBR SMCB SMDR SMGR
    SMKL SMMA SMRA SMSM SOCI SOHO SOSS SPMA SRTG SSIA SSMS STAA STAR STEV SUGI SULI
    SUNI SUPR SWAT TAMA TBIG TCPI TDPM TELE TFAS TGKA TGRA TINS TIRA TKIM TLDN TMAS
    TOWR TPMA TRAM TRIN TRIS TRJA TRON TRST TRUE TUGU TURI UANG UCID ULTJ UNTR UNVR
    VAST VICI VINS VISI VIVA VKTR WAPO WEGE WEHA WIKA WIFI WIIM WINS WTON WSBP WSKT
    WTON XEPM YPAS ZATA
    """
    tickers = sorted(set(fallback.split()))
    return pd.DataFrame({"ticker": tickers})

# ============================================================
# PRICE DATA
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def download_prices(tickers, period="2y"):
    tickers = [t if t.endswith(".JK") else t + ".JK" for t in tickers]
    out = {}
    # yfinance can fail on very large batches; keep chunks moderate.
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i+40]
        try:
            x = yf.download(
                batch,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            if x.empty:
                continue
            if isinstance(x.columns, pd.MultiIndex):
                for t in batch:
                    if t in x.columns.get_level_values(0):
                        d = x[t].copy()
                        d.columns = [str(c).title() for c in d.columns]
                        d = d.dropna(subset=["Close"])
                        if len(d) >= 120:
                            out[t.replace(".JK", "")] = d
            else:
                t = batch[0].replace(".JK", "")
                d = x.copy()
                d.columns = [str(c).title() for c in d.columns]
                d = d.dropna(subset=["Close"])
                if len(d) >= 120:
                    out[t] = d
        except Exception:
            continue
        time.sleep(0.15)
    return out

# ============================================================
# TECHNICAL ENGINE
# ============================================================
def calc_features(d):
    d = d.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]

    d["MA20"] = c.rolling(20).mean()
    d["MA50"] = c.rolling(50).mean()
    d["MA150"] = c.rolling(150).mean()
    d["MA200"] = c.rolling(200).mean()
    d["VOL20"] = v.rolling(20).mean()
    d["VOL50"] = v.rolling(50).mean()

    d["ATR"] = pd.concat(
        [(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1
    ).max(axis=1).rolling(14).mean()
    d["ATR_PCT"] = d["ATR"] / c * 100

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - 100/(1+rs)

    d["RET5"] = c.pct_change(5)*100
    d["RET20"] = c.pct_change(20)*100
    d["RET60"] = c.pct_change(60)*100
    d["RET120"] = c.pct_change(120)*100
    d["VOL_RATIO"] = v / d["VOL20"].replace(0, np.nan)

    d["SLOPE50"] = d["MA50"].pct_change(20)*100
    d["SLOPE150"] = d["MA150"].pct_change(40)*100

    d["HIGH20"] = c.shift(1).rolling(20).max()
    d["HIGH60"] = c.shift(1).rolling(60).max()
    d["LOW20"] = c.shift(1).rolling(20).min()

    d["BREAK20"] = c > d["HIGH20"]
    d["BREAK60"] = c > d["HIGH60"]

    d["UPVOL"] = np.where(c.diff() > 0, v, 0)
    d["DNVOL"] = np.where(c.diff() < 0, v, 0)
    up = pd.Series(d["UPVOL"], index=d.index).rolling(20).sum()
    dn = pd.Series(d["DNVOL"], index=d.index).rolling(20).sum()
    d["UPDN"] = up / dn.replace(0, np.nan)

    # Base compression / contraction
    d["RANGE20"] = (d["HIGH20"] - d["LOW20"]) / c * 100
    d["RANGE60"] = (c.shift(1).rolling(60).max() - c.shift(1).rolling(60).min()) / c * 100
    d["ATR20"] = d["ATR_PCT"].rolling(20).mean()
    d["ATR_CONTRACT"] = d["ATR_PCT"] < d["ATR_PCT"].rolling(60).quantile(0.45)

    # Distribution: down day with meaningful volume
    d["DIST"] = (
        (c < c.shift(1)) &
        (d["VOL_RATIO"] >= 1.25)
    ).astype(int)
    d["DIST20"] = d["DIST"].rolling(20).sum()

    # Drawdown from 20d high
    d["DD20"] = c / c.rolling(20).max() - 1

    # Relative strength vs IHSG is added later.
    return d.dropna(subset=["MA50", "MA150", "RSI"])

def add_rs(d, ihsg):
    x = d.copy()
    b = ihsg["Close"].reindex(x.index).ffill()
    x["RS20"] = x["Close"].pct_change(20) - b.pct_change(20)
    x["RS60"] = x["Close"].pct_change(60) - b.pct_change(60)
    x["RS120"] = x["Close"].pct_change(120) - b.pct_change(120)
    x["RS_SLOPE"] = x["RS20"].rolling(10).mean()
    return x

def pct_rank(v, lo, hi):
    return float(np.clip((v-lo)/(hi-lo), 0, 1))

def score_row(d):
    r = d.iloc[-1]
    prev = d.iloc[-2]

    # 1) EARLY SCORE: designed to catch BASE -> IGNITION.
    base = 0
    base += 16 * (1 if r["ATR_CONTRACT"] else 0)
    base += 10 * pct_rank(-r["RANGE20"], -8, -2)
    base += 8 * pct_rank(r["UPDN"], 0.7, 1.6)
    base += 8 * (1 if r["RS_SLOPE"] > 0 else 0)
    base += 8 * (1 if r["SLOPE50"] > 0 else 0)

    # Breakout / acceptance
    breakout = 0
    breakout += 18 * (1 if r["BREAK20"] else 0)
    breakout += 10 * (1 if r["BREAK60"] else 0)
    breakout += 8 * pct_rank(r["VOL_RATIO"], 0.8, 2.5)
    breakout += 6 * (1 if r["Close"] > r["MA50"] else 0)

    # Trend / persistence
    trend = 0
    trend += 10 * (1 if r["MA20"] > r["MA50"] else 0)
    trend += 10 * (1 if r["MA50"] > r["MA150"] else 0)
    trend += 8 * (1 if r["SLOPE50"] > 1 else 0)
    trend += 7 * (1 if r["SLOPE150"] > 0 else 0)
    trend += 5 * (1 if r["Close"] > r["MA150"] else 0)

    # Relative strength
    rs = 0
    rs += 10 * pct_rank(r["RS20"], -0.02, 0.12)
    rs += 8 * pct_rank(r["RS60"], -0.02, 0.30)
    rs += 5 * (1 if r["RS_SLOPE"] > 0 else 0)

    # Momentum, but punish excessive extension.
    momentum = 0
    momentum += 7 * pct_rank(r["RET20"], 0, 35)
    momentum += 6 * pct_rank(r["RET60"], 0, 80)
    momentum += 5 * pct_rank(r["RSI"], 45, 72)

    extension = (r["Close"]/r["MA20"] - 1) * 100
    risk = 0
    risk += 8 * (1 if extension <= 18 else 0)
    risk += 6 * (1 if r["DD20"] > -0.12 else 0)
    risk += 6 * (1 if r["DIST20"] <= 5 else 0)
    risk += 5 * (1 if r["ATR_PCT"] < 10 else 0)

    early = np.clip(base + breakout + rs + 0.7*momentum + risk, 0, 100)
    persistence = np.mean(
        (d["Close"].tail(40) > d["MA50"].tail(40)) &
        (d["SLOPE50"].tail(40) > 0)
    ) * 100
    rally_quality = np.clip(
        0.40*trend + 0.25*rs + 0.20*breakout + 0.15*risk, 0, 100
    )

    # "Entry score" deliberately penalizes late/overextended moves.
    entry = early
    if extension > 18: entry -= min(25, extension-18)
    if extension > 30: entry -= 20
    if r["RSI"] > 78: entry -= 15
    if r["DD20"] < -0.15: entry -= 10
    if r["BREAK20"] and r["VOL_RATIO"] >= 1.5: entry += 6
    entry = float(np.clip(entry, 0, 100))

    if early >= 72 and persistence >= 65:
        phase = "CONFIRMED RALLY"
    elif early >= 64 and persistence >= 50:
        phase = "EARLY RALLY"
    elif early >= 54:
        phase = "IGNITION"
    elif early >= 45:
        phase = "BASE / WATCH"
    else:
        phase = "NO SIGNAL"

    if entry >= 72 and phase in ["IGNITION", "EARLY RALLY"]:
        decision = "ACCUMULATE GRADUALLY"
    elif entry >= 68 and phase == "CONFIRMED RALLY":
        decision = "HOLD / ADD ON PULLBACK"
    elif phase in ["EARLY RALLY", "CONFIRMED RALLY"]:
        decision = "WATCH — DO NOT CHASE"
    elif phase == "IGNITION":
        decision = "WATCH FOR BREAKOUT"
    else:
        decision = "WATCH"

    return {
        "price": r["Close"],
        "early_score": round(float(early), 1),
        "persistence": round(float(persistence), 1),
        "rally_quality": round(float(rally_quality), 1),
        "entry_score": round(float(entry), 1),
        "extension": round(float(extension), 1),
        "ret20": round(float(r["RET20"]), 1),
        "ret60": round(float(r["RET60"]), 1),
        "ret120": round(float(r["RET120"]), 1),
        "rsi": round(float(r["RSI"]), 1),
        "vol_ratio": round(float(r["VOL_RATIO"]), 2),
        "rs20": round(float(r["RS20"]*100), 1),
        "rs60": round(float(r["RS60"]*100), 1),
        "dist20": int(r["DIST20"]),
        "updn": round(float(r["UPDN"]), 2),
        "slope50": round(float(r["SLOPE50"]), 1),
        "phase": phase,
        "decision": decision,
    }

# ============================================================
# FUNDAMENTAL SNAPSHOT — only for finalists because yfinance info
# can be slow. It is evidence, not a guarantee.
# ============================================================
@st.cache_data(ttl=21600, show_spinner=False)
def fundamentals(ticker):
    try:
        t = yf.Ticker(ticker + ".JK")
        info = t.info
        return {
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        return {}

def fundamental_score(f):
    if not f:
        return 50
    s = 50
    if f.get("revenue_growth") is not None:
        s += np.clip(f["revenue_growth"]*30, -15, 15)
    if f.get("earnings_growth") is not None:
        s += np.clip(f["earnings_growth"]*30, -15, 15)
    if f.get("roe") is not None:
        s += np.clip(f["roe"]*20, -10, 10)
    if f.get("profit_margin") is not None:
        s += np.clip(f["profit_margin"]*15, -8, 8)
    if f.get("debt_to_equity") is not None and f["debt_to_equity"] > 250:
        s -= 12
    return float(np.clip(s, 0, 100))

# ============================================================
# DETAIL
# ============================================================
def chart(ticker, d):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price"
    ))
    for col in ["MA20","MA50","MA150","MA200"]:
        if col in d:
            fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, mode="lines"))
    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False,
        margin=dict(l=10,r=10,t=30,b=10),
        legend=dict(orientation="h")
    )
    return fig

def rally_history(d, horizon=180, target=0.30, stop=-0.20):
    closes = d["Close"].values
    rows = []
    for i in range(220, len(d)-horizon-1):
        sub = d.iloc[:i+1]
        s = score_row(sub)
        if s["entry_score"] < 68:
            continue
        future = closes[i+1:i+1+horizon]
        base = closes[i]
        hit_t = np.where(future >= base*(1+target))[0]
        hit_s = np.where(future <= base*(1+stop))[0]
        first_t = hit_t[0] if len(hit_t) else 99999
        first_s = hit_s[0] if len(hit_s) else 99999
        success = first_t < first_s
        rows.append({
            "date": d.index[i],
            "entry": base,
            "success": success,
            "days_to_target": int(first_t+1) if first_t < 99999 else np.nan,
            "mfe": float(future.max()/base-1),
            "mae": float(future.min()/base-1),
        })
    return pd.DataFrame(rows)

# ============================================================
# UI
# ============================================================
st.title("📈 BEI Rally Screener")
st.caption("BASE → IGNITION → EARLY RALLY → CONFIRMED RALLY. Research/decision-support only.")

with st.sidebar:
    st.header("Pengaturan")
    period = st.selectbox("Data historis", ["1y","2y","3y","5y"], index=1)
    max_scan = st.slider("Maksimum saham yang discan", 50, 900, 300, 50)
    min_price = st.number_input("Harga minimum", 1, 100000, 50)
    min_value = st.number_input("Perkiraan min. nilai transaksi harian (Rp)", 0, 100_000_000_000, 500_000_000, 100_000_000)
    show_fund = st.checkbox("Ambil fundamental untuk 50 kandidat teratas", True)
    run = st.button("🚀 SCAN SEKARANG", type="primary", use_container_width=True)

universe = get_universe()
st.info(f"Universe publik terdeteksi: **{len(universe)} saham**. Untuk scan penuh, naikkan maksimum saham.")

if run:
    tickers = universe["ticker"].head(max_scan).tolist()
    with st.spinner(f"Mengambil data harga {len(tickers)} saham..."):
        px = download_prices(tickers, period)
        ihsg_raw = yf.download("^JKSE", period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(ihsg_raw.columns, pd.MultiIndex):
            ihsg_raw.columns = ihsg_raw.columns.get_level_values(-1)
        ihsg_raw.columns = [str(c).title() for c in ihsg_raw.columns]
        ihsg_raw = ihsg_raw.dropna(subset=["Close"])

    results = []
    series = {}
    for ticker, raw in px.items():
        try:
            d = calc_features(raw)
            d = add_rs(d, ihsg_raw)
            if len(d) < 180:
                continue
            last = float(d["Close"].iloc[-1])
            avg_value = float((d["Close"]*d["Volume"]).tail(20).mean())
            if last < min_price or avg_value < min_value:
                continue
            row = score_row(d)
            row["ticker"] = ticker
            row["avg_value20"] = avg_value
            results.append(row)
            series[ticker] = d
        except Exception:
            continue

    res = pd.DataFrame(results)
    if res.empty:
        st.error("Tidak ada hasil. Coba turunkan filter harga/nilai transaksi.")
        st.stop()

    # Fundamental refinement for finalists.
    if show_fund:
        top = res.sort_values(["entry_score","early_score"], ascending=False).head(50)
        fund_rows = []
        progress = st.progress(0)
        for i, t in enumerate(top["ticker"]):
            f = fundamentals(t)
            fund_rows.append({"ticker":t, "fund_score":fundamental_score(f), **f})
            progress.progress((i+1)/len(top))
        ff = pd.DataFrame(fund_rows)
        res = res.merge(ff[["ticker","fund_score","pe","pb","roe","revenue_growth","earnings_growth","sector"]],
                        on="ticker", how="left")
    else:
        res["fund_score"] = np.nan

    res["total_score"] = (
        0.45*res["entry_score"] +
        0.25*res["rally_quality"] +
        0.15*res["persistence"] +
        0.15*res["fund_score"].fillna(50)
    ).round(1)

    res = res.sort_values(
        ["total_score","early_score","entry_score"], ascending=False
    ).reset_index(drop=True)

    st.session_state["res"] = res
    st.session_state["series"] = series

if "res" not in st.session_state:
    st.warning("Tekan **SCAN SEKARANG** untuk memulai.")
    st.stop()

res = st.session_state["res"]
series = st.session_state["series"]

# Top cards
c1,c2,c3,c4 = st.columns(4)
c1.metric("Early Rally", int((res["phase"]=="EARLY RALLY").sum()))
c2.metric("Confirmed Rally", int((res["phase"]=="CONFIRMED RALLY").sum()))
c3.metric("Ignition", int((res["phase"]=="IGNITION").sum()))
c4.metric("Kandidat skor tinggi", int((res["total_score"]>=70).sum()))

tabs = st.tabs(["🔥 Scanner", "🔎 Detail", "🧪 Backtest", "📋 CSV"])

with tabs[0]:
    cols = [
        "ticker","price","total_score","early_score","entry_score",
        "rally_quality","persistence","phase","decision",
        "ret20","ret60","ret120","rsi","vol_ratio","rs20","rs60",
        "extension","dist20","fund_score"
    ]
    show = res[cols].copy()
    show.columns = [
        "Ticker","Price","Total","Early","Entry","Quality","Persist.",
        "Phase","Decision","20D %","60D %","120D %","RSI","Vol×","RS20 vs IHSG %",
        "RS60 vs IHSG %","Ext. MA20 %","Dist.20D","Fund."
    ]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download hasil CSV",
        res.to_csv(index=False).encode(),
        "bei_rally_scan.csv",
        "text/csv"
    )

with tabs[1]:
    options = res["ticker"].tolist()
    selected = st.selectbox("Pilih saham", options)
    d = series[selected]
    r = res[res.ticker==selected].iloc[0]

    st.plotly_chart(chart(selected,d.tail(400)), use_container_width=True)

    a,b,c,dcol,e = st.columns(5)
    a.metric("Total", r["total_score"])
    b.metric("Early", r["early_score"])
    c.metric("Entry", r["entry_score"])
    dcol.metric("Persistence", f'{r["persistence"]}%')
    e.metric("Phase", r["phase"])

    st.write("### Bukti yang sedang terbaca")
    evidence = pd.DataFrame({
        "Indikator":[
            "Harga vs MA20","MA50 slope","Relative Strength 20D",
            "Relative Strength 60D","Volume vs rata-rata 20D",
            "Distribution days 20D","Extension vs MA20","Return 60D"
        ],
        "Nilai":[
            f'{r["price"]/d["MA20"].iloc[-1]-1:.1%}',
            f'{r["slope50"]:.1f}%',
            f'{r["rs20"]:.1f}%',
            f'{r["rs60"]:.1f}%',
            f'{r["vol_ratio"]:.2f}x',
            r["dist20"],
            f'{r["extension"]:.1f}%',
            f'{r["ret60"]:.1f}%'
        ]
    })
    st.dataframe(evidence, use_container_width=True, hide_index=True)

    f = fundamentals(selected)
    if f:
        st.write("### Fundamental snapshot")
        st.json(f)

    try:
        ins = yf.Ticker(selected+".JK").get_insider_transactions()
        if ins is not None and not ins.empty:
            st.write("### Insider transactions — current public snapshot")
            st.dataframe(ins.head(20), use_container_width=True)
    except Exception:
        pass

    st.warning(
        "Jangan samakan skor teknikal dengan probabilitas pasti. "
        "Data insider/holder gratis dapat tidak lengkap atau terlambat."
    )

with tabs[2]:
    st.write("### Historical signal test")
    st.caption("Ini bukan backtest strategi portofolio penuh; hanya menguji apakah sinyal Entry Score ≥ threshold mencapai target sebelum stop.")
    ticker_bt = st.selectbox("Ticker backtest", res["ticker"].tolist(), key="bt")
    threshold = st.slider("Minimum Entry Score", 55, 85, 68)
    horizon = st.slider("Horizon (hari bursa)", 60, 365, 180)
    target = st.slider("Target", 0.10, 1.00, 0.30, 0.05)
    stop = st.slider("Stop", -0.50, -0.05, -0.20, 0.05)

    dbt = series[ticker_bt]
    if st.button("Jalankan backtest"):
        rows = []
        closes = dbt["Close"].values
        for i in range(220, len(dbt)-horizon-1):
            sub = dbt.iloc[:i+1]
            try:
                s = score_row(sub)
            except:
                continue
            if s["entry_score"] < threshold:
                continue
            future = closes[i+1:i+1+horizon]
            base = closes[i]
            ht = np.where(future >= base*(1+target))[0]
            hs = np.where(future <= base*(1+stop))[0]
            ft = ht[0] if len(ht) else 99999
            fs = hs[0] if len(hs) else 99999
            rows.append({
                "Tanggal": dbt.index[i],
                "Entry": base,
                "Success": ft < fs,
                "MFE": future.max()/base-1,
                "MAE": future.min()/base-1
            })
        bt = pd.DataFrame(rows)
        if bt.empty:
            st.info("Belum ada cukup sinyal historis dengan parameter tersebut.")
        else:
            hit = bt["Success"].mean()
            x,y,z = st.columns(3)
            x.metric("Signal", len(bt))
            y.metric("Target before stop", f"{hit:.1%}")
            z.metric("Median MFE", f'{bt["MFE"].median():.1%}')
            st.dataframe(bt, use_container_width=True, hide_index=True)

with tabs[3]:
    st.download_button(
        "⬇️ Export semua hasil",
        res.to_csv(index=False).encode(),
        "BEI_Rally_Screener.csv",
        "text/csv"
    )
    st.write(res)

st.caption(
    "Sumber data gratis utama: Yahoo Finance untuk OHLCV/fundamental snapshot dan "
    "dataset publik ticker IDX. Untuk modal besar, validasi ulang dengan data resmi/berlisensi, "
    "corporate actions, disclosure kepemilikan, dan biaya/slippage nyata."
)
