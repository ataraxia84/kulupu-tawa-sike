"""
telegram_bot.py  —  Bot de Telegram para señales diarias de rotación sectorial
Envía un resumen con sectores READY, WATCH y cambios de cuadrante
"""

import os
import json
import requests
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SECTORS = {
    "XLK": "Technology",
    "XLF": "Financials", 
    "XLV": "Health Care",
    "XLY": "Consumer Discr.",
    "XLC": "Comm. Services",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}
BENCHMARK = "SPY"
RS_WINDOW = 60
LOOKBACK_DAYS = 365 * 2
MOM_WEEKS = 4

CACHE_FILE = ".score_cache.json"

# ─────────────────────────────────────────────────────────
# FUNCIONES DE ANÁLISIS
# ─────────────────────────────────────────────────────────
def safe(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 2)
    except: 
        return None

def classify_quad(rs, mom):
    if not rs or not mom: return "Lagging"
    if rs >= 100 and mom >= 100: return "Leading"
    if rs >= 100 and mom < 100: return "Weakening"
    if rs < 100 and mom < 100: return "Lagging"
    return "Improving"

def fetch_data():
    """Obtiene datos y calcula métricas"""
    tickers = [BENCHMARK] + list(SECTORS.keys())
    print(f"📥 Descargando datos...")
    
    start = (datetime.today() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].ffill().dropna(how="all")
        volume = raw["Volume"].ffill().dropna(how="all") if "Volume" in raw else None
    else:
        close = raw.ffill().dropna(how="all")
        volume = None
    
    bench = close[BENCHMARK]
    sectors_data = []
    
    for ticker, name in SECTORS.items():
        if ticker not in close.columns:
            continue
            
        c = close[ticker]
        
        # RS y Momentum
        raw_d = c / bench
        rs_d = 100 * raw_d / raw_d.rolling(RS_WINDOW).mean()
        mom_d = 100 + rs_d.pct_change(MOM_WEEKS * 5) * 100
        
        rs_now = safe(rs_d.iloc[-1])
        mom_now = safe(mom_d.iloc[-1])
        quad = classify_quad(rs_now, mom_now)
        
        # RSI
        rsi = calc_rsi(c)
        
        # Medias móviles
        sma20 = c.rolling(20).mean().iloc[-1]
        sma50 = c.rolling(50).mean().iloc[-1]
        price = c.iloc[-1]
        dist_sma20 = safe((price - sma20) / sma20 * 100) if sma20 else None
        
        # Volumen
        vol_ratio = None
        if volume is not None and ticker in volume.columns:
            v_series = volume[ticker].dropna()
            if len(v_series) >= 20:
                v5 = v_series.iloc[-5:].mean()
                v20 = v_series.iloc[-20:].mean()
                vol_ratio = safe(v5 / v20) if v20 else None
        
        # Retornos
        def ret(n):
            return safe((c.iloc[-1] / c.iloc[-n-1] - 1) * 100) if len(c) > n else None
        
        # Score
        score = calc_score(rs_now, mom_now, rsi, dist_sma20, vol_ratio)
        
        # Señal
        signal = get_signal(quad, rsi, dist_sma20)
        
        sectors_data.append({
            "ticker": ticker,
            "name": name,
            "price": safe(price),
            "rs": rs_now,
            "mom": mom_now,
            "quad": quad,
            "rsi": rsi,
            "dist_sma20": dist_sma20,
            "ret5": ret(5),
            "ret20": ret(20),
            "ret60": ret(60),
            "vol_ratio": vol_ratio,
            "score": score,
            "signal": signal,
        })
    
    return sectors_data

def calc_rsi(close, period=14):
    if len(close) < period + 1:
        return None
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + g / l.replace(0, np.nan))
    return safe(rsi.iloc[-1])

def calc_score(rs, mom, rsi, dist_sma20, vol_ratio):
    score = 0.0
    if rs: score += (rs - 100) * 2.5
    if mom: score += (mom - 100) * 1.5
    if rsi:
        if 40 <= rsi <= 65: score += 12
        elif rsi > 75: score -= 18
        elif rsi < 30: score -= 10
    if dist_sma20:
        if dist_sma20 > 20: score -= 25
        elif dist_sma20 > 12: score -= 12
        elif 2 <= dist_sma20 <= 8: score += 8
    if vol_ratio and vol_ratio > 1.5: score += 8
    return round(score, 1)

def get_signal(quad, rsi, dist_sma20):
    conditions = {
        "leading": quad == "Leading",
        "rsi_ok": rsi is not None and 40 <= rsi <= 65,
        "sma20_ok": dist_sma20 is not None and dist_sma20 <= 12
    }
    met = sum(conditions.values())
    
    if met == 3:
        return {"signal": "ready", "met": met, "total": 3}
    elif met >= 2:
        return {"signal": "watch", "met": met, "total": 3}
    else:
        return {"signal": "skip", "met": met, "total": 3}

def load_previous_cache():
    """Carga caché anterior para detectar cambios"""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_current_cache(sectors):
    """Guarda caché actual"""
    cache = {s["ticker"]: {"quad": s["quad"], "score": s["score"]} for s in sectors}
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except:
        pass

def detect_quadrant_changes(current_sectors, previous_cache):
    """Detecta sectores que cambiaron de cuadrante"""
    changes = []
    for s in current_sectors:
        ticker = s["ticker"]
        prev = previous_cache.get(ticker, {})
        prev_quad = prev.get("quad")
        
        if prev_quad and prev_quad != s["quad"]:
            changes.append({
                "ticker": ticker,
                "name": s["name"],
                "from": prev_quad,
                "to": s["quad"]
            })
    return changes

def format_message(sectors, changes):
    """Formatea el mensaje para Telegram"""
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Separar por señal
    ready = [s for s in sectors if s["signal"]["signal"] == "ready"]
    watch = [s for s in sectors if s["signal"]["signal"] == "watch"]
    
    # Top 5 por score
    top5 = sorted(sectors, key=lambda x: x["score"], reverse=True)[:5]
    
    # Construir mensaje
    lines = []
    lines.append(f"📊 *ROTACIÓN SECTORIAL*")
    lines.append(f"📅 {date}")
    lines.append("")
    
    # Cambios de cuadrante
    if changes:
        lines.append("🔄 *CAMBIO DE CUADRANTE*")
        for c in changes:
            arrow = "→"
            emoji = "🟢" if c["to"] == "Leading" else "🔴" if c["to"] == "Lagging" else "🟡"
            lines.append(f"{emoji} *{c['ticker']}* {c['name'][:15]} {c['from']} {arrow} {c['to']}")
        lines.append("")
    
    # READY
    if ready:
        lines.append("✅ *READY* (3/3 condiciones)")
        for s in ready[:5]:
            lines.append(format_sector_line(s))
        lines.append("")
    
    # WATCH
    if watch:
        lines.append("👀 *WATCH* (2/3 condiciones)")
        for s in watch[:7]:
            lines.append(format_sector_line(s))
        lines.append("")
    
    # TOP 5 por Score
    lines.append("🏆 *TOP 5 POR SCORE*")
    for i, s in enumerate(top5[:5], 1):
        arrow = "↑" if s["score"] > 0 else "↓"
        lines.append(f"{i}. *{s['ticker']}* {s['name'][:12]} | Score: {s['score']}{arrow} | {s['signal']['signal'].upper()}")
    
    lines.append("")
    lines.append("📈 *INDICADORES*")
    lines.append(f"RSI ideal: 40-65")
    lines.append(f"SMA20 ≤ 12%")
    lines.append(f"Cuadrante Leading = fortaleza")
    
    lines.append("")
    lines.append("🤖 *Bot de señales diarias*")
    
    return "\n".join(lines)

def format_sector_line(s):
    """Formatea una línea de sector para el mensaje"""
    signal = s["signal"]["signal"].upper()
    emoji = "🟢" if signal == "READY" else "🟡"
    
    rsi_ok = s["rsi"] is not None and 40 <= s["rsi"] <= 65
    rsi_emoji = "✅" if rsi_ok else "❌"
    
    sma_ok = s["dist_sma20"] is not None and s["dist_sma20"] <= 12
    sma_emoji = "✅" if sma_ok else "❌"
    
    return (f"{emoji} *{s['ticker']}* {s['name'][:12]} | "
            f"RS: {s['rs']:.1f} | Mom: {s['mom']:.1f} | "
            f"RSI: {s['rsi']:.0f}{rsi_emoji} | "
            f"SMA20: {s['dist_sma20']:.1f}%{sma_emoji} | "
            f"Score: {s['score']:.1f}")

def send_telegram_message(message):
    """Envía mensaje a Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Mensaje enviado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")
        return False

def main():
    """Función principal"""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "TU_TOKEN_AQUI":
        print("❌ Error: TELEGRAM_TOKEN no configurado")
        return
    
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "TU_CHAT_ID_AQUI":
        print("❌ Error: TELEGRAM_CHAT_ID no configurado")
        return
    
    print("🚀 Iniciando análisis de rotación sectorial...")
    
    try:
        # Obtener datos actuales
        current_sectors = fetch_data()
        
        # Cargar caché anterior
        previous_cache = load_previous_cache()
        
        # Detectar cambios
        changes = detect_quadrant_changes(current_sectors, previous_cache)
        
        # Guardar caché actual
        save_current_cache(current_sectors)
        
        # Formatear mensaje
        message = format_message(current_sectors, changes)
        
        # Enviar a Telegram
        success = send_telegram_message(message)
        
        if success:
            print("📊 Resumen:")
            ready = [s for s in current_sectors if s["signal"]["signal"] == "ready"]
            watch = [s for s in current_sectors if s["signal"]["signal"] == "watch"]
            print(f"   READY: {len(ready)} sectores")
            print(f"   WATCH: {len(watch)} sectores")
            print(f"   Cambios: {len(changes)}")
        
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        # Enviar mensaje de error
        error_msg = f"⚠️ Error en el análisis diario: {str(e)[:100]}"
        send_telegram_message(error_msg)

if __name__ == "__main__":
    main()
