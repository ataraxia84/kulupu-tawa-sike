"""
telegram_bot.py  —  Bot de Telegram para señales diarias de rotación sectorial (VERSIÓN MEJORADA)
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
ALTERNATE_BENCHMARKS = ["QQQ", "IWM"]  # Para referencia adicional

# Ventanas de tiempo (días hábiles)
RS_WINDOW = 60        # 3 meses
MOM_WINDOW = 20       # 4 semanas (20 días)
TREND_WINDOW = 200    # 10 meses (tendencia macro)
LOOKBACK_DAYS = 365 * 2  # 2 años de datos

# Umbrales para señales
SCORE_THRESHOLDS = {
    "ready": 70,   # Score mínimo para READY
    "watch": 50,   # Score mínimo para WATCH
}

# Pesos para el score compuesto (suman 1.0)
SCORE_WEIGHTS = {
    "rs": 0.35,      # Relative Strength
    "mom": 0.25,     # Momentum
    "rsi": 0.15,     # RSI
    "sma20": 0.15,   # Distancia a SMA20
    "volume": 0.10,  # Volumen relativo
}

CACHE_FILE = ".sector_cache.json"

# ─────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────
def safe(value):
    """Convierte a float seguro (sin NaN/Inf)"""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 2)
    except:
        return None

def classify_quadrant(rs, mom):
    """Clasifica el cuadrante según RS y Momentum"""
    if rs is None or mom is None:
        return "Lagging"
    if rs >= 100 and mom >= 100:
        return "Leading"
    if rs >= 100 and mom < 100:
        return "Weakening"
    if rs < 100 and mom < 100:
        return "Lagging"
    return "Improving"  # rs < 100 and mom >= 100

def calc_rsi(close, period=14):
    """Calcula RSI estándar"""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return safe(rsi.iloc[-1])

def calc_moving_averages(close, periods=[20, 50, 200]):
    """Calcula medias móviles y distancias porcentuales"""
    result = {}
    price = close.iloc[-1]
    
    for period in periods:
        ma = close.rolling(period).mean().iloc[-1]
        if ma and not np.isnan(ma):
            result[f"sma{period}"] = safe(ma)
            result[f"dist_sma{period}"] = safe((price / ma - 1) * 100)
        else:
            result[f"sma{period}"] = None
            result[f"dist_sma{period}"] = None
    
    return result

def calculate_percentile_rank(values, current_value):
    """Calcula el rango percentil de un valor (0-100)"""
    if not values or current_value is None:
        return 50
    sorted_vals = sorted([v for v in values if v is not None])
    if not sorted_vals:
        return 50
    count_less = sum(1 for v in sorted_vals if v < current_value)
    return (count_less / len(sorted_vals)) * 100

# ─────────────────────────────────────────────────────────
# FUNCIONES PRINCIPALES
# ─────────────────────────────────────────────────────────
def fetch_data():
    """Obtiene datos y calcula métricas (VERSIÓN CORREGIDA)"""
    tickers = [BENCHMARK] + list(SECTORS.keys())
    print(f"📥 Descargando datos de {len(tickers)} tickers...")
    
    start = (datetime.today() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    
    # Manejar estructura de datos
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].ffill().dropna(how="all")
        volume = raw["Volume"].ffill().dropna(how="all") if "Volume" in raw else None
    else:
        close = raw.ffill().dropna(how="all")
        volume = None
    
    bench = close[BENCHMARK].dropna()
    if len(bench) < RS_WINDOW + MOM_WINDOW:
        raise ValueError(f"No hay suficientes datos para {BENCHMARK}")
    
    sectors_data = []
    
    for ticker, name in SECTORS.items():
        if ticker not in close.columns:
            print(f"⚠️ Advertencia: {ticker} no encontrado")
            continue
        
        c = close[ticker].dropna()
        if len(c) < TREND_WINDOW:
            print(f"⚠️ {ticker}: datos insuficientes ({len(c)}/{TREND_WINDOW})")
            continue
        
        # ────────────── MÉTRICAS CORREGIDAS ──────────────
        
        # Relative Strength (Ratio vs Benchmark)
        ratio = c / bench.loc[c.index]
        
        # RS Rating: percentil de 60 días (0-100)
        rs_ma = ratio.rolling(RS_WINDOW).mean()
        rs_raw = 100 * ratio.iloc[-1] / rs_ma.iloc[-1] if rs_ma.iloc[-1] else 100
        rs = safe(rs_raw)
        
        # Momentum CORREGIDO: cambio porcentual en ventana de 20 días
        if len(ratio) > MOM_WINDOW:
            mom_raw = 100 * (ratio.iloc[-1] / ratio.iloc[-MOM_WINDOW-1] - 1) + 100
        else:
            mom_raw = 100
        mom = safe(mom_raw)
        
        # Cuadrante
        quadrant = classify_quadrant(rs, mom)
        
        # RSI
        rsi = calc_rsi(c)
        
        # Medias móviles
        ma = calc_moving_averages(c)
        
        # Volumen relativo (últimos 5 días vs últimos 20)
        vol_ratio = None
        if volume is not None and ticker in volume.columns:
            v = volume[ticker].dropna()
            if len(v) >= 20:
                v5 = v.iloc[-5:].mean()
                v20 = v.iloc[-20:].mean()
                vol_ratio = safe(v5 / v20) if v20 > 0 else None
        
        # Retornos (1, 4, 12 semanas)
        def returns(weeks):
            days = weeks * 5
            if len(c) > days:
                return safe((c.iloc[-1] / c.iloc[-days-1] - 1) * 100)
            return None
        
        # ────────────── SCORE COMPUESTO ──────────────
        # Este score se recalculará después con percentiles
        # Por ahora guardamos valores brutos
        sectors_data.append({
            "ticker": ticker,
            "name": name,
            "price": safe(c.iloc[-1]),
            "rs": rs,
            "mom": mom,
            "quadrant": quadrant,
            "rsi": rsi,
            "dist_sma20": ma.get("dist_sma20"),
            "dist_sma50": ma.get("dist_sma50"),
            "dist_sma200": ma.get("dist_sma200"),
            "vol_ratio": vol_ratio,
            "ret1w": returns(1),
            "ret4w": returns(4),
            "ret12w": returns(12),
            "raw_score": 0,  # Se calculará después
        })
    
    # Calcular scores basados en percentiles
    sectors_data = calculate_percentile_scores(sectors_data)
    
    # Calcular señales finales
    for s in sectors_data:
        s["signal"] = get_signal(s["score"], s["rsi"], s["dist_sma20"])
    
    return sectors_data

def calculate_percentile_scores(sectors_data):
    """Calcula scores usando percentiles (más robusto que pesos arbitrarios)"""
    if not sectors_data:
        return sectors_data
    
    # Extraer valores para cada métrica
    metric_values = {}
    for metric in ["rs", "mom", "rsi", "dist_sma20", "vol_ratio"]:
        values = [s[metric] for s in sectors_data if s[metric] is not None]
        metric_values[metric] = values
    
    # Calcular score para cada sector
    for s in sectors_data:
        score = 0
        total_weight = 0
        
        for metric, weight in SCORE_WEIGHTS.items():
            value = s.get(metric)
            if value is not None and metric_values[metric]:
                # Percentil rank (0-100)
                percentile = calculate_percentile_rank(metric_values[metric], value)
                
                # Ajustar métricas que son mejores cuando son más bajas
                if metric in ["dist_sma20"]:
                    # Para distancia SMA20: valores entre 2-12% son ideales
                    # Invertir ligeramente el percentil
                    if 2 <= value <= 12:
                        percentile = min(100, percentile + 20)
                    elif value > 20:
                        percentile = max(0, percentile - 30)
                
                score += percentile * weight
                total_weight += weight
        
        # Normalizar a 0-100
        s["raw_score"] = round(score, 1)
        s["score"] = round(score, 1)
    
    return sectors_data

def get_signal(score, rsi, dist_sma20):
    """Determina señal READY/WATCH/SKIP basada en score y condiciones"""
    conditions = {
        "score_ready": score >= SCORE_THRESHOLDS["ready"],
        "score_watch": score >= SCORE_THRESHOLDS["watch"],
        "rsi_good": rsi is not None and 40 <= rsi <= 65,
        "sma20_good": dist_sma20 is not None and dist_sma20 <= 12
    }
    
    if conditions["score_ready"] and conditions["rsi_good"] and conditions["sma20_good"]:
        return {"signal": "ready", "score": score, "met": 3, "total": 3}
    elif conditions["score_watch"] and (conditions["rsi_good"] or conditions["sma20_good"]):
        met = sum([conditions["score_watch"], conditions["rsi_good"], conditions["sma20_good"]])
        return {"signal": "watch", "score": score, "met": met, "total": 3}
    else:
        return {"signal": "skip", "score": score, "met": 0, "total": 3}

# ─────────────────────────────────────────────────────────
# CACHÉ PARA DETECTAR CAMBIOS
# ─────────────────────────────────────────────────────────
def load_previous_cache():
    """Carga caché anterior para detectar cambios de cuadrante"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Error cargando caché: {e}")
    return {}

def save_current_cache(sectors_data):
    """Guarda caché actual (cuadrantes y scores)"""
    cache = {
        s["ticker"]: {
            "quadrant": s["quadrant"],
            "score": s["score"],
            "date": datetime.now().isoformat()
        } for s in sectors_data
    }
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        print("💾 Caché guardado correctamente")
    except Exception as e:
        print(f"⚠️ Error guardando caché: {e}")

def detect_quadrant_changes(current_sectors, previous_cache):
    """Detecta sectores que cambiaron de cuadrante"""
    changes = []
    for s in current_sectors:
        ticker = s["ticker"]
        prev = previous_cache.get(ticker, {})
        prev_quadrant = prev.get("quadrant")
        
        if prev_quadrant and prev_quadrant != s["quadrant"]:
            changes.append({
                "ticker": ticker,
                "name": s["name"],
                "from": prev_quadrant,
                "to": s["quadrant"],
                "score_change": s["score"] - prev.get("score", 0)
            })
    return changes

# ─────────────────────────────────────────────────────────
# FORMATEO DE MENSAJES
# ─────────────────────────────────────────────────────────
def format_message(sectors_data, changes):
    """Formatea el mensaje para Telegram (versión mejorada)"""
    date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    # Separar por señal
    ready = [s for s in sectors_data if s["signal"]["signal"] == "ready"]
    watch = [s for s in sectors_data if s["signal"]["signal"] == "watch"]
    
    # Top 5 por score
    top5 = sorted(sectors_data, key=lambda x: x["score"], reverse=True)[:5]
    
    # Estadísticas del mercado
    avg_score = np.mean([s["score"] for s in sectors_data])
    leading_count = sum(1 for s in sectors_data if s["quadrant"] == "Leading")
    
    lines = []
    lines.append("📊 *ROTACIÓN SECTORIAL - SEÑALES DIARIAS*")
    lines.append(f"📅 {date}")
    lines.append("")
    
    # Resumen rápido
    lines.append(f"📈 *RESUMEN* | Leading: {leading_count}/{len(sectors_data)} | Score medio: {avg_score:.0f}")
    lines.append("")
    
    # Cambios de cuadrante (más importantes)
    if changes:
        lines.append("🔄 *CAMBIO DE CUADRANTE DETECTADO*")
        for c in changes[:5]:  # Limitar a 5
            arrow = "➡️"
            emoji = "🟢" if c["to"] == "Leading" else "🔴" if c["to"] == "Lagging" else "🟡"
            change_str = f"{c['score_change']:+.0f}" if c['score_change'] != 0 else ""
            lines.append(f"{emoji} *{c['ticker']}* {c['name'][:12]} {c['from']} {arrow} {c['to']} {change_str}")
        lines.append("")
    
    # READY (señal más fuerte)
    if ready:
        lines.append("✅ *READY - ALTA PROBABILIDAD*")
        for s in ready[:5]:
            lines.append(format_sector_detail(s, detailed=False))
        lines.append("")
    
    # WATCH (seguimiento)
    if watch:
        lines.append("👀 *WATCH - MONITOREO*")
        for s in watch[:7]:
            lines.append(format_sector_detail(s, detailed=False))
        lines.append("")
    
    # TOP 5 por Score
    lines.append("🏆 *TOP 5 - MEJOR PUNTUACIÓN*")
    for i, s in enumerate(top5, 1):
        trend = "📈" if s["score"] > 70 else "📊" if s["score"] > 50 else "📉"
        signal_icon = "✅" if s["signal"]["signal"] == "ready" else "👀" if s["signal"]["signal"] == "watch" else "⏸️"
        lines.append(f"{i}. {signal_icon} *{s['ticker']}* {s['name'][:10]} | Score: {s['score']:.0f} {trend}")
    lines.append("")
    
    # Leyenda y métricas clave
    lines.append("📖 *LEYENDA*")
    lines.append("• READY = 3/3 condiciones favorables")
    lines.append("• WATCH = 2/3 condiciones")
    lines.append("• Score > 70 = fortaleza técnica")
    lines.append("")
    
    # Condiciones ideales
    lines.append("🎯 *CONDICIONES IDEALES*")
    lines.append(f"RSI: 40-65")
    lines.append(f"SMA20: ≤ 12%")
    lines.append(f"Leading = Momentum + RS positivos")
    lines.append("")
    
    lines.append("🤖 *Bot de rotación sectorial | Datos: Yahoo Finance*")
    
    return "\n".join(lines)

def format_sector_detail(s, detailed=False):
    """Formatea una línea detallada de sector"""
    signal = s["signal"]["signal"].upper()
    emoji = "🟢" if signal == "READY" else "🟡" if signal == "WATCH" else "⚪"
    
    # Indicadores de condición
    rsi_ok = s["rsi"] is not None and 40 <= s["rsi"] <= 65
    rsi_icon = "✅" if rsi_ok else "❌"
    
    sma_ok = s["dist_sma20"] is not None and s["dist_sma20"] <= 12
    sma_icon = "✅" if sma_ok else "❌"
    
    # Cuadrante icono
    quad_icon = {
        "Leading": "🚀",
        "Improving": "📈",
        "Weakening": "📉",
        "Lagging": "⚠️"
    }.get(s["quadrant"], "⚪")
    
    if detailed:
        return (f"{emoji} *{s['ticker']}* {s['name'][:12]} | "
                f"Score: {s['score']:.0f} | {quad_icon} {s['quadrant']} | "
                f"RS: {s['rs']:.0f} | Mom: {s['mom']:.0f}")
    else:
        return (f"{emoji} *{s['ticker']}* {s['name'][:12]} | "
                f"Score: {s['score']:.0f} | "
                f"RSI: {s['rsi']:.0f}{rsi_icon} | "
                f"SMA20: {s['dist_sma20']:.1f}%{sma_icon}")

def send_telegram_message(message):
    """Envía mensaje a Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        print("✅ Mensaje enviado correctamente")
        return True
    except requests.exceptions.Timeout:
        print("❌ Timeout al enviar mensaje")
        return False
    except Exception as e:
        print(f"❌ Error al enviar mensaje: {e}")
        return False

# ─────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────
def main():
    """Función principal"""
    print("🚀 Iniciando análisis de rotación sectorial...")
    print(f"📊 Analizando {len(SECTORS)} sectores vs {BENCHMARK}")
    
    # Validar credenciales
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "TU_TOKEN_AQUI":
        print("❌ Error: TELEGRAM_TOKEN no configurado")
        return
    
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "TU_CHAT_ID_AQUI":
        print("❌ Error: TELEGRAM_CHAT_ID no configurado")
        return
    
    try:
        # Obtener datos actuales
        current_sectors = fetch_data()
        
        if not current_sectors:
            print("❌ No se obtuvieron datos de sectores")
            return
        
        print(f"📈 Datos obtenidos: {len(current_sectors)} sectores")
        
        # Cargar caché anterior y detectar cambios
        previous_cache = load_previous_cache()
        changes = detect_quadrant_changes(current_sectors, previous_cache)
        
        if changes:
            print(f"🔄 Detectados {len(changes)} cambios de cuadrante")
        
        # Guardar caché actual
        save_current_cache(current_sectors)
        
        # Formatear y enviar mensaje
        message = format_message(current_sectors, changes)
        
        # Opcional: guardar mensaje para debugging
        if os.environ.get("DEBUG"):
            with open("last_message.txt", "w") as f:
                f.write(message)
            print("💾 Mensaje guardado en last_message.txt")
        
        # Enviar a Telegram
        success = send_telegram_message(message)
        
        if success:
            ready_count = sum(1 for s in current_sectors if s["signal"]["signal"] == "ready")
            watch_count = sum(1 for s in current_sectors if s["signal"]["signal"] == "watch")
            print(f"📊 Resumen enviado: READY: {ready_count}, WATCH: {watch_count}, Cambios: {len(changes)}")
        
    except Exception as e:
        print(f"❌ Error en el análisis: {e}")
        import traceback
        traceback.print_exc()
        
        # Enviar mensaje de error
        error_msg = f"⚠️ Error en análisis sectorial: {str(e)[:150]}"
        send_telegram_message(error_msg)

if __name__ == "__main__":
    main()
