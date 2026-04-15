import pandas as pd
import numpy as np
import json, sys, os, time, warnings, itertools
from datetime import datetime
from joblib import Parallel, delayed
from smartmoneyconcepts import smc as SMC

warnings.filterwarnings("ignore")

CSV_PATH = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output"
N_CORES = 32
CAPITAL = 50_000.0
MULT_WDO = 10.0
COMISSAO = 5.0
SLIPPAGE = 2.0

GRID = {
    "swing_length": [3, 5, 7, 10],
    "rr_min": [1.5, 2.0, 2.5, 3.0],
    "atr_mult_sl": [0.3, 0.5, 0.7, 1.0],
    "poi_janela": [10, 20, 30, 50],
    "choch_janela": [10, 20, 30, 50],
    "close_break": [True, False],
}

MIN_TRADES = 20
MIN_PF = 0.8
MAX_DD = -50.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def carregar(path=CSV_PATH):
    print(f"[DATA] Carregando {path}...")
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna()
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] OK {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def preparar_smc(df, swing_length=5, close_break=True):
    swings = SMC.swing_highs_lows(df, swing_length=swing_length)
    estrutura = SMC.bos_choch(df, swings, close_break=close_break)
    fvg = SMC.fvg(df)
    ob = SMC.ob(df, swings)

    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    r = df.copy()
    r["swing_hl"] = swings["HighLow"].fillna(0)
    r["swing_lvl"] = swings["Level"].fillna(0)
    r["bos"] = estrutura["BOS"].fillna(0)
    r["choch"] = estrutura["CHOCH"].fillna(0)
    r["fvg"] = fvg["FVG"].fillna(0)
    r["fvg_top"] = fvg["Top"].fillna(np.nan)
    r["fvg_bot"] = fvg["Bottom"].fillna(np.nan)
    r["ob"] = ob["OB"].fillna(0)
    r["ob_top"] = ob["Top"].fillna(np.nan)
    r["ob_bot"] = ob["Bottom"].fillna(np.nan)
    r["atr"] = atr
    return r


def backtest(df_ind, rr_min=2.0, atr_mult_sl=0.5, poi_janela=20, choch_janela=20, capital=CAPITAL):
    trades = []
    equity = [capital]
    cap = capital
    em_pos = False
    trade = None

    fvgs_bull, fvgs_bear = [], []
    obs_bull, obs_bear = [], []
    ult_choch_bull = ult_choch_bear = -9999

    arr = df_ind.values
    cols = {c: i for i, c in enumerate(df_ind.columns)}

    def v(row, col):
        return row[cols[col]]

    for i in range(20, len(df_ind)):
        row = arr[i]
        close = v(row, "close")
        atr = v(row, "atr")
        if np.isnan(atr) or atr <= 0:
            atr = 5.0

        if em_pos and trade:
            d, sl, tp, en = trade["d"], trade["sl"], trade["tp"], trade["entry"]
            lo, hi = v(row, "low"), v(row, "high")
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)
            if hit_sl or hit_tp:
                saida = sl if hit_sl else tp
                pts = (saida - en) * d
                brl = pts * MULT_WDO - COMISSAO - SLIPPAGE * MULT_WDO * 0.5
                cap += brl
                equity.append(round(cap, 2))
                trade["saida"] = round(saida, 2)
                trade["pnl_pts"] = round(pts, 2)
                trade["pnl_brl"] = round(brl, 2)
                trade["resultado"] = "WIN" if hit_tp else "LOSS"
                trade["saida_dt"] = str(df_ind.index[i])[:16]
                trades.append(trade)
                em_pos = False
                trade = None
            continue

        choch = v(row, "choch")
        if choch == 1:
            ult_choch_bull = i
            fvgs_bull.clear()
            obs_bull.clear()
        elif choch == -1:
            ult_choch_bear = i
            fvgs_bear.clear()
            obs_bear.clear()

        fvg_v = v(row, "fvg")
        if fvg_v == 1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bull.append({"top": v(row, "fvg_top"), "bot": v(row, "fvg_bot"), "i": i, "tipo": "FVG"})
        elif fvg_v == -1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bear.append({"top": v(row, "fvg_top"), "bot": v(row, "fvg_bot"), "i": i, "tipo": "FVG"})

        ob_v = v(row, "ob")
        if ob_v == 1 and not np.isnan(v(row, "ob_top")):
            obs_bull.append({"top": v(row, "ob_top"), "bot": v(row, "ob_bot"), "i": i, "tipo": "OB"})
        elif ob_v == -1 and not np.isnan(v(row, "ob_top")):
            obs_bear.append({"top": v(row, "ob_top"), "bot": v(row, "ob_bot"), "i": i, "tipo": "OB"})

        fvgs_bull = [x for x in fvgs_bull if i - x["i"] <= poi_janela]
        fvgs_bear = [x for x in fvgs_bear if i - x["i"] <= poi_janela]
        obs_bull = [x for x in obs_bull if i - x["i"] <= poi_janela]
        obs_bear = [x for x in obs_bear if i - x["i"] <= poi_janela]

        sinal = poi = None

        if (i - ult_choch_bull) <= choch_janela:
            for p in reversed(fvgs_bull + obs_bull):
                if p["bot"] <= close <= p["top"]:
                    sinal = 1
                    poi = p
                    break

        if sinal is None and (i - ult_choch_bear) <= choch_janela:
            for p in reversed(fvgs_bear + obs_bear):
                if p["bot"] <= close <= p["top"]:
                    sinal = -1
                    poi = p
                    break

        if sinal is None or poi is None:
            continue

        slip = SLIPPAGE * 0.5
        if sinal == 1:
            entry = close + slip
            sl = poi["bot"] - atr * atr_mult_sl
        else:
            entry = close - slip
            sl = poi["top"] + atr * atr_mult_sl

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        tp = entry + sinal * risk * rr_min
        rr_r = abs(tp - entry) / risk

        if rr_r < rr_min * 0.95:
            continue
        if risk * MULT_WDO / cap > 0.05:
            continue

        em_pos = True
        trade = {
            "entry_dt": str(df_ind.index[i])[:16],
            "d": sinal,
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "rr": round(rr_r, 2),
            "poi_tipo": poi["tipo"],
            "capital_pre": round(cap, 2),
        }

    if em_pos and trade:
        last = arr[-1][cols["close"]]
        pts = (last - trade["entry"]) * trade["d"]
        brl = pts * MULT_WDO - COMISSAO
        cap += brl
        trade.update(
            {
                "saida": last,
                "pnl_pts": round(pts, 2),
                "pnl_brl": round(brl, 2),
                "resultado": "ABERTO",
                "saida_dt": str(df_ind.index[-1])[:16],
            }
        )
        trades.append(trade)
        equity.append(round(cap, 2))

    return trades, equity


def metricas(trades, equity, capital=CAPITAL):
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 5:
        return None

    df_t = pd.DataFrame(fechados)
    wins = df_t[df_t["resultado"] == "WIN"]
    loses = df_t[df_t["resultado"] == "LOSS"]
    n = len(df_t)
    wr = len(wins) / n * 100
    avg_w = wins["pnl_brl"].mean() if len(wins) else 0
    avg_l = loses["pnl_brl"].mean() if len(loses) else -1
    pf = abs(wins["pnl_brl"].sum() / loses["pnl_brl"].sum()) if loses["pnl_brl"].sum() != 0 else 9999
    pnl = df_t["pnl_brl"].sum()

    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    mdd = float(dd.min())

    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    neg = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else 0

    return {
        "total_trades": n,
        "wins": int(len(wins)),
        "losses": int(len(loses)),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "avg_win_brl": round(avg_w, 2),
        "avg_loss_brl": round(avg_l, 2),
        "expectancy_brl": round((wr / 100 * avg_w) + ((1 - wr / 100) * avg_l), 2),
        "total_pnl_brl": round(pnl, 2),
        "retorno_pct": round(pnl / capital * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final": round(capital + pnl, 2),
        "trades_fvg": int((df_t["poi_tipo"] == "FVG").sum()),
        "trades_ob": int((df_t["poi_tipo"] == "OB").sum()),
    }


def worker(params, df_json):
    sw, rr, am, pj, cj, cb = params
    try:
        df = pd.read_json(df_json)
        df.index = pd.to_datetime(df.index)
        df_ind = preparar_smc(df, swing_length=sw, close_break=cb)
        trades, equity = backtest(
            df_ind,
            rr_min=rr,
            atr_mult_sl=am,
            poi_janela=pj,
            choch_janela=cj,
        )
        m = metricas(trades, equity)
        if not m:
            return None
        if m["total_trades"] < MIN_TRADES:
            return None
        if m["profit_factor"] < MIN_PF:
            return None
        if m["max_drawdown_pct"] < MAX_DD:
            return None

        pf = min(m["profit_factor"], 10) / 10
        sh = min(max(m["sharpe_ratio"], 0), 8) / 8
        so = min(max(m["sortino_ratio"], 0), 10) / 10
        wr = m["win_rate"] / 100
        tr = min(m["total_trades"], 300) / 300

        score = pf * 0.35 + sh * 0.25 + so * 0.15 + wr * 0.15 + tr * 0.10

        return {
            "swing_length": sw,
            "rr_min": rr,
            "atr_mult_sl": am,
            "poi_janela": pj,
            "choch_janela": cj,
            "close_break": cb,
            "score": round(score, 6),
            **m,
        }
    except Exception:
        return None


def grid_search(df, mini=False):
    g = GRID
    combos = list(
        itertools.product(
            g["swing_length"],
            g["rr_min"],
            g["atr_mult_sl"],
            g["poi_janela"],
            g["choch_janela"],
            g["close_break"],
        )
    )

    if mini:
        combos = [(5, 2.0, 0.5, 20, 20, True)]
        print("\n[GRID] Modo MINI - 1 combo de validacao")
    else:
        print("\n[GRID] Grid Search com biblioteca SMC oficial")
        print(f"       Combinacoes: {len(combos):,}")
        print(f"       Cores: {N_CORES}")
        print(f"       Filtros: trades>={MIN_TRADES} | PF>={MIN_PF} | DD>={MAX_DD}%")

    split = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_json = df_ins.to_json(date_format="iso")

    t0 = time.time()

    if mini or len(combos) == 1:
        resultados = [r for r in [worker(c, df_json) for c in combos] if r]
    else:
        print(f"\n[GRID] Iniciando joblib Loky ({N_CORES} workers)...")
        resultados = Parallel(n_jobs=N_CORES, backend="loky", verbose=5)(
            delayed(worker)(c, df_json) for c in combos
        )
        resultados = [r for r in resultados if r]

    elapsed = time.time() - t0
    resultados.sort(key=lambda x: -x["score"])

    print(f"\n[GRID] OK {elapsed:.1f}s | {len(resultados)} validos de {len(combos):,}")

    if resultados:
        exibir_top(resultados)

    return {
        "melhor": resultados[0] if resultados else None,
        "top20": resultados[:20],
        "total_combos": len(combos),
        "validos": len(resultados),
        "elapsed_s": round(elapsed, 1),
    }


def exibir_top(resultados, n=15):
    print(f"\n{'=' * 76}")
    print(f"  TOP {min(n, len(resultados))} CONFIGURACOES")
    print(f"{'=' * 76}")
    print(
        f"  {'#':>2} {'SW':>3} {'RR':>4} {'ATR':>4} {'POI':>4} {'CHoCH':>6} {'CB':>5} "
        f"{'PF':>6} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'DD%':>6} {'Score':>7}"
    )
    print(f"  {'-' * 74}")
    for i, r in enumerate(resultados[:n], 1):
        star = "*" if i == 1 else " "
        print(
            f"  {star}{i:>2} {r['swing_length']:>3} {r['rr_min']:>4} {r['atr_mult_sl']:>4} "
            f"{r['poi_janela']:>4} {r['choch_janela']:>6} {str(r['close_break'])[0]:>5} "
            f"{r['profit_factor']:>6.3f} {r['sharpe_ratio']:>7.3f} "
            f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
            f"{r['max_drawdown_pct']:>6.1f} {r['score']:>7.4f}"
        )
    print(f"{'=' * 76}")


def walk_forward(df, config, n_splits=5):
    print(f"\n[WF] Walk-Forward: {n_splits} splits")
    resultados = []
    step = len(df) // n_splits

    for i in range(n_splits - 1):
        ini = i * step
        fim = (i + 2) * step
        split = ini + int((fim - ini) * 0.7)
        df_tr = df.iloc[ini:split]
        df_te = df.iloc[split:fim]

        if len(df_tr) < 200 or len(df_te) < 50:
            continue

        d0 = df_tr.index[0].strftime("%Y-%m-%d")
        d1 = df_tr.index[-1].strftime("%Y-%m-%d")
        d2 = df_te.index[0].strftime("%Y-%m-%d")
        d3 = df_te.index[-1].strftime("%Y-%m-%d")

        try:
            sw = config["swing_length"]
            cb = config["close_break"]
            df_tr_ind = preparar_smc(df_tr, swing_length=sw, close_break=cb)
            df_te_ind = preparar_smc(df_te, swing_length=sw, close_break=cb)

            bt_params = {k: v for k, v in config.items() if k in ["rr_min", "atr_mult_sl", "poi_janela", "choch_janela"]}

            tr_t, tr_e = backtest(df_tr_ind, **bt_params)
            te_t, te_e = backtest(df_te_ind, **bt_params)
            m_tr = metricas(tr_t, tr_e) or {}
            m_te = metricas(te_t, te_e) or {}

            print(f"\n  Split {i + 1}: Train [{d0}->{d1}] | Test [{d2}->{d3}]")
            if m_tr:
                print(
                    f"    TRAIN -> WR:{m_tr.get('win_rate', 0)}% | PF:{m_tr.get('profit_factor', 0)} | "
                    f"Trades:{m_tr.get('total_trades', 0)} | PnL:R${m_tr.get('total_pnl_brl', 0):,.0f}"
                )
            if m_te:
                print(
                    f"    TEST  -> WR:{m_te.get('win_rate', 0)}% | PF:{m_te.get('profit_factor', 0)} | "
                    f"Trades:{m_te.get('total_trades', 0)} | PnL:R${m_te.get('total_pnl_brl', 0):,.0f}"
                )

            resultados.append(
                {
                    "split": i + 1,
                    "train": m_tr,
                    "test": m_te,
                    "train_start": d0,
                    "train_end": d1,
                    "test_start": d2,
                    "test_end": d3,
                }
            )
        except Exception as e:
            print(f"    Split {i + 1} erro: {e}")

    lucrativos = sum(1 for r in resultados if r["test"].get("total_pnl_brl", 0) > 0)
    print(f"\n[WF] OK {lucrativos}/{len(resultados)} splits out-of-sample lucrativos")
    return resultados


def monte_carlo(trades, n_sim=1000, capital=CAPITAL):
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 10:
        return {}
    print(f"\n[MC] Monte Carlo: {n_sim:,} simulacoes...")
    pnls = np.array([t["pnl_brl"] for t in fechados])
    np.random.seed(42)
    rets, dds, ruinas = [], [], 0
    for _ in range(n_sim):
        seq = np.random.choice(pnls, size=len(pnls), replace=True)
        eq = np.insert(capital + np.cumsum(seq), 0, capital)
        pk = np.maximum.accumulate(eq)
        dd = ((eq - pk) / pk * 100).min()
        r = (eq[-1] - capital) / capital * 100
        rets.append(r)
        dds.append(dd)
        if eq[-1] < capital * 0.5:
            ruinas += 1
    rf, md = np.array(rets), np.array(dds)
    res = {
        "n_simulacoes": n_sim,
        "prob_lucro_pct": round(float((rf > 0).mean() * 100), 1),
        "retorno_mediana": round(float(np.median(rf)), 2),
        "retorno_p10": round(float(np.percentile(rf, 10)), 2),
        "retorno_p90": round(float(np.percentile(rf, 90)), 2),
        "dd_mediano": round(float(np.median(md)), 2),
        "dd_pior": round(float(md.min()), 2),
        "prob_ruina_pct": round(float(ruinas / n_sim * 100), 2),
    }
    print(
        f"[MC] OK Prob. lucro: {res['prob_lucro_pct']}% | DD mediano: {res['dd_mediano']}% | Ruina: {res['prob_ruina_pct']}%"
    )
    return res


def relatorio(m, mc=None, titulo="RESULTADO"):
    if not m:
        return
    sep = "=" * 58

    def L(lb, vl):
        print(f"  {lb:<30} {str(vl):>24}")

    print(f"\n{sep}")
    print(f"  SMC WDO - {titulo}")
    print(sep)
    L("Total Trades", m["total_trades"])
    L("Wins / Losses", f"{m['wins']} W  /  {m['losses']} L")
    L("Win Rate", f"{m['win_rate']}%")
    L("Profit Factor", m["profit_factor"])
    L("Sharpe Ratio", m["sharpe_ratio"])
    L("Sortino Ratio", m["sortino_ratio"])
    L("Expectancy", f"R$ {m['expectancy_brl']:,.2f}")
    L("Total PnL", f"R$ {m['total_pnl_brl']:,.2f}")
    L("Retorno %", f"{m['retorno_pct']}%")
    L("Max Drawdown", f"{m['max_drawdown_pct']}%")
    L("Capital Final", f"R$ {m['capital_final']:,.2f}")
    L("Trades FVG / OB", f"{m['trades_fvg']} / {m['trades_ob']}")
    if mc:
        print(f"  {'-' * 54}")
        L("MC Prob. Lucro", f"{mc['prob_lucro_pct']}%")
        L("MC Retorno Med.", f"{mc['retorno_mediana']}%")
        L("MC P10/P90", f"{mc['retorno_p10']}% / {mc['retorno_p90']}%")
        L("MC DD Mediano", f"{mc['dd_mediano']}%")
        L("MC Risco Ruina", f"{mc['prob_ruina_pct']}%")
    print(sep)


def main():
    MINI = "--mini" in sys.argv

    print("=" * 68)
    print("  SMC OPTIMIZER v4 -- BIBLIOTECA OFICIAL smartmoneyconcepts")
    print("  Sem lookahead bias · CHoCH+FVG+OB reais · WDO B3")
    print("=" * 68)

    df = carregar()
    split = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"  In-sample : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}")
    print(f"  Out-sample: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}")

    if MINI:
        print("\n[MINI] Validando 1 combo...")
        df_ind = preparar_smc(df_ins, swing_length=5, close_break=True)
        t, e = backtest(df_ind, rr_min=2.0, atr_mult_sl=0.5, poi_janela=20, choch_janela=20)
        m = metricas(t, e)
        if m:
            print(f"\nOK {m['total_trades']} trades gerados!")
            relatorio(m, titulo="MINI TEST")
        else:
            print("AVISO: Poucos trades. Dados insuficientes ou parametros muito restritivos.")
        return

    grid = grid_search(df, mini=False)

    if not grid["melhor"]:
        print("\n[ERRO] Nenhuma configuracao valida encontrada.")
        return

    melhor = grid["melhor"]
    CONFIG = {
        k: melhor[k]
        for k in ["swing_length", "rr_min", "atr_mult_sl", "poi_janela", "choch_janela", "close_break"]
    }

    print(
        f"\nMELHOR: SW={melhor['swing_length']} RR={melhor['rr_min']} "
        f"ATR={melhor['atr_mult_sl']} POI={melhor['poi_janela']} "
        f"CHoCH={melhor['choch_janela']} CB={melhor['close_break']}"
    )

    print("\n[OOS] Backtest Out-of-Sample...")
    df_oos_ind = preparar_smc(df_oos, swing_length=CONFIG["swing_length"], close_break=CONFIG["close_break"])
    bt_params = {k: v for k, v in CONFIG.items() if k in ["rr_min", "atr_mult_sl", "poi_janela", "choch_janela"]}
    t_oos, e_oos = backtest(df_oos_ind, **bt_params)
    m_oos = metricas(t_oos, e_oos)
    relatorio(m_oos, titulo="OUT-OF-SAMPLE")

    print("\n[FULL] Backtest dataset completo...")
    df_full_ind = preparar_smc(df, swing_length=CONFIG["swing_length"], close_break=CONFIG["close_break"])
    t_full, e_full = backtest(df_full_ind, **bt_params)
    m_full = metricas(t_full, e_full)

    wf = walk_forward(df, CONFIG, n_splits=5)
    mc = monte_carlo(t_full, n_sim=1000)
    relatorio(m_full, mc, titulo="COMPLETO + MONTE CARLO")

    out = {
        "config_melhor": CONFIG,
        "metricas_full": m_full,
        "metricas_oos": m_oos,
        "walk_forward": wf,
        "monte_carlo": mc,
        "grid_top20": grid["top20"],
        "trades": t_full,
        "equity_curve": e_full,
        "gerado_em": datetime.now().isoformat(),
    }
    path = f"{OUTPUT_DIR}/resultado_v4.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[OK] Salvo em {path}")
    print(
        f"CONCLUIDO! Melhor PF: {melhor['profit_factor']} | Score: {melhor['score']} | Trades: {melhor['total_trades']}"
    )


if __name__ == "__main__":
    main()