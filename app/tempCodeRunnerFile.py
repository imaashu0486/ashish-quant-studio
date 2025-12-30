def predict(request, ticker_value, number_of_days):
    """
    Stocks -> regression (price), Index -> classification (direction).
    Saves CSVs. Theme-aware charts. Safer fundamentals (IPO, 52w). Shows best metric value.
    """
    colors = _plotly_theme_colors(request)

    if request.method == 'POST':
        ticker_value = request.POST.get('ticker', ticker_value)
    ticker_value = normalize_in_symbol(ticker_value)
    index_mode = is_index_symbol(ticker_value)
    horizon = 1

    # Intraday
    df_intra = yf_download_single(ticker_value, period='1d', interval='1m', auto_adjust=True, max_retries=2)
    if df_intra.empty:
        df_intra = yf_download_single(ticker_value, period='5d', interval='5m', auto_adjust=True, max_retries=2)

    fig_candle = go.Figure()
    if df_intra.empty or not all(k in df_intra.columns for k in ['Open','High','Low','Close']):
        fig_candle.add_annotation(text="No intraday data (market closed or symbol illiquid).",
                                  x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    else:
        fig_candle.add_trace(go.Candlestick(
            x=df_intra.index,
            open=df_intra['Open'], high=df_intra['High'],
            low=df_intra['Low'],  close=df_intra['Close'],
            increasing_line_color='lightgreen', decreasing_line_color='red',
            increasing_fillcolor='rgba(144,238,144,0.2)', decreasing_fillcolor='rgba(255,0,0,0.2)',
            name='market data'
        ))
    fig_candle.update_layout(
        title=f'{ticker_value} intraday',
        yaxis_title='Price', paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
        xaxis=dict(gridcolor=colors["grid"]), yaxis=dict(gridcolor=colors["grid"])
    )
    fig_candle.update_xaxes(rangeslider_visible=True)
    plot_div = plot(fig_candle, auto_open=False, output_type='div')

    # Daily history
    hist_period = '10y' if index_mode else '5y'
    hist = yf_download_single(ticker_value, period=hist_period, interval='1d', auto_adjust=True)
    if hist.empty or 'Close' not in hist:
        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
            'error_message': "No historical data available (Yahoo empty/ratelimit).",
            'fundamentals': {}, 'metrics': [], 'best_model': "—", 'best_value': "—",
            'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
            'number_of_days': horizon, 'task_mode': 'classification' if index_mode else 'regression',
            'next_day_up_prob': None,
        })

    keep = [c for c in ['Open','High','Low','Close','Volume'] if c in hist.columns]
    hist = hist[keep].dropna(subset=['Close']).copy()

    # Market for features
    mkt = yf_download_single('^NSEI', period=hist_period, interval='1d', auto_adjust=True)
    if mkt.empty or 'Close' not in mkt:
        mkt = pd.DataFrame(index=hist.index, data={'Close': hist['Close'].rolling(10).mean().ffill()})
    else:
        mkt = mkt[['Close']].reindex(hist.index).ffill().bfill()

    vol_ok = 'Volume' in hist and (hist['Volume'].dropna().sum() > 0) and (not index_mode)

    df_feat = build_features(hist, mkt, use_volume=vol_ok).sort_index().copy()
    close_series = hist['Close'].sort_index()
    # Do NOT reindex df_feat to close_series (it introduces NaNs)

    # ---------------- INDEX: classification ----------------
    if index_mode:
        lookahead = 2  # smoother than 1-day noise
        y_dir = (close_series.shift(-lookahead) > close_series).astype(int).rename('target_up')

        df_all = df_feat.join(y_dir, how='inner')
        df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

        feature_cols = [c for c in df_all.columns if c != 'target_up']
        if len(df_all) < 150 or not feature_cols:
            fund = get_fundamentals(ticker_value, hist_df=hist)
            return render(request, 'result.html', {
                'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
                'error_message': "Not enough clean history to train index classifier.",
                'fundamentals': fund, 'metrics': [], 'best_model': "—", 'best_value': "—",
                'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
                'number_of_days': horizon, 'task_mode': 'classification', 'next_day_up_prob': None,
            })

        X = df_all[feature_cols].values
        y = df_all['target_up'].values.astype(int)

        split_idx = int(len(df_all)*0.75)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        idx_test = df_all.index[split_idx:]

        # validation slice
        val_frac = 0.15
        val_start = max(5, int(len(X_train)*(1 - val_frac)))
        X_tr, X_val = X_train[:val_start], X_train[val_start:]
        y_tr, y_val = y_train[:val_start], y_train[val_start:]

        # impute + scale
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy='median')
        X_tr_imp = imp.fit_transform(X_tr)
        X_val_imp = imp.transform(X_val)
        X_test_imp = imp.transform(X_test)
        X_last_imp = imp.transform(df_all[feature_cols].iloc[[-1]].values)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_val_s = scaler.transform(X_val_imp)
        X_test_s = scaler.transform(X_test_imp)
        X_last_s = scaler.transform(X_last_imp)

        def best_threshold(proba, y_true):
            ths = np.linspace(0.35, 0.65, 31)
            accs = [accuracy_score(y_true, (proba >= t).astype(int)) for t in ths]
            i = int(np.argmax(accs))
            return float(ths[i]), float(accs[i]*100.0)

        models = {}
        models['LogReg'] = LogisticRegression(max_iter=200, solver='lbfgs').fit(X_tr_s, y_tr)
        models['RF_cls'] = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1).fit(X_tr_imp, y_tr)
        if _HAS_XGB:
            try:
                models['XGB_cls'] = XGBClassifier(
                    n_estimators=600, max_depth=5, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                    objective='binary:logistic', random_state=42, n_jobs=0
                ).fit(X_tr_imp, y_tr)
            except Exception as e:
                logger.info("XGB_cls skipped: %s", e)

        evals, preds_df, val_prob_store = {}, {}, {}

        for name, mdl in models.items():
            if name == 'LogReg':
                proba_val = mdl.predict_proba(X_val_s)[:,1]
                th, _ = best_threshold(proba_val, y_val)
                proba_test = mdl.predict_proba(X_test_s)[:,1]
                next_p = float(mdl.predict_proba(X_last_s)[0,1])
            else:
                proba_val = mdl.predict_proba(X_val_imp)[:,1]
                th, _ = best_threshold(proba_val, y_val)
                proba_test = mdl.predict_proba(X_test_imp)[:,1]
                next_p = float(mdl.predict_proba(X_last_imp)[0,1])

            val_prob_store[name] = proba_val
            y_hat = (proba_test >= th).astype(int)

            evals[name] = {
                "acc": float(accuracy_score(y_test, y_hat))*100.0,
                "prec": float(precision_score(y_test, y_hat, zero_division=0))*100.0,
                "rec": float(recall_score(y_test, y_hat, zero_division=0))*100.0,
                "next_up_prob": next_p,
                "threshold": th,
                "proba": proba_test
            }
            preds_df[name] = pd.Series(y_hat, index=idx_test)

        # Ensemble with its own threshold
        if evals:
            members = list(val_prob_store.keys())
            if members:
                avg_val = np.mean([val_prob_store[m] for m in members], axis=0)
                ens_th, _ = best_threshold(avg_val, y_val)
            else:
                ens_th = 0.5
            prob_mat = np.column_stack([v["proba"] for v in evals.values()])
            ens_prob = prob_mat.mean(axis=1)
            ens_hat  = (ens_prob >= ens_th).astype(int)
            evals['Ensemble'] = {
                "acc": float(accuracy_score(y_test, ens_hat))*100.0,
                "prec": float(precision_score(y_test, ens_hat, zero_division=0))*100.0,
                "rec": float(recall_score(y_test, ens_hat, zero_division=0))*100.0,
                "next_up_prob": float(np.mean([v["next_up_prob"] for v in evals.values()])),
                "threshold": float(ens_th),
                "proba": ens_prob
            }
            preds_df['Ensemble'] = pd.Series(ens_hat, index=idx_test)

        # Chart
        palette = {"LogReg":"#22c55e","RF_cls":"#f59e0b","XGB_cls":"#ef4444","Ensemble":"#60a5fa"}
        fig_pred_all = go.Figure()
        fig_pred_all.add_trace(go.Scatter(x=close_series.index, y=close_series.values,
                                          name='Actual', mode='lines', line=dict(width=2)))
        for name, series in preds_df.items():
            fig_pred_all.add_trace(go.Scatter(
                x=series.index, y=(series.values*(close_series.reindex(series.index).pct_change().std()*1000)).astype(float),
                name=name, mode='lines', line=dict(width=2, dash='dash', color=palette.get(name))
            ))
        fig_pred_all.update_layout(
            title=f"{ticker_value} — Direction Predictions (Test; lookahead={lookahead}d)",
            paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
            hovermode="x unified",
            xaxis=dict(type="date", rangeslider=dict(visible=True), gridcolor=colors["grid"]),
            yaxis=dict(title="(Scaled) Direction Overlay", gridcolor=colors["grid"])
        )
        plot_div_pred = plot(fig_pred_all, auto_open=False, output_type='div')

        metrics = [{
            "Model": name,
            "ACC": round(res["acc"], 2),
            "Precision": round(res["prec"], 2),
            "Recall": round(res["rec"], 2),
            "NextUpProb": round(res["next_up_prob"]*100.0, 2)
        } for name, res in evals.items()]

        metrics_sorted = sorted(metrics, key=lambda m: (-m["ACC"], -m["Precision"], -m["Recall"])) if metrics else []
        best_model = metrics_sorted[0]["Model"] if metrics_sorted else "—"

        # >>> Always show the TRUE best model; do NOT prefer Ensemble by default
        headline_key = best_model
        headline = next((m for m in metrics if m["Model"] == headline_key), None)

        accuracy_val = headline["ACC"] if headline else 0.0
        next_up_prob = headline["NextUpProb"] if headline else None
        best_value = accuracy_val  # shown on the card


        fund = get_fundamentals(ticker_value, hist_df=hist)

        # Save CSVs
        base_media = getattr(settings, "MEDIA_ROOT", os.path.join(getattr(settings,"BASE_DIR",os.getcwd()), "media"))
        out_dir = os.path.join(base_media, "training"); os.makedirs(out_dir, exist_ok=True)
        safe_t = ticker_value.replace(".", "_")
        feat_path = os.path.join(out_dir, f"{safe_t}_features.csv")
        test_path = os.path.join(out_dir, f"{safe_t}_test_actual_vs_preds.csv")
        df_all_out = df_all.copy(); df_all_out.insert(0, "Ticker", ticker_value); df_all_out.to_csv(feat_path, index=True)
        out_pred = pd.DataFrame(index=idx_test); out_pred["ActualDir"] = y_test
        for name, res in evals.items():
            out_pred[name+"_proba"] = res["proba"]
        out_pred.to_csv(test_path, index=True)

        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': plot_div_pred,
            'ticker_value': ticker_value,
            'accuracy': accuracy_val,
            'confidence': accuracy_val,
            'next_day_prediction': None,
            'next_day_up_prob': next_up_prob,
            'number_of_days': horizon,
            'fundamentals': fund,
            'metrics': metrics_sorted,
            'best_model': best_model,
            'best_value': round(best_value, 2),
            'csv_features_path': feat_path,
            'csv_test_path': test_path,
            'task_mode': 'classification',
            'error_message': None
        })

    # ---------------- STOCKS: regression ----------------
    df_all = df_feat.copy()
    target_series = close_series.shift(-1).rename('target_price')
    df_all = df_all.join(target_series, how='inner')
    df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    if len(df_all) < 100:
        fund = get_fundamentals(ticker_value, hist_df=hist)
        return render(request, 'result.html', {
            'plot_div': plot_div, 'plot_div_pred': None, 'ticker_value': ticker_value,
            'error_message': "Not enough clean history to train (need ~100+ rows after indicators).",
            'fundamentals': fund, 'metrics': [], 'best_model': "—", 'best_value': "—",
            'accuracy': 0.0, 'confidence': 0.0, 'next_day_prediction': None,
            'number_of_days': horizon, 'task_mode': 'regression', 'next_day_up_prob': None
        })

    feature_cols = [c for c in df_all.columns if c != 'target_price']
    X = df_all[feature_cols].values
    y = df_all['target_price'].values

    split_idx = int(len(df_all) * 0.75)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    idx_test = df_all.index[split_idx:]

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)
    X_last_imp  = imputer.transform(df_all[feature_cols].iloc[[-1]].values)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s  = scaler.transform(X_test_imp)
    feats_last_s = scaler.transform(X_last_imp)

    models = {}
    models['Ridge'] = Ridge(alpha=1.0).fit(X_train_s, y_train)
    models['RandomForest'] = RandomForestRegressor(n_estimators=700, random_state=42, n_jobs=-1).fit(X_train_imp, y_train)
    if _HAS_XGB:
        try:
            models['XGBoost'] = XGBRegressor(
                n_estimators=800, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective='reg:squarederror', random_state=42, n_jobs=0
            ).fit(X_train_imp, y_train)
        except Exception as e:
            logger.info("XGBRegressor skipped: %s", e)

    evals, preds_df = {}, {}
    cur_close_test = close_series.reindex(idx_test).values
    for name, mdl in models.items():
        if name == 'Ridge':
            y_hat = mdl.predict(X_test_s)
            next_pred = float(mdl.predict(feats_last_s)[0])
        else:
            y_hat = mdl.predict(X_test_imp)
            next_pred = float(mdl.predict(X_last_imp)[0])

        true_dir = np.sign(y_test - cur_close_test)
        pred_dir = np.sign(y_hat - cur_close_test)

        evals[name] = {
            "y_pred": y_hat,
            "r2": float(r2_score(y_test, y_hat)) if len(y_test) > 1 else 0.0,
            "mae_pct": float(np.mean(np.abs((y_hat - y_test) / y_test))) * 100.0,
            "dir_acc": float((pred_dir == true_dir).mean()) * 100.0,
            "next_price": next_pred
        }
        preds_df[name] = pd.Series(y_hat, index=idx_test)

    if preds_df:
        ens_series = pd.concat(preds_df, axis=1).mean(axis=1)
        ens_true_dir = np.sign(y_test - cur_close_test)
        ens_pred_dir = np.sign(ens_series.values - cur_close_test)
        evals['Ensemble'] = {
            "y_pred": ens_series.values,
            "r2": float(r2_score(y_test, ens_series.values)) if len(y_test) > 1 else 0.0,
            "mae_pct": float(np.mean(np.abs((ens_series.values - y_test) / y_test))) * 100.0,
            "dir_acc": float((ens_pred_dir == ens_true_dir).mean()) * 100.0,
            "next_price": float(np.mean([v["next_price"] for v in evals.values() if "next_price" in v]))
        }
        preds_df['Ensemble'] = ens_series

    palette = {"Ridge":"#22c55e","RandomForest":"#f59e0b","XGBoost":"#ef4444","Ensemble":"#60a5fa"}
    fig_pred_all = go.Figure()
    fig_pred_all.add_trace(go.Scatter(x=idx_test, y=y_test, name='Actual', mode='lines', line=dict(width=2)))
    for name, series in preds_df.items():
        fig_pred_all.add_trace(go.Scatter(
            x=series.index, y=series.values, name=name, mode='lines',
            line=dict(width=2, dash='dash', color=palette.get(name)),
            hovertemplate="%{x|%b %d, %Y}<br>₹ %{y:,.2f}<extra>"+name+"</extra>"
        ))
    fig_pred_all.update_layout(
        title=f"{ticker_value} — Actual vs Model Predictions (Test)",
        paper_bgcolor=colors["paper"], plot_bgcolor=colors["plot"], font_color=colors["font"],
        hovermode="x unified",
        xaxis=dict(type="date", rangeslider=dict(visible=True), gridcolor=colors["grid"]),
        yaxis=dict(title="Price", tickprefix="₹ ", tickformat=",.2f", gridcolor=colors["grid"])
    )
    plot_div_pred = plot(fig_pred_all, auto_open=False, output_type='div')

    metrics = [{
        "Model": name,
        "R2": round(res["r2"], 4),
        "MAE_ret": round(res["mae_pct"], 3),
        "Direction": round(res["dir_acc"], 2),
        "NextDayPrice": round(res["next_price"], 2)
    } for name, res in evals.items()]

    metrics_sorted = sorted(metrics, key=lambda m: (-m["R2"], -m["Direction"], m["MAE_ret"])) if metrics else []
    best_model = metrics_sorted[0]["Model"] if metrics_sorted else "—"

    # >>> Always show the TRUE best model; do NOT prefer Ensemble by default
    headline_key = best_model
    headline = next((m for m in metrics if m["Model"] == headline_key), None)

    accuracy_val = headline["R2"] if headline else 0.0
    next_price_val = headline["NextDayPrice"] if headline else float('nan')
    best_value = accuracy_val  # shown on the card

    fund = get_fundamentals(ticker_value, hist_df=hist)

    base_media = getattr(settings, "MEDIA_ROOT", os.path.join(getattr(settings,"BASE_DIR",os.getcwd()), "media"))
    out_dir = os.path.join(base_media, "training"); os.makedirs(out_dir, exist_ok=True)
    safe_t = ticker_value.replace(".", "_")
    feat_path = os.path.join(out_dir, f"{safe_t}_features.csv")
    test_path = os.path.join(out_dir, f"{safe_t}_test_actual_vs_preds.csv")
    df_all_out = df_all.copy(); df_all_out.insert(0, "Ticker", ticker_value); df_all_out.to_csv(feat_path, index=True)

    out_pred = pd.DataFrame(index=idx_test)
    out_pred["Actual"] = y_test
    for name, res in evals.items(): out_pred[name] = res["y_pred"]
    out_pred.to_csv(test_path, index=True)

    return render(request, 'result.html', {
        'plot_div': plot_div,
        'plot_div_pred': plot_div_pred,
        'ticker_value': ticker_value,
        'accuracy': accuracy_val,
        'confidence': accuracy_val,
        'next_day_prediction': next_price_val,
        'next_day_up_prob': None,
        'number_of_days': horizon,
        'fundamentals': fund,
        'metrics': metrics_sorted,
        'best_model': best_model,
        'best_value': round(best_value, 4),
        'csv_features_path': feat_path,
        'csv_test_path': test_path,
        'task_mode': 'regression',
        'error_message': None
    })

