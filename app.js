// TradingView Backtesting Rankings - Dashboard Logic
document.addEventListener("DOMContentLoaded", () => {
    
    // Application State
    let db = null;
    let selectedStrategyId = null;
    let currentTicker = "SPY"; // Active ticker for detail visualization
    let activeFilter = "GLOBAL"; // Active sorting filter for sidebar
    let activeTab = "tab-chart";
    let equityChart = null; // Chart.js instance

    // Dom Elements
    const elTickerFilter = document.getElementById("ticker-filter");
    const elStrategyList = document.getElementById("strategy-list");
    const elStrategyCount = document.getElementById("strategy-count");
    const elNoSelectionState = document.getElementById("no-selection-state");
    const elDashboardState = document.getElementById("dashboard-state");
    
    // Header Elements
    const elStratRank = document.getElementById("strat-rank");
    const elStratName = document.getElementById("strat-name");
    const elStratDesc = document.getElementById("strat-desc");
    const elStratTags = document.getElementById("strat-tags");
    
    // Agg Metrics Elements
    const elAggReturn = document.getElementById("agg-return");
    const elAggReturnSub = document.getElementById("agg-return-sub");
    const elAggSharpe = document.getElementById("agg-sharpe");
    const elAggSharpeStatus = document.getElementById("agg-sharpe-status");
    const elAggDrawdown = document.getElementById("agg-drawdown");
    const elAggBeaten = document.getElementById("agg-beaten");
    
    // Ticker Details Elements
    const elCurrentTickerName = document.getElementById("current-ticker-name");
    const elTkReturn = document.getElementById("tk-return");
    const elTkBhReturn = document.getElementById("tk-bh-return");
    const elTkDrawdown = document.getElementById("tk-drawdown");
    const elTkBhDrawdown = document.getElementById("tk-bh-drawdown");
    const elTkSharpe = document.getElementById("tk-sharpe");
    const elTkWinrate = document.getElementById("tk-winrate");
    const elTkPf = document.getElementById("tk-pf");
    const elTkTrades = document.getElementById("tk-trades");
    const elTkDuration = document.getElementById("tk-duration");
    const elOutperformanceMargin = document.getElementById("outperformance-margin");
    const elBenchCompBanner = document.getElementById("bench-comp-banner");
    const elTickerChips = document.getElementById("ticker-chips");
    
    // Tables & Tabs Elements
    const elCompareTableBody = document.getElementById("compare-table-body");
    const elTradesTableBody = document.getElementById("trades-table-body");
    const elTradesTickerTitle = document.getElementById("trades-ticker-title");
    
    // Pine Script Elements
    const elPineScriptCode = document.getElementById("pinescript-code");
    const elBtnCopyCode = document.getElementById("btn-copy-code");
    const elToastMessage = document.getElementById("toast-message");

    // -------------------------------------------------------------------------
    // 1. Data Initialization & Dynamic Recalculation
    // -------------------------------------------------------------------------
    const elRecalcBtn = document.getElementById("recalc-btn");
    const elCommissionInput = document.getElementById("commission-input");

    if (elRecalcBtn) {
        elRecalcBtn.addEventListener("click", () => {
            const comm = elCommissionInput.value || 0.4;
            loadDatabase(comm);
        });
    }

    async function loadDatabase(commission = null) {
        try {
            if (commission !== null) {
                elStrategyList.innerHTML = `<li class="loading-item"><i class="fa-solid fa-spinner fa-spin"></i> Recalculando...</li>`;
            }
            
            const url = commission !== null ? `/api/recalculate?commission=${commission}` : "results.json";
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error("No se pudo cargar results.json. Asegúrate de iniciar un servidor local.");
            }
            db = await response.json();
            
            
            // Set up UI
            if (commission === null) {
                initUI();
            } else {
                // Just refresh list and current view
                renderSidebarRanking();
                if (selectedStrategyId) {
                    selectStrategy(selectedStrategyId);
                }
            }
        } catch (error) {
            console.error(error);
            elStrategyList.innerHTML = `<li class="loading-item text-danger" style="padding:20px;">
                <i class="fa-solid fa-triangle-exclamation" style="font-size:24px;margin-bottom:10px;display:block;"></i>
                <strong>Error al cargar resultados:</strong><br>
                ${error.message}<br><br>
                <small>Ejecuta <code>uv run python -m http.server 8000</code> en el directorio del proyecto y abre la URL en tu navegador.</small>
            </li>`;
        }
    }

    function initUI() {
        elStrategyCount.textContent = db.ranking.length;
        
        // Populate sidebar rankings for the first time
        renderSidebarRanking();
        
        // Select the first strategy by default if available
        if (db.ranking.length > 0) {
            selectStrategy(db.ranking[0].strategy_id);
        } else {
            showNoSelectionState(true);
        }
    }

    // -------------------------------------------------------------------------
    // 2. Sidebar Rendering & Sorting
    // -------------------------------------------------------------------------
    function renderSidebarRanking() {
        elStrategyList.innerHTML = "";
        
        // Sort ranking list depending on the activeFilter (GLOBAL or specific ticker)
        let sortedRanking = [...db.ranking];
        
        if (activeFilter !== "GLOBAL") {
            // Sort by return on specific ticker descending
            sortedRanking.sort((a, b) => {
                const returnA = a.ticker_results[activeFilter]?.metrics.total_return || 0;
                const returnB = b.ticker_results[activeFilter]?.metrics.total_return || 0;
                return returnB - returnA;
            });
        } else {
            // Sort by average return descending
            sortedRanking.sort((a, b) => b.aggregate_metrics.avg_return - a.aggregate_metrics.avg_return);
        }

        sortedRanking.forEach((strat, idx) => {
            const li = document.createElement("li");
            li.className = `strategy-item ${strat.strategy_id === selectedStrategyId ? 'active' : ''}`;
            li.dataset.id = strat.strategy_id;
            
            // Calculate label metric
            let metricLabel = "";
            let metricValue = "";
            
            if (activeFilter === "GLOBAL") {
                metricLabel = "Avg Return";
                metricValue = `${strat.aggregate_metrics.avg_return.toFixed(1)}%`;
            } else {
                const tr = strat.ticker_results[activeFilter]?.metrics.total_return || 0;
                metricLabel = `${activeFilter} Return`;
                metricValue = `${tr.toFixed(1)}%`;
            }

            li.innerHTML = `
                <div class="strat-card-header">
                    <span class="strat-card-rank">Rank #${idx + 1}</span>
                    <span class="strat-card-score">${strat.strategy_id}</span>
                </div>
                <div class="strat-card-title">${strat.name}</div>
                <div class="strat-card-metrics">
                    <span>${metricLabel}:</span>
                    <span class="strat-card-return">${metricValue}</span>
                </div>
            `;
            
            li.addEventListener("click", () => selectStrategy(strat.strategy_id));
            elStrategyList.appendChild(li);
        });
    }

    // Handle Active Ticker Filter for Rankings
    elTickerFilter.addEventListener("change", (e) => {
        activeFilter = e.target.value;
        
        // Re-render sidebar rankings
        renderSidebarRanking();
        
        // If we have selected a strategy, update its dashboard header rank as well
        if (selectedStrategyId) {
            updateDashboardHeaderRank();
        }
    });

    function updateDashboardHeaderRank() {
        const sortedList = Array.from(elStrategyList.querySelectorAll(".strategy-item"));
        const activeIdx = sortedList.findIndex(li => li.dataset.id === selectedStrategyId);
        if (activeIdx !== -1) {
            elStratRank.textContent = `#${activeIdx + 1}`;
        }
    }

    // -------------------------------------------------------------------------
    // 3. Strategy Selection & Metrics Populating
    // -------------------------------------------------------------------------
    function selectStrategy(strategyId) {
        selectedStrategyId = strategyId;
        showNoSelectionState(false);

        // Highlight sidebar active item
        elStrategyList.querySelectorAll(".strategy-item").forEach(li => {
            if (li.dataset.id === strategyId) {
                li.classList.add("active");
            } else {
                li.classList.remove("active");
            }
        });

        const strategy = db.ranking.find(s => s.strategy_id === strategyId);
        if (!strategy) return;

        // 1. Header Information
        elStratName.textContent = strategy.name;
        elStratDesc.textContent = strategy.description;
        updateDashboardHeaderRank();
        
        // Render Indicators Tags
        elStratTags.innerHTML = "";
        strategy.indicators.forEach(ind => {
            const tag = document.createElement("span");
            tag.className = "tag-item";
            tag.textContent = ind;
            elStratTags.appendChild(tag);
        });

        // 2. Summary Metric Cards (Dynamic depending on activeFilter context)
        let displayReturn, displayCagr, displaySharpe, displayMaxDD, displayBeaten;
        
        if (activeFilter === "GLOBAL") {
            displayReturn = `${strategy.aggregate_metrics.avg_return.toFixed(1)}%`;
            displayCagr = `${strategy.aggregate_metrics.avg_cagr.toFixed(1)}%`;
            displaySharpe = strategy.aggregate_metrics.avg_sharpe.toFixed(2);
            displayMaxDD = `${strategy.aggregate_metrics.avg_max_dd.toFixed(1)}%`;
            const totalTickers = Object.keys(strategy.ticker_results).length;
            displayBeaten = `${strategy.aggregate_metrics.outperform_count} / ${totalTickers}`;
            
            elAggReturnSub.textContent = `CAGR Promedio: ${displayCagr}`;
            elAggSharpeStatus.textContent = getSharpeStatus(strategy.aggregate_metrics.avg_sharpe);
        } else {
            const tkData = strategy.ticker_results[activeFilter];
            displayReturn = `${tkData.metrics.total_return.toFixed(1)}%`;
            displayCagr = `${tkData.metrics.cagr.toFixed(1)}%`;
            displaySharpe = tkData.metrics.sharpe.toFixed(2);
            displayMaxDD = `${tkData.metrics.max_drawdown.toFixed(1)}%`;
            displayBeaten = tkData.outperformed ? "Superado" : "No Superado";
            
            elAggReturnSub.textContent = `CAGR en ${activeFilter}: ${displayCagr}`;
            elAggSharpeStatus.textContent = getSharpeStatus(tkData.metrics.sharpe);
        }
        
        elAggReturn.textContent = displayReturn;
        elAggSharpe.textContent = displaySharpe;
        elAggDrawdown.textContent = displayMaxDD;
        elAggBeaten.textContent = displayBeaten;

        // 3. Render Ticker Selector Chips for Chart tab
        renderTickerChips(strategy);

        // 4. Update current ticker detail & plot equity curve
        // Keep the previous selection if it's one of the options, otherwise fallback to SPY
        if (!strategy.ticker_results[currentTicker]) {
            currentTicker = "SPY";
        }
        updateTickerDetails(strategy, currentTicker);

        // 5. Populate Ticker Breakdown Comparison Table
        populateComparisonTable(strategy);

        // 6. Populate Pine Script v5 Code
        elPineScriptCode.textContent = strategy.pinescript;

        // Reset scroll position of code container
        elPineScriptCode.parentElement.parentElement.scrollTop = 0;
    }

    function getSharpeStatus(val) {
        if (val >= 2) return "Excelente";
        if (val >= 1.5) return "Muy Bueno";
        if (val >= 1.0) return "Bueno";
        if (val >= 0.5) return "Moderado";
        return "Bajo/Riesgoso";
    }

    function showNoSelectionState(show) {
        if (show) {
            elNoSelectionState.classList.remove("hidden");
            elDashboardState.classList.add("hidden");
        } else {
            elNoSelectionState.classList.add("hidden");
            elDashboardState.classList.remove("hidden");
        }
    }

    // -------------------------------------------------------------------------
    // 4. Ticker Chips & Specific Ticker Details (Chart Tab)
    // -------------------------------------------------------------------------
    function renderTickerChips(strategy) {
        elTickerChips.innerHTML = "";
        
        // Add SPY, QQQ, etc. dynamically
        const tickers = Object.keys(strategy.ticker_results);
        tickers.forEach(tk => {
            const chip = document.createElement("div");
            chip.className = `ticker-chip ${tk === currentTicker ? 'active' : ''}`;
            chip.textContent = tk;
            
            chip.addEventListener("click", () => {
                currentTicker = tk;
                elTickerChips.querySelectorAll(".ticker-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                
                updateTickerDetails(strategy, tk);
            });
            elTickerChips.appendChild(chip);
        });
    }

    function updateTickerDetails(strategy, ticker) {
        const tkData = strategy.ticker_results[ticker];
        const tkBh = db.benchmarks[ticker];
        if (!tkData || !tkBh) return;

        // UI Labels
        elCurrentTickerName.textContent = ticker;
        elTradesTickerTitle.textContent = ticker;
        
        elTkReturn.textContent = `${tkData.metrics.total_return.toFixed(1)}%`;
        elTkBhReturn.textContent = `${tkBh.total_return.toFixed(1)}%`;
        
        elTkDrawdown.textContent = `${tkData.metrics.max_drawdown.toFixed(1)}%`;
        elTkBhDrawdown.textContent = `${tkBh.max_drawdown.toFixed(1)}%`;
        
        elTkSharpe.textContent = tkData.metrics.sharpe.toFixed(2);
        elTkWinrate.textContent = `${tkData.metrics.win_rate.toFixed(1)}%`;
        elTkPf.textContent = tkData.metrics.profit_factor.toFixed(2);
        elTkTrades.textContent = tkData.metrics.num_trades;
        elTkDuration.textContent = `${Math.round(tkData.metrics.avg_duration)} días`;

        // Style return indicators
        elTkReturn.className = tkData.metrics.total_return >= 0 ? "text-success" : "text-danger";
        elTkDrawdown.className = "text-danger";

        // Performance banner
        const diff = tkData.metrics.total_return - tkBh.total_return;
        elOutperformanceMargin.textContent = `${Math.abs(diff).toFixed(1)}%`;
        
        if (diff >= 0) {
            elBenchCompBanner.className = "bench-comparison-card";
            elBenchCompBanner.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i>
                <span>¡Supera a Buy & Hold por <strong id="outperformance-margin">${diff.toFixed(1)}%</strong>!</span>`;
        } else {
            elBenchCompBanner.className = "bench-comparison-card beaten-false";
            elBenchCompBanner.innerHTML = `<i class="fa-solid fa-circle-xmark text-danger"></i>
                <span>Rinde <strong id="outperformance-margin">${Math.abs(diff).toFixed(1)}%</strong> menos que Buy & Hold</span>`;
        }

        // Draw Equity Curves
        renderEquityChart(tkData.equity_curve, tkBh.equity_curve, ticker);

        // Populate Recent Trades Table
        populateTradesTable(tkData.trades);
    }

    // -------------------------------------------------------------------------
    // 5. Chart.js Equity Curve Graph
    // -------------------------------------------------------------------------
    function renderEquityChart(strategyCurve, benchmarkCurve, tickerName) {
        if (equityChart) {
            equityChart.destroy();
        }

        // Align dates
        const dates = strategyCurve.map(pt => pt.date);
        const strategyValues = strategyCurve.map(pt => pt.value);
        
        // Map benchmark values. Dates might be slightly misaligned due to decimation steps,
        // so we map benchmark values to closest dates or align indexing.
        // Since we decimated them with the same step, lengths match closely.
        // Let's map benchmark values directly based on matching indices to keep it simple.
        const benchmarkValues = benchmarkCurve.map(pt => pt.value);

        const ctx = document.getElementById("equityChart").getContext("2d");
        
        // Custom Area Gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, "rgba(92, 96, 245, 0.25)");
        gradient.addColorStop(1, "rgba(92, 96, 245, 0.0)");

        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Estrategia',
                        data: strategyValues,
                        borderColor: '#5c60f5',
                        borderWidth: 2,
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.15,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Buy & Hold (Referencia)',
                        data: benchmarkValues,
                        borderColor: 'rgba(255, 179, 0, 0.8)',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.15,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#8a94a6',
                            font: {
                                family: 'Inter',
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 17, 24, 0.95)',
                        titleColor: '#f5f6fa',
                        bodyColor: '#8a94a6',
                        borderColor: 'rgba(255, 255, 255, 0.08)',
                        borderWidth: 1,
                        padding: 10,
                        font: {
                            family: 'Inter'
                        },
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.02)'
                        },
                        ticks: {
                            color: '#8a94a6',
                            font: { size: 10 },
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.02)'
                        },
                        ticks: {
                            color: '#8a94a6',
                            font: { size: 10 },
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------------------
    // 6. Cross-Ticker Breakdown Table
    // -------------------------------------------------------------------------
    function populateComparisonTable(strategy) {
        elCompareTableBody.innerHTML = "";
        
        const tickers = Object.keys(strategy.ticker_results);
        
        tickers.forEach(tk => {
            const tkData = strategy.ticker_results[tk];
            const tkBh = db.benchmarks[tk];
            if (!tkData || !tkBh) return;

            const diff = tkData.metrics.total_return - tkBh.total_return;
            const diffClass = diff >= 0 ? "text-success" : "text-danger";
            const diffPrefix = diff >= 0 ? "+" : "";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="compare-row-ticker">${tk}</td>
                <td class="${tkData.metrics.total_return >= 0 ? 'text-success' : 'text-danger'} font-bold">${tkData.metrics.total_return.toFixed(1)}%</td>
                <td>${tkBh.total_return.toFixed(1)}%</td>
                <td class="${diffClass} font-bold">${diffPrefix}${diff.toFixed(1)}%</td>
                <td class="text-danger">${tkData.metrics.max_drawdown.toFixed(1)}%</td>
                <td>${tkBh.max_drawdown.toFixed(1)}%</td>
                <td>${tkData.metrics.sharpe.toFixed(2)}</td>
                <td>${tkData.metrics.num_trades}</td>
                <td>${tkData.metrics.win_rate.toFixed(1)}%</td>
                <td class="${tkData.metrics.is_open ? 'text-success font-bold' : ''}">${tkData.metrics.is_open ? 'SÍ' : 'NO'}</td>
                <td>${tkData.metrics.exit_threshold || '-'}</td>
                <td class="live-price-cell" data-ticker="${tk}" data-last-price="${tkData.metrics.current_price}">$${tkData.metrics.current_price.toFixed(2)}</td>
                <td>
                    <span class="badge ${tkData.outperformed ? 'badge-success' : 'badge-danger'}">
                        ${tkData.outperformed ? 'Superó B&H' : 'No Superó'}
                    </span>
                </td>
            `;
            
            // Double click row to select that ticker in the dashboard chart
            tr.addEventListener("click", () => {
                currentTicker = tk;
                
                // Swap back to the chart tab to view details
                switchTab("tab-chart");
                
                // Highlight the correct chip in UI
                elTickerChips.querySelectorAll(".ticker-chip").forEach(chip => {
                    if (chip.textContent === tk) {
                        chip.classList.add("active");
                    } else {
                        chip.classList.remove("active");
                    }
                });
                
                updateTickerDetails(strategy, tk);
            });
            
            elCompareTableBody.appendChild(tr);
        });
    }

    // -------------------------------------------------------------------------
    // 7. Trade Log Table
    // -------------------------------------------------------------------------
    function populateTradesTable(trades) {
        elTradesTableBody.innerHTML = "";
        
        if (!trades || trades.length === 0) {
            elTradesTableBody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--text-muted);padding:30px;">
                No se registran operaciones en el historial para este activo.
            </td></tr>`;
            return;
        }

        // Show all trades (reverse chronological order - newest first)
        const recentTrades = [...trades].reverse();
        
        recentTrades.forEach((trade, idx) => {
            const retClass = trade.pct_return >= 0 ? "text-success" : "text-danger";
            const retPrefix = trade.pct_return >= 0 ? "+" : "";

            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${trades.length - idx}</td>
                <td>${trade.entry_date}</td>
                <td>$${trade.entry_price.toFixed(2)}</td>
                <td>${trade.exit_date}</td>
                <td>$${trade.exit_price.toFixed(2)}</td>
                <td>${trade.duration_days} días</td>
                <td class="${retClass} font-bold">${retPrefix}${trade.pct_return.toFixed(2)}%</td>
                <td><span style="font-size:11px;opacity:0.8;">${trade.reason}</span></td>
            `;
            elTradesTableBody.appendChild(tr);
        });
    }

    // -------------------------------------------------------------------------
    // 8. Clipboard Utilities & UI Interactivity
    // -------------------------------------------------------------------------
    elBtnCopyCode.addEventListener("click", () => {
        const code = elPineScriptCode.textContent;
        navigator.clipboard.writeText(code).then(() => {
            // Show toast
            elToastMessage.classList.add("show");
            setTimeout(() => {
                elToastMessage.classList.remove("show");
            }, 3000);
        }).catch(err => {
            alert("No se pudo copiar el código: " + err);
        });
    });

    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.dataset.tab;
            switchTab(targetTab);
        });
    });

    function switchTab(tabId) {
        activeTab = tabId;
        
        // Buttons highlight
        document.querySelectorAll(".tab-btn").forEach(b => {
            if (b.dataset.tab === tabId) {
                b.classList.add("active");
            } else {
                b.classList.remove("active");
            }
        });
        
        // Contents switch
        document.querySelectorAll(".tab-content").forEach(content => {
            if (content.id === tabId) {
                content.classList.add("active");
            } else {
                content.classList.remove("active");
            }
        });

        // Trigger chart redraw to handle resizing if swapping back to chart
        if (tabId === "tab-chart" && equityChart) {
            setTimeout(() => {
                equityChart.resize();
            }, 50);
        }
    }

    // -------------------------------------------------------------------------
    // 9. Live Price Polling (10s)
    // -------------------------------------------------------------------------
    setInterval(async () => {
        try {
            const res = await fetch('/api/live-prices');
            if (!res.ok) return;
            const prices = await res.json();
            
            // Update table cells
            document.querySelectorAll('.live-price-cell').forEach(cell => {
                const tk = cell.getAttribute('data-ticker');
                if (prices[tk]) {
                    const newPrice = prices[tk];
                    const oldPrice = parseFloat(cell.getAttribute('data-last-price'));
                    
                    if (newPrice !== oldPrice) {
                        cell.innerHTML = `$${newPrice.toFixed(2)}`;
                        cell.setAttribute('data-last-price', newPrice);
                        
                        // Flash red or green
                        cell.classList.remove('flash-green', 'flash-red');
                        // Force reflow to restart animation
                        void cell.offsetWidth;
                        
                        if (newPrice > oldPrice) {
                            cell.classList.add('flash-green');
                        } else {
                            cell.classList.add('flash-red');
                        }
                    }
                }
            });
        } catch (e) {
            // Ignore polling errors silently
        }
    }, 10000);

    // -------------------------------------------------------------------------
    // Start Load
    // -------------------------------------------------------------------------
    loadDatabase();
});
