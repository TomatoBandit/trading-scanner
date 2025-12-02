#!/usr/bin/env python3
"""
Integrated Mean Reversion Trading System - S&P 500
Automatically runs S&P 500 analysis and feeds top performers to daily scanner
Complete automation from analysis to trading instructions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import requests

warnings.filterwarnings('ignore', category=FutureWarning)

class IntegratedTradingSystem:
    def __init__(self, account_size=10000, position_size_pct=20, stop_loss_pct=15):
        self.account_size = account_size
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.backtest_trades = []
        
        # S&P 500 symbols for analysis (comprehensive list)
        self.sp500_symbols = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'CRM', 'ORCL', 'ADBE', 'NFLX', 'INTC', 'AMD', 'QCOM', 'CSCO',
            'IBM', 'TXN', 'INTU', 'LRCX', 'MU', 'AMAT', 'ADI', 'KLAC',
            'MRVL', 'SNPS', 'CDNS', 'FTNT', 'WDAY', 'TEAM', 'NOW', 'PYPL',
            'PANW', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'SNOW', 'MDB',
            
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY',
            'LLY', 'AMGN', 'MDT', 'CVS', 'GILD', 'ISRG', 'VRTX', 'REGN',
            'SYK', 'BDX', 'HUM', 'EW', 'CNC', 'CI', 'MCK', 'ABC', 'CAH',
            'BAX', 'BSX', 'ALGN', 'XRAY', 'ZBH', 'HCA', 'UHS', 'DGX', 'LH',
            
            # Financials
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'BLK', 'SCHW',
            'SPGI', 'CB', 'MMC', 'AON', 'ICE', 'CME', 'MSCI', 'TROW',
            'BK', 'STT', 'COF', 'DFS', 'MTB', 'FITB', 'HBAN', 'RF', 'KEY',
            'ZION', 'FRC', 'PNC', 'USB', 'AIG', 'MET', 'PRU', 'ALL', 'PGR',
            
            # Consumer Discretionary
            'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'TGT', 'DG',
            'ROST', 'YUM', 'CMG', 'ORLY', 'AZO', 'HLT', 'MAR', 'LVS', 'MGM',
            'RCL', 'CCL', 'DLTR', 'KMX', 'BBY', 'GM', 'F', 'TSLA', 'HOG',
            'NCLH', 'WYNN', 'EXPE', 'ULTA', 'EBAY', 'ETSY', 'LYFT', 'UBER',
            
            # Consumer Staples
            'PG', 'KO', 'PEP', 'WMT', 'COST', 'MO', 'PM', 'MDLZ', 'CL',
            'KMB', 'EL', 'GIS', 'K', 'HSY', 'SYY', 'KR', 'ADM', 'CAG',
            'HRL', 'KHC', 'CPB', 'TSN', 'BVNRY', 'STZ', 'MNST',
            
            # Industrials
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'FDX', 'LMT', 'NOC',
            'ETN', 'EMR', 'ITW', 'PH', 'CMI', 'DE', 'CSX', 'UNP', 'NSC',
            'FTV', 'ROK', 'DOV', 'XYL', 'FLS', 'AOS', 'CARR', 'OTIS', 'PWR',
            'J', 'JCI', 'IEX', 'TT', 'GNRC', 'SWK', 'FAST', 'PCAR', 'CHRW',
            'EXPD', 'JBHT', 'ODFL', 'KNX', 'PNR', 'ROL', 'WSO', 'LDOS',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI',
            'OKE', 'WMB', 'BKR', 'HAL', 'DVN', 'PXD', 'FANG', 'MRO', 'APA',
            'HES', 'OXY', 'CTRA', 'EQT', 'CNX', 'AR', 'SM', 'RRC', 'CLR',
            'NOV', 'FTI', 'TDW', 'RIG', 'VAL', 'TRGP', 'EPD', 'ET',
            
            # Materials
            'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NUE', 'DOW', 'DD', 'PPG',
            'VMC', 'MLM', 'NEM', 'FMC', 'LYB', 'CE', 'CF', 'MOS', 'IFF',
            'ALB', 'EMN', 'RPM', 'SEE', 'AVY', 'PKG', 'BALL', 'CCK', 'IP',
            'WRK', 'SON', 'NEU', 'SLGN', 'AA', 'X', 'CLF', 'MP', 'HUN',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'SRE', 'D', 'PEG', 'XEL', 'ED',
            'ES', 'AWK', 'WEC', 'CNP', 'ETR', 'EVRG', 'FE', 'AEE', 'CMS',
            'DTE', 'PPL', 'ATO', 'WTR', 'NI', 'LNT', 'OGE', 'POR', 'PNW',
            'UGI', 'SJI', 'SW', 'SR', 'MDU', 'BKH', 'AVA',
            
            # Real Estate
            'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'DLR', 'PSA',
            'EXR', 'AVB', 'EQR', 'BXP', 'VTR', 'ARE', 'DOC', 'UDR', 'ESS',
            'MAA', 'KIM', 'REG', 'HST', 'PEAK', 'CPT', 'FRT', 'BRX', 'KRG',
            'AIV', 'AKR', 'BDN', 'CDR', 'DEI', 'ELS', 'EPR', 'MAC', 'NNN'
        ]
    
    def check_for_existing_analysis(self):
        """Check for recent analysis results"""
        
        # Find all analysis files
        files = [f for f in os.listdir('.') if f.startswith('sp500_yf_analysis_results_') and f.endswith('.json')]
        if not files:
            return None
            
        # Get most recent file
        latest_file = max(files, key=os.path.getctime)
        
        try:
            # Extract date from filename
            date_str = latest_file.replace('sp500_yf_analysis_results_', '').replace('.json', '')
            file_date = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
            
            # If less than 5 days old, offer to use it
            if datetime.now() - file_date < timedelta(days=5):
                print(f"📁 Found recent analysis: {latest_file}")
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                return results
            else:
                print(f"📁 Found analysis but it's {(datetime.now() - file_date).days} days old")
                
        except Exception as e:
            print(f"⚠️ Error reading analysis file: {e}")
            
        return None
    
    def run_sp500_analysis(self, force_new=False):
        """Run S&P 500 analysis or load existing results"""
        
        if not force_new:
            existing_results = self.check_for_existing_analysis()
            if existing_results:
                use_existing = input("Use existing analysis? (y/n): ").lower().strip()
                if use_existing == 'y':
                    return existing_results
        
        print("🔍 Running fresh S&P 500 analysis...")
        print("⏱️ This will take 8-12 minutes")
        print()
        
        results = []
        start_time = time.time()
        
        for i, symbol in enumerate(self.sp500_symbols, 1):
            print(f"🧪 Analyzing {symbol} ({i}/{len(self.sp500_symbols)})...")
            
            df = self.get_stock_data(symbol, period="6mo")
            if df is None:
                continue
                
            performance = self.analyze_mean_reversion_performance(df, symbol)
            if performance is None:
                continue
                
            results.append(performance)
            
            if i % 20 == 0:
                elapsed = (time.time() - start_time) / 60
                remaining = (elapsed / i) * (len(self.sp500_symbols) - i)
                print(f"   📊 Progress: {i}/{len(self.sp500_symbols)} | ETA: {remaining:.1f} min")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sp500_yf_analysis_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Analysis complete! Saved to {filename}")
        # Generate and save backtest equity curve (CSV + PNG, optional Discord upload)
        self.generate_backtest_equity_curve()
        return results
    
    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data using Yahoo Finance"""
        try:
            time.sleep(0.1)
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                return None
            
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_mean_reversion_performance(self, df, symbol):
        """Analyze mean reversion performance with trend + RSI filters."""
        # Need enough data for MA20/50 and RSI
        if len(df) < 60:
            return None

        df = df.copy()
        # Core indicators: mean and standard deviation
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['Threshold'] = df['MA_20'] - (2 * df['STD_20'])
        df['Below_Threshold'] = df['Close'] < df['Threshold']

        # Trend filter (uptrend only)
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['Uptrend'] = (df['Close'] > df['MA_50']) & (df['MA_20'] > df['MA_50'])

        # RSI 14
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        df['RSI_14'] = 100 - (100 / (1 + rs))
        df['RSI_OK'] = df['RSI_14'] < 35  # oversold

        # Entry: fresh drop below threshold, in uptrend, and oversold
        below_threshold_prev = df['Below_Threshold'].shift(1).fillna(False)
        df['Entry_Signal'] = (
            df['Below_Threshold']
            & (~below_threshold_prev)
            & df['Uptrend']
            & df['RSI_OK']
        )

        trades = []
        for i, (date, row) in enumerate(df.iterrows()):
            if row['Entry_Signal'] and pd.notna(row['MA_20']):
                entry_price = row['Close']
                target_price = row['MA_20']

                # Use the configured stop-loss percentage (e.g., 15%)
                stop_loss_price = entry_price * (1 - (self.stop_loss_pct / 100.0))

                max_drawdown = 0.0
                peak_price = entry_price
                exit_reason = "Time Limit"
                exit_price = df['Close'].iloc[-1]
                exit_date = df.index[-1]

                # Look ahead up to 90 bars for exit
                for j in range(i + 1, min(i + 91, len(df))):
                    future_row = df.iloc[j]
                    future_date = df.index[j]
                    current_price = future_row['Close']

                    if current_price > peak_price:
                        peak_price = current_price

                    drawdown = (peak_price - current_price) / peak_price * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                    if current_price >= target_price:
                        exit_reason = "Target Hit"
                        exit_price = current_price
                        exit_date = future_date
                        break
                    elif current_price <= stop_loss_price:
                        exit_reason = "Stop Loss"
                        exit_price = current_price
                        exit_date = future_date
                        break
                    elif j == min(i + 90, len(df) - 1):
                        exit_reason = "Time Limit"
                        exit_price = current_price
                        exit_date = future_date

                profit_pct = (exit_price - entry_price) / entry_price * 100

                trade = {
                    'symbol': symbol,
                    'entry_date': date,
                    'exit_date': exit_date,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'profit_pct': float(profit_pct),
                    'max_drawdown': float(max_drawdown),
                    'exit_reason': exit_reason,
                }
                trades.append(trade)

        if not trades:
            return None

        # Store trades at the class level so we can build a portfolio equity curve later
        self.backtest_trades.extend(trades)

        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit_pct'] > 0]
        target_hit_trades = [t for t in trades if t['exit_reason'] == 'Target Hit']

        win_rate = len(winning_trades) / total_trades * 100
        avg_profit = np.mean([t['profit_pct'] for t in trades])
        target_hit_rate = len(target_hit_trades) / total_trades * 100
        avg_max_drawdown = np.mean([t['max_drawdown'] for t in trades])

        score = (
            win_rate * 0.4
            + target_hit_rate * 0.3
            + max(0, avg_profit) * 0.2
            - avg_max_drawdown * 0.1
        )

        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit_pct': avg_profit,
            'target_hit_rate': target_hit_rate,
            'avg_max_drawdown': avg_max_drawdown,
            'score': score,
            'grade': self.assign_grade(score)
        }

    def build_equity_curve_from_trades(
        self,
        trades,
        starting_equity: float = 10000.0
    ):
        """Build a simple sequential equity curve from a list of trades.
        Assumes one trade at a time, processed in order of exit_date."""
        if not trades:
            print("⚠️ No trades available to build equity curve.")
            return pd.DataFrame()

        # Sort trades by exit_date
        sorted_trades = sorted(trades, key=lambda t: t['exit_date'])

        equity_values = []
        equity = starting_equity

        for t in sorted_trades:
            profit_pct = t['profit_pct'] / 100.0
            equity *= (1 + profit_pct)
            equity_values.append({
                'date': t['exit_date'],
                'equity': equity,
                'symbol': t['symbol'],
                'profit_pct': t['profit_pct'],
                'exit_reason': t['exit_reason'],
            })

        equity_df = pd.DataFrame(equity_values)
        equity_df.sort_values('date', inplace=True)
        equity_df.reset_index(drop=True, inplace=True)
        return equity_df

    def generate_backtest_equity_curve(self):
        """Generate an equity curve from backtest trades, save CSV + PNG,
        and optionally send the image to Discord if DISCORD_WEBHOOK_URL is set."""
        if not self.backtest_trades:
            print("⚠️ No backtest trades recorded. Skipping equity curve generation.")
            return

        equity_df = self.build_equity_curve_from_trades(self.backtest_trades)

        if equity_df.empty:
            print("⚠️ Equity curve DataFrame is empty. Skipping.")
            return

        # Save CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"backtest_equity_curve_{timestamp}.csv"
        equity_df.to_csv(csv_name, index=False)
        print(f"💾 Saved backtest equity curve CSV: {csv_name}")

        # Plot equity curve
        png_name = f"backtest_equity_curve_{timestamp}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(equity_df['date'], equity_df['equity'])
        plt.title("Backtest Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(png_name)
        plt.close()
        print(f"🖼️ Saved backtest equity curve image: {png_name}")

        # Optional: send to Discord
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if not webhook_url:
            print("ℹ️ DISCORD_WEBHOOK_URL not set. Skipping Discord equity curve upload.")
            return

        try:
            with open(png_name, "rb") as f:
                files = {"file": (png_name, f, "image/png")}
                data = {
                    "content": "📊 Backtest equity curve generated by integrated_trading_system.py"
                }
                resp = requests.post(webhook_url, data=data, files=files, timeout=15)
            if resp.status_code in (200, 204):
                print("✅ Backtest equity curve image sent to Discord.")
            else:
                print(f"⚠️ Discord upload failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"⚠️ Error sending equity curve to Discord: {e}")

    def assign_grade(self, score):
        """Assign grade based on score"""
        if score >= 50: return 'A+'
        if score >= 40: return 'A'
        if score >= 30: return 'B'
        if score >= 20: return 'C'
        if score >= 10: return 'D'
        return 'F'
    
    def extract_watchlist(self, analysis_results, grade_filter=None, max_stocks=50):
        """Extract watchlist from analysis results"""
        
        # Sort by score
        analysis_results.sort(key=lambda x: x['score'], reverse=True)
        
        if grade_filter:
            filtered = [r for r in analysis_results if r['grade'] in grade_filter]
        else:
            filtered = analysis_results
        
        # Apply max_stocks limit if specified
        if max_stocks and max_stocks > 0:
            top_stocks = filtered[:max_stocks]
        else:
            top_stocks = filtered  # Use all matching stocks
        
        print()
        print("🎯 Watchlist Selection Criteria:")
        if grade_filter:
            print(f"   • Grades included: {', '.join(grade_filter)}")
        else:
            print("   • Grades: All")
        if max_stocks:
            print(f"   • Maximum stocks: {max_stocks}")
        else:
            print("   • Maximum stocks: Unlimited")
        print(f"   • Final selection: {len(top_stocks)} stocks")
        print()
        
        return top_stocks
    
    def save_watchlist(self, watchlist, filename=None):
        """Save watchlist to a text file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'integrated_watchlist_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write("# Integrated S&P 500 Mean Reversion Watchlist\n")
            f.write("# Generated by integrated_trading_system.py\n")
            f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for stock in watchlist:
                f.write(f"{stock['symbol']}\n")
        
        print(f"📄 Watchlist saved to: {filename}")
        return filename
    
    def calculate_signals(self, df, ma_period=20, std_multiplier=2):
        """Calculate current mean reversion signals"""
        if len(df) < ma_period + 5:
            return None
            
        df['MA_20'] = df['Close'].rolling(window=ma_period).mean()
        df['STD_20'] = df['Close'].rolling(window=ma_period).std()
        df['Threshold'] = df['MA_20'] - (std_multiplier * df['STD_20'])
        df['Below_Threshold'] = df['Close'] < df['Threshold']
        
        return df
    
    def calculate_trading_instructions(self, symbol, current_price, ma_20, threshold):
        """Calculate complete trading instructions"""
        
        position_size = (self.account_size * (self.position_size_pct / 100)) / current_price
        position_size = int(position_size)
        
        if position_size <= 0:
            return None
            
        entry_price = current_price
        target_price = ma_20
        stop_loss_price = entry_price * (1 - self.stop_loss_pct / 100)
        
        potential_gain_pct = (target_price - entry_price) / entry_price * 100
        potential_loss_pct = (entry_price - stop_loss_price) / entry_price * 100
        
        risk_reward_ratio = potential_gain_pct / potential_loss_pct if potential_loss_pct > 0 else 0
        
        if potential_gain_pct < 3 or risk_reward_ratio < 0.5:
            return None
        
        return {
            'symbol': symbol,
            'position_size': position_size,
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss_price': stop_loss_price,
            'potential_gain_pct': potential_gain_pct,
            'potential_loss_pct': potential_loss_pct,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def display_trade_instructions(self, instructions):
        """Display trading instructions in a clear format"""
        print("\n💼 Trading Instructions:\n")
        
        print(f"Symbol: {instructions['symbol']}")
        print(f"Position Size: {instructions['position_size']} shares")
        print(f"Entry Price: ${instructions['entry_price']:.2f}")
        print(f"Target Price (20-day MA): ${instructions['target_price']:.2f}")
        print(f"Stop Loss Price: ${instructions['stop_loss_price']:.2f}")
        print()
        print(f"Potential Gain: {instructions['potential_gain_pct']:.2f}%")
        print(f"Potential Loss: {instructions['potential_loss_pct']:.2f}%")
        print(f"Risk-Reward Ratio: {instructions['risk_reward_ratio']:.2f}")
        print()
    
    def run_interactive_session(self, analysis_results):
        """Interactive session to get trading instructions for a symbol"""
        print("\n📈 Interactive Trading Session")
        print("-----------------------------")
        print("You can:")
        print("1. See top ranked symbols")
        print("2. Enter a specific symbol for detailed analysis")
        print("3. Generate a narrowed watchlist for daily scanner")
        print("4. Save current watchlist")
        print("5. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\n🏆 Top 20 Ranked Symbols:")
                print("------------------------")
                for i, stock in enumerate(sorted(analysis_results, key=lambda x: x['score'], reverse=True)[:20], 1):
                    print(f"{i}. {stock['symbol']} | Grade: {stock['grade']} | Score: {stock['score']:.2f}")
            
            elif choice == '2':
                symbol = input("\nEnter symbol to analyze: ").upper().strip()
                
                if not any(stock['symbol'] == symbol for stock in analysis_results):
                    print("❌ Symbol not found in analysis results")
                    continue
                
                df = self.get_stock_data(symbol, period="6mo")
                if df is None:
                    print("❌ Unable to fetch data for that symbol")
                    continue
                
                signals = self.calculate_signals(df)
                if signals is None:
                    print("❌ Not enough data to calculate signals")
                    continue
                
                latest = signals.iloc[-1]
                current_price = latest['Close']
                ma_20 = latest['MA_20']
                threshold = latest['Threshold']
                
                print(f"\n📊 Current Mean Reversion Status for {symbol}:")
                print(f"Current Price: ${current_price:.2f}")
                print(f"20-day MA: ${ma_20:.2f}")
                print(f"Threshold (2 STD below MA): ${threshold:.2f}")
                
                if latest['Below_Threshold']:
                    print("Status: 📉 Below threshold - potential mean reversion opportunity")
                    
                    instructions = self.calculate_trading_instructions(symbol, current_price, ma_20, threshold)
                    if instructions:
                        self.display_trade_instructions(instructions)
                    else:
                        print("⚠️ Risk/reward not attractive enough for trading instructions")
                else:
                    print("Status: ⚪ Above threshold - no current mean reversion signal")
            
            elif choice == '3':
                print("\n🎯 Creating Watchlist for Daily Scanner")
                print("-------------------------------------")
                
                grade_choice = input("Include only grade A and A+ stocks? (y/n): ").lower().strip()
                if grade_choice == 'y':
                    grade_filter = ['A+', 'A']
                else:
                    grade_filter = None
                
                max_stocks = input("Maximum number of stocks for watchlist (default 50): ").strip()
                if max_stocks == '':
                    max_stocks = 50
                else:
                    try:
                        max_stocks = int(max_stocks)
                    except ValueError:
                        print("⚠️ Invalid number, using default 50")
                        max_stocks = 50
                
                watchlist = self.extract_watchlist(analysis_results, grade_filter, max_stocks)
                filename = self.save_watchlist(watchlist)
                print("\n✅ Watchlist generated and saved.")
                print("You can now use this watchlist with the automated daily scanner.")
            
            elif choice == '4':
                print("\n💾 Saving Current Watchlist")
                print("--------------------------")
                
                grade_filter = input("Filter by grades (comma-separated, or press Enter for all): ").strip()
                if grade_filter:
                    grades = [g.strip().upper() for g in grade_filter.split(',')]
                else:
                    grades = None
                
                max_stocks = input("Maximum number of stocks to save (or press Enter for all): ").strip()
                if max_stocks:
                    try:
                        max_stocks = int(max_stocks)
                    except ValueError:
                        print("⚠️ Invalid number, using all stocks")
                        max_stocks = None
                else:
                    max_stocks = None
                
                watchlist = self.extract_watchlist(analysis_results, grades, max_stocks)
                self.save_watchlist(watchlist)
                print("\n✅ Watchlist saved.")
            
            elif choice == '5':
                print("\n👋 Exiting interactive session.")
                break
            
            else:
                print("⚠️ Invalid choice. Please enter a number between 1 and 5.")

def is_first_monday_utc() -> bool:
    """Return True if today (UTC) is the first Monday of the month."""
    today = datetime.utcnow().date()
    # Monday is 0 in Python's weekday()
    return today.weekday() == 0 and 1 <= today.day <= 7

def main():
    print("📈 Integrated S&P 500 Mean Reversion Trading System")
    print("-------------------------------------------------")
    
    system = IntegratedTradingSystem()

    # Detect if running in GitHub Actions
    running_in_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    event_name = os.getenv("GITHUB_EVENT_NAME", "").lower()

    if running_in_ci:
        # If it's a scheduled run, only execute on the first Monday (UTC)
        if event_name == "schedule":
            if not is_first_monday_utc():
                print("🗓️ Not the first Monday of the month (UTC). Skipping analysis.")
                return
            print("📅 First Monday (scheduled run) — running full analysis.")

        # If it's a manual run (workflow_dispatch), always run
        elif event_name == "workflow_dispatch":
            print("🧪 Manual CI run (workflow_dispatch) — running full analysis regardless of date.")
        else:
            # Fallback for any other CI context
            print(f"ℹ️ CI run with event '{event_name}' — running analysis.")

        print("🤖 Running in CI: forcing fresh analysis, no prompts.")
        force_new = True

    else:
        # Local / interactive mode (same behavior as before)
        choice = input("Use existing analysis if available? (y/n): ").lower().strip()
        force_new = (choice != 'y')
    
    analysis_results = system.run_sp500_analysis(force_new=force_new)
    
    print("\n📊 Summary of Top 10 Stocks:")
    print("---------------------------")
    for i, stock in enumerate(sorted(analysis_results, key=lambda x: x['score'], reverse=True)[:10], 1):
        print(f"{i}. {stock['symbol']} | Grade: {stock['grade']} | Score: {stock['score']:.2f}")
    
    if not running_in_ci:
        # Only run the interactive session when you're at the keyboard
        system.run_interactive_session(analysis_results)

if __name__ == "__main__":
    main()


