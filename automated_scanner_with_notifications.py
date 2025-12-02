#!/usr/bin/env python3
"""
AUTOMATED SCANNER WITH NOTIFICATIONS
Sends Discord/email alerts when quality signals are found
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import warnings
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore', category=FutureWarning)

class AutomatedSignalNotifier:
    def __init__(self):
        # Configuration - Multiple ways to set Discord webhook
        
        # Method 1: From environment variable (GitHub Actions)
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
        self.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.sender_email = os.getenv('SENDER_EMAIL', '')
        self.sender_password = os.getenv('SENDER_PASSWORD', '')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL', '')
        
    def send_discord_notification(self, message, title="📊 Trading Signal Alert"):
        """Send notification to Discord via webhook"""
        if not self.discord_webhook_url:
            print("❌ Discord webhook URL not configured")
            return False
            
        try:
            data = {
                "embeds": [{
                    "title": title,
                    "description": message,
                    "color": 0x00ff00,  # Green color
                    "timestamp": datetime.now().isoformat()
                }]
            }
            
            response = requests.post(self.discord_webhook_url, json=data)
            
            if response.status_code == 204:
                print("✅ Discord notification sent successfully")
                return True
            else:
                print(f"❌ Discord notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Discord notification error: {e}")
            return False
        
def market_is_tradable(self, vix_threshold=22.0):
    """Simple VIX-based kill switch. Returns False in very high volatility."""
    try:
        vix = yf.download('^VIX', period='5d', interval='1d', progress=False)
        if vix.empty:
            print("⚠️ Could not fetch VIX, proceeding anyway.")
            return True

        latest_vix = float(vix['Close'].iloc[-1])
        print(f"📊 Current VIX: {latest_vix:.2f}")

        if latest_vix > vix_threshold:
            print(f"⛔ VIX {latest_vix:.1f} > {vix_threshold} — skipping new trades today.")
            return False

        return True

    except Exception as e:
        print(f"⚠️ VIX check failed ({e}), proceeding without filter.")
        return True

    def send_email_notification(self, subject, message):
        """Send email notification"""
        if not self.email_enabled or not all([self.sender_email, self.sender_password, self.recipient_email]):
            print("❌ Email not configured or not enabled")
            return False
            
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Gmail SMTP
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
            server.quit()
            
            print("✅ Email notification sent successfully")
            return True
            
        except Exception as e:
            print(f"❌ Email notification error: {e}")
            return False
    
    def get_stock_data(self, symbol, period="3mo"):
        """Get recent stock data using Yahoo Finance"""
        try:
            time.sleep(0.1)  # Be respectful to API
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                return None
            
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            print(f"   ❌ Error with {symbol}: {e}")
            return None
    
def calculate_signals(self, df, ma_period=20, std_multiplier=2.5):
    """Calculate mean reversion signals with trend + RSI filters."""
    # Need enough data for MA20, MA50, RSI
    min_len = max(ma_period, 50) + 5
    if len(df) < min_len:
        return None

    df = df.copy()
    df['MA_20'] = df['Close'].rolling(window=ma_period).mean()
    df['STD_20'] = df['Close'].rolling(window=ma_period).std()
    df['Threshold'] = df['MA_20'] - (std_multiplier * df['STD_20'])
    df['Below_Threshold'] = df['Close'] < df['Threshold']
    df['Distance_from_MA'] = (df['Close'] - df['MA_20']) / df['MA_20'] * 100

    # Trend filter (uptrend on this timeframe)
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['Trend_OK'] = (df['Close'] > df['MA_50']) & (df['MA_20'] > df['MA_50'])

    # RSI 14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_OK'] = df['RSI_14'] < 35  # oversold

    return df

    
    def scan_watchlist(self, symbols, account_size=1000):
        """Scan watchlist and return opportunities"""
        print(f"🔍 Scanning {len(symbols)} stocks...")
        
        opportunities = []
        
        for symbol in symbols:
            try:
                df = self.get_stock_data(symbol)
                if df is None:
                    continue
                    
                df = self.calculate_signals(df)
                if df is None:
                    continue
                    
                latest = df.iloc[-1]
                
                # Check for signal with quality filters
                # Check for signal with quality filters
                if latest['Below_Threshold'] and latest.get('Trend_OK', True) and latest.get('RSI_OK', True):

                    potential_gain = ((latest['MA_20'] - latest['Close']) / latest['Close']) * 100
                    risk_level = abs(latest['Distance_from_MA'])
                    
                    # Calculate risk/reward ratio
                    target_position = account_size * 0.20  # 20% position
                    potential_gain_dollars = (latest['MA_20'] - latest['Close']) * (target_position / latest['Close'])
                    potential_loss_dollars = (latest['Close'] * 0.15) * (target_position / latest['Close'])
                    risk_reward_ratio = potential_gain_dollars / potential_loss_dollars if potential_loss_dollars > 0 else 0
                    
                    # Quality filters
                    min_risk_reward = 0.75
                    min_profit_potential = 4.0
                    
                    if risk_reward_ratio >= min_risk_reward and potential_gain >= min_profit_potential:
                        opportunities.append({
                            'symbol': symbol,
                            'current_price': latest['Close'],
                            'ma_20': latest['MA_20'],
                            'threshold': latest['Threshold'],
                            'distance_from_ma': latest['Distance_from_MA'],
                            'potential_gain': potential_gain,
                            'risk_level': risk_level,
                            'risk_reward_ratio': risk_reward_ratio,
                            'potential_gain_dollars': potential_gain_dollars,
                            'potential_loss_dollars': potential_loss_dollars
                        })
                        
                        print(f"✅ {symbol}: Signal detected! Gain: {potential_gain:.1f}%, R/R: 1:{risk_reward_ratio:.2f}")
                    else:
                        print(f"⚠️ {symbol}: Signal filtered (Gain: {potential_gain:.1f}%, R/R: 1:{risk_reward_ratio:.2f})")
                else:
                    print(f"📊 {symbol}: No signal")
                    
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
        
        return opportunities
    
    def format_notification_message(self, opportunities, account_size=1000):
        """Format opportunities into notification message"""
        if not opportunities:
            return "📊 **Daily Scan Complete**\n\n✅ No quality signals found today.\nWaiting for better opportunities with good risk/reward ratios."
        
        message = f"🚨 **{len(opportunities)} QUALITY SIGNAL(S) DETECTED!**\n\n"
        
        for i, opp in enumerate(opportunities, 1):
            target_position = account_size * 0.20
            shares_needed = target_position / opp['current_price']
            
            risk_quality = "🟢 EXCELLENT" if opp['risk_reward_ratio'] >= 1.5 else "🟡 GOOD" if opp['risk_reward_ratio'] >= 1.0 else "🟠 ACCEPTABLE"
            
            message += f"**{i}. {opp['symbol']} - {risk_quality}**\n"
            message += f"📊 Current: ${opp['current_price']:.2f}\n"
            message += f"🎯 Target: ${opp['ma_20']:.2f}\n"
            message += f"📈 Profit: +{opp['potential_gain']:.1f}% (${opp['potential_gain_dollars']:.0f})\n"
            message += f"⚖️ Risk/Reward: 1:{opp['risk_reward_ratio']:.2f}\n"
            message += f"💰 Position: {shares_needed:.3f} shares (${target_position:.0f})\n"
            message += f"🛑 Stop Loss: ${opp['current_price'] * 0.85:.2f}\n\n"
        
        message += "🎯 **Action Required:**\n"
        message += "• Log into your trading platform\n"
        message += "• Place dollar-based orders for fractional shares\n"
        message += "• Set stop losses at 15% below entry\n"
        message += "• Set target alerts at MA levels\n\n"
        message += f"⏰ Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return message
    
    def run_automated_scan(self):
        """Run the full automated scan with notifications"""
        print("🚀 Starting automated scan...")

    # VIX-based kill switch
    if not self.market_is_tradable():
        print("🛑 Market conditions not suitable (high VIX). No scan run.")
        return []
    
        print("🤖 AUTOMATED TRADING SIGNAL SCANNER")
        print("=" * 40)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load watchlist
        watchlist_file = None
        
        # Try to find latest watchlist file
        import glob
        watchlist_files = glob.glob('integrated_watchlist_*.txt')
        
        if watchlist_files:
            watchlist_file = max(watchlist_files, key=os.path.getctime)
            print(f"📋 Loading watchlist: {watchlist_file}")
            
            try:
                with open(watchlist_file, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                print(f"✅ Loaded {len(symbols)} symbols from watchlist")
            except Exception as e:
                print(f"❌ Error loading watchlist: {e}")
                symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Fallback
                print("📊 Using fallback symbols")
        else:
            # Fallback symbols if no watchlist found
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
            print("📊 No watchlist found, using default symbols")
        
        print()
        
        # Run scan
        opportunities = self.scan_watchlist(symbols)
        
        # Prepare notifications
        notification_message = self.format_notification_message(opportunities)
        
        print()
        print("📱 SENDING NOTIFICATIONS...")
        print("=" * 30)
        
        # Send Discord notification
        discord_sent = self.send_discord_notification(notification_message)
        
        # Send email notification
        email_subject = f"Trading Signals: {len(opportunities)} opportunities found" if opportunities else "Trading Signals: No opportunities today"
        email_sent = self.send_email_notification(email_subject, notification_message)
        
        # Summary
        print()
        print("📊 SCAN SUMMARY:")
        print("=" * 16)
        print(f"🔍 Stocks scanned: {len(symbols)}")
        print(f"🚨 Signals found: {len(opportunities)}")
        print(f"📱 Discord sent: {'✅' if discord_sent else '❌'}")
        print(f"📧 Email sent: {'✅' if email_sent else '❌'}")
        
        if opportunities:
            print()
            print("🎯 OPPORTUNITIES SUMMARY:")
            for opp in opportunities:
                print(f"   {opp['symbol']}: +{opp['potential_gain']:.1f}% potential, 1:{opp['risk_reward_ratio']:.2f} R/R")
        
        return opportunities

def main():
    """Main function for automated scanning"""
    notifier = AutomatedSignalNotifier()
    opportunities = notifier.run_automated_scan()
    
    # Return exit code for automation systems
    return 0 if opportunities else 1

if __name__ == "__main__":
    exit(main())
