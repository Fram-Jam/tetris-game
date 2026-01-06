"""
Alert Notification System
==========================
Send alerts via:
- Email (SMTP)
- SMS (Twilio)
- Desktop notifications
- Webhook (Slack, Discord, etc.)
"""

import os
import json
import logging
import smtplib
import html
import shlex
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Deque
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger("stock_analyzer.alerts")


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertType(Enum):
    """Types of alerts."""
    SIGNAL_BUY = "signal_buy"
    SIGNAL_SELL = "signal_sell"
    PRICE_ALERT = "price_alert"
    RISK_WARNING = "risk_warning"
    MODEL_DRIFT = "model_drift"
    BACKTEST_COMPLETE = "backtest_complete"
    ERROR = "error"


@dataclass
class Alert:
    """Represents an alert to be sent."""
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    symbol: Optional[str] = None
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.alert_type.value,
            'priority': self.priority.value,
            'title': self.title,
            'message': self.message,
            'symbol': self.symbol,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    def format_text(self) -> str:
        """Format alert as plain text."""
        lines = [
            f"[{self.priority.name}] {self.title}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if self.symbol:
            lines.append(f"Symbol: {self.symbol}")
        if self.price:
            lines.append(f"Price: ${self.price:.2f}")
        lines.append("")
        lines.append(self.message)
        return "\n".join(lines)

    def format_html(self) -> str:
        """Format alert as HTML with proper escaping to prevent XSS."""
        priority_colors = {
            AlertPriority.LOW: "#6c757d",
            AlertPriority.MEDIUM: "#17a2b8",
            AlertPriority.HIGH: "#ffc107",
            AlertPriority.CRITICAL: "#dc3545"
        }
        color = priority_colors.get(self.priority, "#6c757d")

        # Escape all user-provided content to prevent XSS
        safe_title = html.escape(self.title)
        safe_message = html.escape(self.message)
        safe_symbol = html.escape(self.symbol) if self.symbol else None

        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">{safe_title}</h2>
                <small>{self.priority.name} Priority</small>
            </div>
            <div style="background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                <p><strong>Time:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        if safe_symbol:
            html_content += f"<p><strong>Symbol:</strong> {safe_symbol}</p>"
        if self.price:
            html_content += f"<p><strong>Price:</strong> ${self.price:.2f}</p>"

        html_content += f"""
                <hr>
                <p>{safe_message}</p>
            </div>
            <div style="background: #e9ecef; padding: 10px; text-align: center; border-radius: 0 0 5px 5px;">
                <small>StockAnalyzer Pro Alert System</small>
            </div>
        </div>
        """
        return html_content


class AlertChannel(ABC):
    """Abstract base class for alert channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert. Returns True if successful."""
        pass

    @abstractmethod
    def test(self) -> bool:
        """Test the channel connection."""
        pass


class EmailChannel(AlertChannel):
    """Send alerts via email using SMTP."""

    def __init__(
        self,
        smtp_host: str = None,
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
        from_email: str = None,
        to_emails: List[str] = None,
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("SMTP_FROM_EMAIL")
        self.to_emails = to_emails or os.getenv("ALERT_EMAILS", "").split(",")
        self.use_tls = use_tls

    def send(self, alert: Alert) -> bool:
        """Send email alert."""
        if not all([self.smtp_host, self.username, self.password, self.from_email]):
            logger.warning("Email not configured properly")
            return False

        if not self.to_emails or not self.to_emails[0]:
            logger.warning("No recipient emails configured")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[StockAnalyzer] {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ", ".join(self.to_emails)

            text_part = MIMEText(alert.format_text(), 'plain')
            html_part = MIMEText(alert.format_html(), 'html')

            msg.attach(text_part)
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.to_emails, msg.as_string())

            logger.info(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def test(self) -> bool:
        """Test email connection."""
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
            return True
        except Exception as e:
            logger.error(f"Email test failed: {e}")
            return False


class SMSChannel(AlertChannel):
    """Send alerts via SMS using Twilio."""

    def __init__(
        self,
        account_sid: str = None,
        auth_token: str = None,
        from_number: str = None,
        to_numbers: List[str] = None
    ):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
        self.to_numbers = to_numbers or os.getenv("ALERT_PHONE_NUMBERS", "").split(",")

    def send(self, alert: Alert) -> bool:
        """Send SMS alert."""
        if not all([self.account_sid, self.auth_token, self.from_number]):
            logger.warning("Twilio not configured")
            return False

        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)

            # Truncate message for SMS
            sms_text = f"[{alert.priority.name}] {alert.title}"
            if alert.symbol:
                sms_text += f" | {alert.symbol}"
            if alert.price:
                sms_text += f" @ ${alert.price:.2f}"
            sms_text += f"\n{alert.message[:100]}..."

            for number in self.to_numbers:
                if number:
                    client.messages.create(
                        body=sms_text,
                        from_=self.from_number,
                        to=number
                    )

            logger.info(f"SMS alert sent: {alert.title}")
            return True

        except ImportError:
            logger.warning("Twilio library not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

    def test(self) -> bool:
        """Test Twilio connection."""
        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)
            client.api.accounts(self.account_sid).fetch()
            return True
        except Exception as e:
            logger.error(f"Twilio test failed: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Send alerts via webhook (Slack, Discord, etc.)."""

    def __init__(self, webhook_url: str = None, platform: str = "slack"):
        self.webhook_url = webhook_url or os.getenv("WEBHOOK_URL")
        self.platform = platform

    def send(self, alert: Alert) -> bool:
        """Send webhook alert."""
        if not self.webhook_url:
            logger.warning("Webhook URL not configured")
            return False

        try:
            import requests

            if self.platform == "slack":
                payload = self._format_slack(alert)
            elif self.platform == "discord":
                payload = self._format_discord(alert)
            else:
                payload = {"text": alert.format_text()}

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            logger.info(f"Webhook alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False

    def _format_slack(self, alert: Alert) -> Dict:
        """Format for Slack."""
        color_map = {
            AlertPriority.LOW: "#6c757d",
            AlertPriority.MEDIUM: "#17a2b8",
            AlertPriority.HIGH: "#ffc107",
            AlertPriority.CRITICAL: "#dc3545"
        }

        fields = []
        if alert.symbol:
            fields.append({"title": "Symbol", "value": alert.symbol, "short": True})
        if alert.price:
            fields.append({"title": "Price", "value": f"${alert.price:.2f}", "short": True})

        return {
            "attachments": [{
                "color": color_map.get(alert.priority, "#6c757d"),
                "title": alert.title,
                "text": alert.message,
                "fields": fields,
                "footer": "StockAnalyzer Pro",
                "ts": int(alert.timestamp.timestamp())
            }]
        }

    def _format_discord(self, alert: Alert) -> Dict:
        """Format for Discord."""
        color_map = {
            AlertPriority.LOW: 0x6c757d,
            AlertPriority.MEDIUM: 0x17a2b8,
            AlertPriority.HIGH: 0xffc107,
            AlertPriority.CRITICAL: 0xdc3545
        }

        fields = []
        if alert.symbol:
            fields.append({"name": "Symbol", "value": alert.symbol, "inline": True})
        if alert.price:
            fields.append({"name": "Price", "value": f"${alert.price:.2f}", "inline": True})

        return {
            "embeds": [{
                "title": alert.title,
                "description": alert.message,
                "color": color_map.get(alert.priority, 0x6c757d),
                "fields": fields,
                "footer": {"text": "StockAnalyzer Pro"},
                "timestamp": alert.timestamp.isoformat()
            }]
        }

    def test(self) -> bool:
        """Test webhook connection."""
        try:
            import requests
            response = requests.post(
                self.webhook_url,
                json={"text": "StockAnalyzer test message"},
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False


class DesktopChannel(AlertChannel):
    """Send desktop notifications."""

    def send(self, alert: Alert) -> bool:
        """Send desktop notification."""
        try:
            # Try different notification methods
            if self._try_plyer(alert):
                return True
            if self._try_macos(alert):
                return True
            if self._try_linux(alert):
                return True

            logger.warning("No desktop notification method available")
            return False

        except Exception as e:
            logger.error(f"Desktop notification failed: {e}")
            return False

    def _try_plyer(self, alert: Alert) -> bool:
        """Try plyer library."""
        try:
            from plyer import notification
            notification.notify(
                title=alert.title,
                message=alert.message[:256],
                app_name="StockAnalyzer Pro",
                timeout=10
            )
            return True
        except ImportError:
            return False

    def _try_macos(self, alert: Alert) -> bool:
        """Try macOS notification with proper escaping to prevent command injection."""
        try:
            import subprocess
            import sys

            if sys.platform != 'darwin':
                return False

            # Sanitize inputs: remove quotes and escape special characters
            # AppleScript uses backslash to escape quotes
            safe_title = alert.title.replace('\\', '\\\\').replace('"', '\\"')[:100]
            safe_message = alert.message.replace('\\', '\\\\').replace('"', '\\"')[:100]

            script = f'display notification "{safe_message}" with title "{safe_title}"'
            subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
            return True
        except Exception:
            return False

    def _try_linux(self, alert: Alert) -> bool:
        """Try Linux notification with arguments passed safely (no shell)."""
        try:
            import subprocess
            import sys

            if sys.platform != 'linux':
                return False

            # Passing arguments as list avoids shell injection
            # Limit message length to prevent potential issues
            safe_title = alert.title[:100]
            safe_message = alert.message[:256]

            subprocess.run([
                'notify-send',
                '--',  # End of options, prevents title starting with - being interpreted
                safe_title,
                safe_message
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False

    def test(self) -> bool:
        """Test desktop notifications."""
        test_alert = Alert(
            alert_type=AlertType.SIGNAL_BUY,
            priority=AlertPriority.LOW,
            title="Test Notification",
            message="This is a test from StockAnalyzer Pro"
        )
        return self.send(test_alert)


class AlertManager:
    """
    Central alert management system.

    Features:
    - Multiple channel support
    - Priority-based filtering
    - Rate limiting
    - Alert history
    """

    def __init__(self, max_history: int = 1000):
        self.channels: Dict[str, AlertChannel] = {}
        self.min_priority = AlertPriority.LOW
        # Use deque with maxlen to automatically limit memory usage
        self.alert_history: Deque[Alert] = deque(maxlen=max_history)
        self.rate_limits: Dict[str, datetime] = {}
        self.rate_limit_seconds = 60  # Minimum seconds between similar alerts

    def add_channel(self, name: str, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self.channels[name] = channel
        logger.info(f"Added alert channel: {name}")

    def remove_channel(self, name: str) -> None:
        """Remove an alert channel."""
        if name in self.channels:
            del self.channels[name]

    def set_min_priority(self, priority: AlertPriority) -> None:
        """Set minimum priority for sending alerts."""
        self.min_priority = priority

    def _should_rate_limit(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        key = f"{alert.alert_type.value}_{alert.symbol or ''}"

        if key in self.rate_limits:
            elapsed = (datetime.now() - self.rate_limits[key]).total_seconds()
            if elapsed < self.rate_limit_seconds:
                return True

        self.rate_limits[key] = datetime.now()
        return False

    def send(self, alert: Alert, channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send alert through specified channels.

        Args:
            alert: Alert to send
            channels: List of channel names (None = all channels)

        Returns:
            Dict of channel name -> success status
        """
        results = {}

        # Check priority
        if alert.priority.value < self.min_priority.value:
            logger.debug(f"Alert below min priority: {alert.title}")
            return results

        # Check rate limiting
        if self._should_rate_limit(alert):
            logger.debug(f"Alert rate limited: {alert.title}")
            return results

        # Record in history (deque automatically handles size limit)
        self.alert_history.append(alert)

        # Send through channels
        target_channels = channels or list(self.channels.keys())

        for channel_name in target_channels:
            if channel_name in self.channels:
                try:
                    success = self.channels[channel_name].send(alert)
                    results[channel_name] = success
                except Exception as e:
                    logger.error(f"Channel {channel_name} failed: {e}")
                    results[channel_name] = False

        return results

    def test_all_channels(self) -> Dict[str, bool]:
        """Test all configured channels."""
        results = {}
        for name, channel in self.channels.items():
            results[name] = channel.test()
        return results


# Convenience functions
def create_signal_alert(
    symbol: str,
    signal: str,
    price: float,
    confidence: float,
    reason: str = ""
) -> Alert:
    """Create a trading signal alert."""
    if signal.upper() == "BUY":
        alert_type = AlertType.SIGNAL_BUY
        title = f"BUY Signal: {symbol}"
        priority = AlertPriority.HIGH if confidence > 0.7 else AlertPriority.MEDIUM
    else:
        alert_type = AlertType.SIGNAL_SELL
        title = f"SELL Signal: {symbol}"
        priority = AlertPriority.HIGH if confidence > 0.7 else AlertPriority.MEDIUM

    message = f"Trading signal generated with {confidence:.1%} confidence."
    if reason:
        message += f"\n\nReason: {reason}"

    return Alert(
        alert_type=alert_type,
        priority=priority,
        title=title,
        message=message,
        symbol=symbol,
        price=price,
        metadata={'confidence': confidence, 'reason': reason}
    )


def create_risk_alert(
    symbol: str,
    risk_type: str,
    current_value: float,
    limit_value: float
) -> Alert:
    """Create a risk warning alert."""
    return Alert(
        alert_type=AlertType.RISK_WARNING,
        priority=AlertPriority.CRITICAL,
        title=f"Risk Warning: {symbol}",
        message=f"{risk_type} limit triggered!\n"
                f"Current: {current_value:.2%}, Limit: {limit_value:.2%}",
        symbol=symbol,
        metadata={'risk_type': risk_type, 'current': current_value, 'limit': limit_value}
    )


# Global alert manager instance
alert_manager = AlertManager()
