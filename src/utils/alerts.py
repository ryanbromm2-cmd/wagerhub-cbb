"""
Alert manager for CBB Totals Model.
Sends Discord webhooks for significant edges and daily summaries.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import discord_webhook; graceful fallback if not installed
try:
    from discord_webhook import DiscordWebhook, DiscordEmbed
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.debug("discord-webhook not installed; Discord alerts disabled.")


# ── Color constants for Discord embeds ────────────────────────────────────────
COLOR_OVER = 0x00AA00    # Green  → OVER lean
COLOR_UNDER = 0xCC0000   # Red    → UNDER lean
COLOR_INFO = 0x0066CC    # Blue   → informational
COLOR_WARN = 0xFF6600    # Orange → warning


class AlertManager:
    """
    Manages all outgoing alerts for the CBB Totals pipeline.

    Reads webhook URL from DISCORD_WEBHOOK_URL env var or config.
    Gracefully no-ops when webhook is not configured.
    """

    def __init__(self, config: dict):
        self.config = config
        alert_cfg = config.get("alerts", {})

        self.alerts_enabled: bool = alert_cfg.get("enabled", True)
        self.discord_enabled: bool = alert_cfg.get("discord_enabled", False)
        self.threshold: float = float(
            os.environ.get("ALERT_THRESHOLD", alert_cfg.get("threshold", 6.0))
        )

        self.webhook_url: str = os.environ.get("DISCORD_WEBHOOK_URL", "") or ""

        if self.discord_enabled and not self.webhook_url:
            logger.warning(
                "Discord alerts enabled but DISCORD_WEBHOOK_URL is not set. "
                "Alerts will only be logged."
            )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _can_send_discord(self) -> bool:
        return (
            DISCORD_AVAILABLE
            and self.discord_enabled
            and bool(self.webhook_url)
        )

    def _post_webhook(self, webhook: "DiscordWebhook") -> bool:
        """Execute a webhook post; return True on success."""
        try:
            response = webhook.execute()
            if hasattr(response, "status_code") and response.status_code not in (200, 204):
                logger.warning(f"Discord webhook returned status {response.status_code}")
                return False
            return True
        except Exception as exc:
            logger.error(f"Failed to post Discord webhook: {exc}")
            return False

    # ── Core send methods ─────────────────────────────────────────────────

    def send_discord_alert(
        self,
        message: str,
        embeds: Optional[list] = None,
        username: str = "CBB Totals Bot",
    ) -> bool:
        """
        Post a plain message (with optional embeds) to the Discord webhook.

        Args:
            message: Plain text content.
            embeds:  List of pre-built DiscordEmbed objects.
            username: Override bot display name.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.alerts_enabled:
            return False

        # Always log the alert text
        logger.info(f"[ALERT] {message}")

        if not self._can_send_discord():
            return False

        webhook = DiscordWebhook(
            url=self.webhook_url,
            content=message,
            username=username,
        )
        if embeds:
            for embed in embeds:
                webhook.add_embed(embed)

        return self._post_webhook(webhook)

    # ── Edge alerting ─────────────────────────────────────────────────────

    def format_edge_alert(self, row: dict | "pd.Series") -> "DiscordEmbed":
        """
        Build a Discord embed for a single game edge.

        Args:
            row: Dict or Series with columns:
                 game_id, home_team, away_team, model_total, market_total,
                 differential, edge_side, confidence_score, predicted_possessions.

        Returns:
            DiscordEmbed ready to attach to a webhook.
        """
        if not DISCORD_AVAILABLE:
            raise RuntimeError("discord-webhook package is required to build embeds.")

        differential = float(row.get("differential", 0))
        edge_side = str(row.get("edge_side", "?"))
        home = str(row.get("home_team", "?"))
        away = str(row.get("away_team", "?"))
        model_total = row.get("ensemble_total") or row.get("model_total", "?")
        market_total = row.get("market_total", "?")
        confidence = row.get("confidence_score", None)
        possessions = row.get("predicted_possessions", None)
        game_date = row.get("game_date", "")

        color = COLOR_OVER if edge_side == "OVER" else COLOR_UNDER
        sign = "+" if differential > 0 else ""

        embed = DiscordEmbed(
            title=f"{'OVER' if edge_side == 'OVER' else 'UNDER'} EDGE: {away} @ {home}",
            description=(
                f"**{away} @ {home}**\n"
                f"Date: {game_date}\n\n"
                f"Market Total: **{market_total}**\n"
                f"Model Total:  **{model_total:.1f}**\n"
                f"Differential: **{sign}{differential:.1f}**\n"
                f"Edge Side:    **{edge_side}**"
            ),
            color=color,
        )
        if confidence is not None:
            embed.add_embed_field(
                name="Confidence",
                value=f"{confidence:.2f}",
                inline=True,
            )
        if possessions is not None:
            embed.add_embed_field(
                name="Proj. Possessions",
                value=f"{possessions:.1f}",
                inline=True,
            )
        embed.set_footer(text=f"CBB Totals Model • {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
        return embed

    def check_and_alert(
        self,
        projections_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> int:
        """
        Scan projections and send an alert for each game exceeding the threshold.

        Args:
            projections_df: DataFrame with projection + odds columns.
            threshold: Override the configured threshold.

        Returns:
            Number of alerts sent.
        """
        if not self.alerts_enabled:
            return 0

        alert_threshold = threshold if threshold is not None else self.threshold

        if projections_df is None or projections_df.empty:
            logger.debug("check_and_alert: no projections to evaluate.")
            return 0

        if "abs_differential" not in projections_df.columns:
            if "differential" in projections_df.columns:
                projections_df = projections_df.copy()
                projections_df["abs_differential"] = projections_df["differential"].abs()
            else:
                logger.warning("projections_df missing 'differential' column; skipping alerts.")
                return 0

        edges = projections_df[projections_df["abs_differential"] >= alert_threshold]

        if edges.empty:
            logger.info(f"No edges >= {alert_threshold} pts found; no alerts sent.")
            return 0

        logger.info(f"Found {len(edges)} edge(s) >= {alert_threshold} pts. Sending alerts.")

        sent = 0
        for _, row in edges.iterrows():
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            diff = row.get("differential", 0)
            side = row.get("edge_side", "?")
            sign = "+" if diff > 0 else ""

            msg = (
                f":alert: **{side} EDGE** | {away} @ {home} | "
                f"Diff: {sign}{diff:.1f}"
            )

            embeds = []
            if DISCORD_AVAILABLE and self._can_send_discord():
                try:
                    embeds = [self.format_edge_alert(row)]
                except Exception as exc:
                    logger.debug(f"Could not build embed: {exc}")

            self.send_discord_alert(msg, embeds=embeds if embeds else None)
            sent += 1

        return sent

    # ── Daily summary ─────────────────────────────────────────────────────

    def send_daily_summary(self, projections_df: pd.DataFrame) -> bool:
        """
        Send a daily summary embed showing today's top edges.

        Args:
            projections_df: Full DataFrame of today's projections (with odds).

        Returns:
            True if sent (or logged) successfully.
        """
        if not self.alerts_enabled:
            return False

        if projections_df is None or projections_df.empty:
            logger.info("No projections available for daily summary.")
            return False

        today = datetime.now().strftime("%Y-%m-%d")

        # Sort by abs_differential desc
        if "abs_differential" not in projections_df.columns:
            if "differential" in projections_df.columns:
                projections_df = projections_df.copy()
                projections_df["abs_differential"] = projections_df["differential"].abs()
            else:
                logger.warning("Cannot build daily summary: missing differential column.")
                return False

        df = projections_df.dropna(subset=["differential"]).sort_values(
            "abs_differential", ascending=False
        )

        top_n = df.head(5)
        game_count = len(projections_df)

        summary_lines = [f"**CBB Totals — Daily Summary ({today})**\n"]
        summary_lines.append(f"Games analyzed: {game_count}\n")

        over_count = (df["differential"] > 0).sum() if "differential" in df.columns else 0
        under_count = (df["differential"] < 0).sum() if "differential" in df.columns else 0
        summary_lines.append(f"OVER leans: {over_count} | UNDER leans: {under_count}\n\n")
        summary_lines.append("**Top Edges:**\n")

        for _, row in top_n.iterrows():
            home = row.get("home_team", "?")
            away = row.get("away_team", "?")
            diff = row.get("differential", 0)
            side = row.get("edge_side", "?")
            mkt = row.get("market_total", "?")
            model = row.get("ensemble_total") or row.get("model_total", "?")
            sign = "+" if diff > 0 else ""
            try:
                model_str = f"{float(model):.1f}"
            except (TypeError, ValueError):
                model_str = str(model)
            summary_lines.append(
                f"• **{away} @ {home}** | Mkt: {mkt} | Model: {model_str} | "
                f"**{side} {sign}{diff:.1f}**\n"
            )

        summary_text = "".join(summary_lines)

        # Log always
        logger.info(f"\n{summary_text}")

        if not self._can_send_discord():
            return True

        # Build embed
        embed = DiscordEmbed(
            title=f"CBB Totals — Daily Summary {today}",
            description=summary_text[:4096],   # Discord embed description limit
            color=COLOR_INFO,
        )
        embed.set_footer(text="CBB Totals Model")

        webhook = DiscordWebhook(
            url=self.webhook_url,
            username="CBB Totals Bot",
        )
        webhook.add_embed(embed)
        return self._post_webhook(webhook)

    # ── Generic notification ──────────────────────────────────────────────

    def notify(self, title: str, body: str, level: str = "info") -> None:
        """
        Generic notification (pipeline errors, job completions, etc.).

        Args:
            title: Short title string.
            body:  Longer descriptive text.
            level: 'info' | 'warning' | 'error'
        """
        color_map = {"info": COLOR_INFO, "warning": COLOR_WARN, "error": COLOR_UNDER}
        color = color_map.get(level, COLOR_INFO)

        logger.info(f"[NOTIFY:{level.upper()}] {title} — {body}")

        if not self._can_send_discord():
            return

        embed = DiscordEmbed(title=title, description=body[:4096], color=color)
        embed.set_footer(text=f"CBB Totals Model • {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        webhook = DiscordWebhook(
            url=self.webhook_url,
            content="",
            username="CBB Totals Bot",
        )
        webhook.add_embed(embed)
        self._post_webhook(webhook)
