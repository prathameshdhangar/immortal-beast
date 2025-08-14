from dataclasses import dataclass
from typing import Dict, Any, Optional
from collections import defaultdict
import json
import os
import time
import logging
from .rate_limiter import RateLimiter# Import your async RateLimiter class


@dataclass
class GuildRateLimits:
    """Rate limit configuration for a specific guild"""
    xp_cooldown: int = 60
    daily_stones_cooldown: int = 86400  # 24 hours
    beast_adoption_cooldown: int = 172800  # 48 hours
    max_catch_attempts: int = 3
    api_requests_per_minute: int = 60
    battle_timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            'xp_cooldown': self.xp_cooldown,
            'daily_stones_cooldown': self.daily_stones_cooldown,
            'beast_adoption_cooldown': self.beast_adoption_cooldown,
            'max_catch_attempts': self.max_catch_attempts,
            'api_requests_per_minute': self.api_requests_per_minute,
            'battle_timeout': self.battle_timeout
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GuildRateLimits':
        return cls(**data)


class GuildRateLimitManager:
    """Manages rate limits per guild with persistent storage"""

    def __init__(self, config_file: str = "guild_rate_limits.json"):  # Fixed __init__
        self.config_file = config_file
        self.guild_limits: Dict[int, GuildRateLimits] = {}
        self.guild_rate_limiters: Dict[int, RateLimiter] = {}
        self.user_cooldowns: Dict[int, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        self.load_config()

    def load_config(self):
        """Load guild configurations from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    for guild_id_str, limits_data in data.items():
                        guild_id = int(guild_id_str)
                        self.guild_limits[guild_id] = GuildRateLimits.from_dict(limits_data)
                        self._create_rate_limiter(guild_id)
            except Exception as e:
                logging.error(f"Failed to load guild rate limits: {e}")

    def save_config(self):
        """Save guild configurations to file"""
        try:
            data = {}
            for guild_id, limits in self.guild_limits.items():
                data[str(guild_id)] = limits.to_dict()
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save guild rate limits: {e}")

    def _create_rate_limiter(self, guild_id: int):
        """Create API rate limiter for guild"""
        limits = self.get_guild_limits(guild_id)
        self.guild_rate_limiters[guild_id] = RateLimiter(
            max_calls=limits.api_requests_per_minute, period=60)

    def get_guild_limits(self, guild_id: int) -> GuildRateLimits:
        """Get rate limits for a specific guild"""
        if guild_id not in self.guild_limits:
            self.guild_limits[guild_id] = GuildRateLimits()
            self._create_rate_limiter(guild_id)
            self.save_config()
        return self.guild_limits[guild_id]

    def update_guild_limits(self, guild_id: int, **kwargs):
        """Update rate limits for a specific guild"""
        limits = self.get_guild_limits(guild_id)
        for key, value in kwargs.items():
            if hasattr(limits, key):
                setattr(limits, key, value)

        # Recreate rate limiter if API limits changed
        if 'api_requests_per_minute' in kwargs:
            self._create_rate_limiter(guild_id)

        self.save_config()

    async def check_user_cooldown(self, guild_id: int, user_id: int, action: str) -> Optional[float]:
        """Check if user is on cooldown for specific action"""
        limits = self.get_guild_limits(guild_id)
        cooldown_map = {
            'xp': limits.xp_cooldown,
            'daily_stones': limits.daily_stones_cooldown,
            'beast_adoption': limits.beast_adoption_cooldown
        }

        if action not in cooldown_map:
            return None

        cooldown_duration = cooldown_map[action]
        last_use = self.user_cooldowns[guild_id][user_id].get(action, 0)
        time_since_last = time.time() - last_use

        if time_since_last < cooldown_duration:
            return cooldown_duration - time_since_last

        return None

    def set_user_cooldown(self, guild_id: int, user_id: int, action: str):
        """Set cooldown for user action"""
        self.user_cooldowns[guild_id][user_id][action] = time.time()

    async def get_api_rate_limiter(self, guild_id: int) -> RateLimiter:
        """Get API rate limiter for guild"""
        if guild_id not in self.guild_rate_limiters:
            self._create_rate_limiter(guild_id)
        return self.guild_rate_limiters[guild_id]
