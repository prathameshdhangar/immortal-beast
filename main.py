"""Immortal Beasts Discord Bot - Professional Edition
A comprehensive Discord bot for beast collection, battles, and trading.
"""

import asyncio
import datetime
import json
import logging
import os
import random
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union
import sqlite3
import time
import functools
import discord
from discord.ext import commands, tasks
import yaml
from pydantic import BaseModel, Field
import threading
from flask import Flask
import base64
import aiohttp
from utils.guild_rate_manager import GuildRateLimitManager
from utils.rate_limiter import RateLimiter
from cogs.rate_limit_config import RateLimitConfig


# Configuration Management
class BotConfig(BaseModel):
    """Bot configuration with validation"""
    token: str = Field(..., description="Discord bot token")
    prefix: str = Field(default="!", description="Command prefix")
    database_path: str = Field(default="data/immortal_beasts.db",
                               description="Database file path")
    backup_interval_hours: int = Field(default=6,
                                       ge=1,
                                       le=24,
                                       description="Backup interval in hours")
    backup_retention_count: int = Field(
        default=10, ge=1, le=100, description="Number of backups to retain")
    backup_max_size_mb: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum total backup size in MB")
    backup_enabled: bool = Field(
        default=True, description="Enable/disable automatic backups")
    fixed_spawn_interval_minutes: int = Field(
        default=45,
        ge=5,
        le=120,
        description=
        "Fixed spawn interval in minutes (consistent 45min spawning)")
    log_level: str = Field(default="INFO", description="Logging level")

    api_max_requests_per_minute: int = Field(
        default=60, description="API max requests per minute")
    api_retry_backoff_seconds: int = Field(
        default=2, description="API retry backoff seconds")

    # Inventory limits
    normal_user_beast_limit: int = Field(
        default=6, description="Beast limit for normal users")
    special_role_beast_limit: int = Field(
        default=8, description="Beast limit for special role users")
    personal_role_beast_limit: int = Field(
        default=10, description="Beast limit for personal role users")

    battle_channel_ids: List[int] = Field(
        default=[],
        description="List of channel IDs where battles are allowed")
    adopt_channel_id: int = Field(
        default=0, description="Channel ID where adopt commands are allowed")
    spawn_channel_id: int = Field(
        default=0,
        description="Channel ID where beasts spawn (single channel)")

    # Role IDs - UPDATED FOR MULTIPLE ROLES
    special_role_ids: List[int] = Field(
        default=[],
        description=
        "List of special role IDs for 8 beast limit and adopt legend")
    personal_role_id: int = Field(
        default=0,
        description="Personal role ID for 10 beast limit and adopt mythic")

    # XP System - MESSAGE-BASED ONLY (NO FREE XP!)
    xp_chat_channel_ids: List[int] = Field(
        default=[],
        description="List of channel IDs where beasts gain XP from messages")

    # ==========
    # MESSAGE-BASED XP - PRIMARY SYSTEM
    # ==========
    xp_per_message: int = Field(
        default=25, description="XP reward per message - MAIN XP SOURCE")
    xp_cooldown_seconds: int = Field(
        default=60, description="1 minute cooldown between XP gains")

    # Role multipliers for message XP
    message_xp_multiplier_special: float = Field(
        default=1.3, description="30% bonus for special roles")
    message_xp_multiplier_personal: float = Field(
        default=1.6, description="60% bonus for personal role")

    # ==========
    # DISABLE TIME-BASED XP COMPLETELY (NO FREE XP!)
    # ==========
    xp_per_minute_normal: int = Field(default=0,
                                      description="DISABLED - No passive XP")
    xp_per_minute_special: int = Field(default=0,
                                       description="DISABLED - No passive XP")
    xp_per_minute_personal: int = Field(default=0,
                                        description="DISABLED - No passive XP")

    # Keep these but they won't be used since time-based XP is disabled
    xp_gain_interval_seconds: int = Field(
        default=60, description="Not used in message-only system")
    user_activity_timeout_minutes: int = Field(default=5,
                                               description="Not used for XP")

    # XP Anti-Spam Settings
    min_message_length: int = Field(
        default=5, description="Minimum message length for XP")
    max_similar_messages: int = Field(
        default=2, description="Max similar messages before spam detection")
    message_frequency_limit: int = Field(
        default=6, description="Max messages per 2 minutes for XP")
    xp_spam_cooldown_minutes: int = Field(
        default=5, description="Cooldown after spam detection")

    # Starting resources
    starting_beast_stones: int = Field(
        default=1000, ge=0, description="Starting beast stones for new users")

    # Adopt cooldowns (in hours)
    adopt_cooldown_hours: int = Field(
        default=48,
        description="Adopt command cooldown in hours (2 days = 48)")

    @classmethod
    def load_from_file(cls, config_path: str = "config.yml") -> "BotConfig":
        """Load config from YAML file or environment"""
        # Check if we're in production (Render sets PORT environment variable)
        if os.getenv('PORT'):
            return cls.load_from_environment()

        # Original file loading code for local development
        if not os.path.exists(config_path):
            # Create default config with your specific IDs
            default_config = cls(
                token="YOUR_BOT_TOKEN_HERE",
                special_role_ids=[1393927051685400790, 1393094845479780426],
                personal_role_id=1393176170601775175,
                xp_chat_channel_ids=[
                    1393424880787259482, 1393626191935705198,
                    1394289930515124325, 1393163125850640414
                ],
                battle_channel_ids=[1397783271961792604, 1397783317163806730],
                adopt_channel_id=1397783378618748948,
                spawn_channel_id=1397783188394475520)
            with open(config_path, 'w') as f:
                yaml.dump(default_config.model_dump(),
                          f,
                          default_flow_style=False)
            raise FileNotFoundError(
                f"Config file created at {config_path}. Please fill in your bot token, role IDs, and XP channel IDs."
            )

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def load_from_environment(cls) -> 'BotConfig':
        """Load configuration from environment variables (for production)"""
        token = os.getenv('DISCORD_BOT_TOKEN')
        if not token:
            raise ValueError(
                "DISCORD_BOT_TOKEN environment variable is required")

        # Parse battle channel IDs from environment (comma-separated)
        battle_channel_ids = []
        battle_channels_env = os.getenv('BATTLE_CHANNEL_IDS', '')
        if battle_channels_env:
            battle_channel_ids = [
                int(channel_id.strip())
                for channel_id in battle_channels_env.split(',')
                if channel_id.strip()
            ]

        # Parse role IDs from environment (comma-separated)
        special_role_ids = []
        special_roles_env = os.getenv(
            'SPECIAL_ROLE_IDS', '1393927051685400790,1393094845479780426')
        if special_roles_env:
            special_role_ids = [
                int(role_id.strip())
                for role_id in special_roles_env.split(',') if role_id.strip()
            ]

        # Parse XP channel IDs from environment (comma-separated)
        xp_channel_ids = []
        xp_channels_env = os.getenv(
            'XP_CHANNEL_IDS',
            '1393424880787259482,1393626191935705198,1394289930515124325,1393163125850640414'
        )
        if xp_channels_env:
            xp_channel_ids = [
                int(channel_id.strip())
                for channel_id in xp_channels_env.split(',')
                if channel_id.strip()
            ]

        # Determine backup settings based on environment
        if os.getenv('PORT'):  # Production (Heroku/Railway)
            backup_retention = 3
            backup_interval = 12
            database_path = os.getenv('DATABASE_PATH',
                                      'data/immortal_beasts.db')
        else:  # Development
            backup_retention = 10
            backup_interval = 6
            database_path = '/tmp/immortal_beasts.db'  # Development path

        # FIXED: Return statement now works for both production and development
        return cls(
            token=token,
            prefix=os.getenv('BOT_PREFIX', '!'),
            database_path=
            database_path,  # Now properly set for both environments
            special_role_ids=special_role_ids,
            personal_role_id=int(
                os.getenv('PERSONAL_ROLE_ID', '1393176170601775175')),
            xp_chat_channel_ids=xp_channel_ids,
            log_level=os.getenv('LOG_LEVEL', 'INFO'),

            # TIME-BASED XP SETTINGS (DISABLED)
            xp_per_minute_normal=0,
            xp_per_minute_special=0,
            xp_per_minute_personal=0,
            xp_gain_interval_seconds=int(
                os.getenv('XP_GAIN_INTERVAL_SECONDS', '60')),
            user_activity_timeout_minutes=int(
                os.getenv('USER_ACTIVITY_TIMEOUT_MINUTES', '5')),

            # MESSAGE-BASED XP SETTINGS
            xp_per_message=int(os.getenv('XP_PER_MESSAGE', '25')),
            xp_cooldown_seconds=int(os.getenv('XP_COOLDOWN_SECONDS', '60')),
            message_xp_multiplier_special=float(
                os.getenv('MESSAGE_XP_MULTIPLIER_SPECIAL', '1.3')),
            message_xp_multiplier_personal=float(
                os.getenv('MESSAGE_XP_MULTIPLIER_PERSONAL', '1.6')),

            # BACKUP SETTINGS
            backup_interval_hours=int(
                os.getenv('BACKUP_INTERVAL_HOURS', str(backup_interval))),
            backup_retention_count=int(
                os.getenv('BACKUP_RETENTION_COUNT', str(backup_retention))),
            backup_max_size_mb=int(os.getenv('BACKUP_MAX_SIZE_MB', '100')),
            backup_enabled=os.getenv('BACKUP_ENABLED',
                                     'true').lower() == 'true',

            # SPAWN AND RESOURCE SETTINGS
            fixed_spawn_interval_minutes=int(
                os.getenv('FIXED_SPAWN_INTERVAL_MINUTES', '45')),
            starting_beast_stones=int(
                os.getenv('STARTING_BEAST_STONES', '1000')),
            adopt_cooldown_hours=int(os.getenv('ADOPT_COOLDOWN_HOURS', '48')),

            # CHANNEL SETTINGS
            battle_channel_ids=battle_channel_ids,
            adopt_channel_id=int(
                os.getenv('ADOPT_CHANNEL_ID', '1397783378618748948')),
            spawn_channel_id=int(
                os.getenv('SPAWN_CHANNEL_ID', '1397783188394475520')),

            # API SETTINGS
            api_max_requests_per_minute=int(
                os.getenv('API_MAX_REQUESTS_PER_MINUTE', '60')),
            api_retry_backoff_seconds=int(
                os.getenv('API_RETRY_BACKOFF_SECONDS', '2')))


# Enums and Constants


class BeastRarity(Enum):
    """Beast rarity levels"""
    COMMON = 1
    UNCOMMON = 2
    RARE = 3
    EPIC = 4
    LEGENDARY = 5
    MYTHIC = 6

    @property
    def color(self) -> int:
        """Discord embed color for this rarity"""
        colors = {
            1: 0x808080,  # Gray
            2: 0x00FF00,  # Green
            3: 0x0080FF,  # Blue
            4: 0x8000FF,  # Purple
            5: 0xFF8000,  # Orange
            6: 0xFF0000,  # Red
        }
        return colors[self.value]

    @property
    def catch_rate(self) -> int:
        """Base catch rate percentage"""
        rates = {1: 90, 2: 80, 3: 65, 4: 30, 5: 2, 6: 0.1}
        return rates[self.value]

    @property
    def emoji(self) -> str:
        """Star emoji representation"""
        return '⭐' * self.value


class BattleResult(Enum):
    """Battle outcome types"""
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class UserRole(Enum):
    """User role types for beast limits"""
    NORMAL = "normal"
    SPECIAL = "special"
    PERSONAL = "personal"


# Move this OUTSIDE the BeastStats class, around line 400-500 (after BeastStats class definition)
def calculate_enhanced_battle_xp(winner_beast,
                                 loser_beast,
                                 battle_turns,
                                 is_winner=True):
    """Enhanced XP calculation considering rarity and other factors"""

    # Base XP scales with opponent's rarity
    rarity_multipliers = {
        BeastRarity.COMMON: 1.0,
        BeastRarity.UNCOMMON: 1.2,
        BeastRarity.RARE: 1.4,
        BeastRarity.EPIC: 1.6,
        BeastRarity.LEGENDARY: 1.8,
        BeastRarity.MYTHIC: 2.0
    }

    opponent_rarity = loser_beast.rarity if is_winner else winner_beast.rarity
    base_xp = int(40 * rarity_multipliers[opponent_rarity])

    if is_winner:
        victory_bonus = 20

        # David vs Goliath bonus (defeating higher rarity)
        if loser_beast.rarity.value > winner_beast.rarity.value:
            underdog_bonus = (loser_beast.rarity.value -
                              winner_beast.rarity.value) * 15
        else:
            underdog_bonus = 0

        # Level difference bonus
        level_diff = max(0, loser_beast.stats.level - winner_beast.stats.level)
        level_bonus = level_diff * 3

        # Battle intensity bonus
        intensity_bonus = min(battle_turns * 1, 15)

        total_xp = base_xp + victory_bonus + underdog_bonus + level_bonus + intensity_bonus

    else:
        # Loser rewards
        participation_xp = base_xp // 3

        # Bonus for lasting against stronger opponent
        if winner_beast.rarity.value > loser_beast.rarity.value:
            survival_bonus = (winner_beast.rarity.value -
                              loser_beast.rarity.value) * 8
        else:
            survival_bonus = 0

        # Small level difference consolation
        level_diff = max(0, winner_beast.stats.level - loser_beast.stats.level)
        level_consolation = level_diff * 1

        total_xp = participation_xp + survival_bonus + level_consolation

    return min(total_xp, 150)  # Higher cap for enhanced system


def has_sufficient_bond(beast) -> bool:
    """Check if beast has been owned for at least 12 hours"""
    if not beast.caught_at:
        return False
    owned_time = datetime.datetime.now() - beast.caught_at
    return owned_time.total_seconds() >= (12 * 3600)  # 12 hours in seconds


def calculate_percentage_stat_boosts(sacrificed_beast):
    """Calculate stat boosts based on percentage of sacrificed beast's stats"""
    rarity_percent_map = {
        1: 0.05,  # Common (5%)
        2: 0.07,  # Uncommon (7%)
        3: 0.09,  # Rare (9%)
        4: 0.11,  # Epic (11%)
        5: 0.13,  # Legendary (13%)
        6: 0.15  # Mythic (15%)
    }

    percent = rarity_percent_map.get(sacrificed_beast.rarity.value, 0.05)
    stats = sacrificed_beast.stats

    return {
        'hp': max(1, int(stats.max_hp * percent)),
        'attack': max(1, int(stats.attack * percent)),
        'defense': max(1, int(stats.defense * percent)),
        'speed': max(1, int(stats.speed * percent))
    }

def get_base_stats_for_rarity(rarity):
    """Get base stats for each rarity - adjust these if needed"""
    base_stats = {
        BeastRarity.COMMON: {'hp': 100, 'attack': 20, 'defense': 15, 'speed': 25},
        BeastRarity.UNCOMMON: {'hp': 130, 'attack': 30, 'defense': 20, 'speed': 30},
        BeastRarity.RARE: {'hp': 180, 'attack': 40, 'defense': 25, 'speed': 35},
        BeastRarity.EPIC: {'hp': 300, 'attack': 70, 'defense': 35, 'speed': 40},
        BeastRarity.LEGENDARY: {'hp': 500, 'attack': 120, 'defense': 50, 'speed': 45},
        BeastRarity.MYTHIC: {'hp': 800, 'attack': 180, 'defense': 70, 'speed': 50}
    }
    return base_stats[rarity]


async def fix_all_beast_stats_preserve_boosts(bot):
    """Recalculate beast stats while preserving sacrifice/spar boosts"""
    print("🔄 Starting enhanced beast stat migration (preserving boosts)...")

    try:
        conn = bot.db._get_connection()
        user_cursor = conn.execute('SELECT user_id FROM users')
        all_users = user_cursor.fetchall()
        conn.close()

        total_beasts_fixed = 0
        users_processed = 0

        for user_row in all_users:
            try:
                user_id = user_row['user_id']
                user_beasts = await bot.db.get_user_beasts(user_id)

                for beast_id, beast in user_beasts:
                    # ✅ PRESERVE current stats (includes all boosts)
                    preserved_stats = {
                        'hp': beast.stats.hp,
                        'max_hp': beast.stats.max_hp, 
                        'attack': beast.stats.attack,
                        'defense': beast.stats.defense,
                        'speed': beast.stats.speed
                    }

                    target_level = beast.stats.level
                    current_exp = beast.stats.exp

                    # Reset to base stats
                    base_stats = get_base_stats_for_rarity(beast.rarity)
                    beast.stats.hp = base_stats['hp']
                    beast.stats.max_hp = base_stats['hp']
                    beast.stats.attack = base_stats['attack'] 
                    beast.stats.defense = base_stats['defense']
                    beast.stats.speed = base_stats['speed']
                    beast.stats.level = 1
                    beast.stats.exp = 0

                    # Simulate proper level-ups for base stats
                    for level in range(1, target_level):
                        leveled_up, bonus_level, stat_gains = beast.stats.level_up(beast.rarity)
                        if not leveled_up:
                            break

                    # ✅ CALCULATE and PRESERVE boosts
                    boosts = {}
                    for stat in ['max_hp', 'attack', 'defense', 'speed']:
                        base_after_levelup = getattr(beast.stats, stat)
                        preserved_value = preserved_stats[stat]
                        boost = preserved_value - base_after_levelup
                        boosts[stat] = max(0, boost)  # Only positive boosts

                    # HP boost calculation (special handling)
                    hp_boost = preserved_stats['hp'] - beast.stats.max_hp
                    boosts['hp'] = max(0, hp_boost)

                    # ✅ APPLY preserved boosts on top of base stats
                    beast.stats.max_hp += boosts['max_hp']
                    beast.stats.hp = beast.stats.max_hp + boosts['hp']  # Full HP + any HP boosts
                    beast.stats.attack += boosts['attack']
                    beast.stats.defense += boosts['defense']
                    beast.stats.speed += boosts['speed']

                    # Restore current XP
                    beast.stats.exp = current_exp

                    await bot.db.update_beast(beast_id, beast)
                    total_beasts_fixed += 1

                    # Log significant boosts for verification
                    total_boost = sum(boosts.values())
                    if total_boost > 50:  # Significant boost detected
                        print(f"🎯 Beast #{beast_id} ({beast.name}) preserved {total_boost:.0f} total stat boosts")

                users_processed += 1
                if users_processed % 10 == 0:
                    print(f"📊 Processed {users_processed} users, {total_beasts_fixed} beasts fixed")

            except Exception as e:
                print(f"❌ Error processing user {user_row['user_id']}: {e}")
                continue

        print(f"✅ Enhanced stat migration complete! {total_beasts_fixed} beasts fixed with boosts preserved")
        return True

    except Exception as e:
        print(f"❌ Enhanced stat migration failed: {e}")
        return False


# Data Models
@dataclass
class BeastStats:
    """Beast statistics"""
    hp: int
    max_hp: int
    attack: int
    defense: int = 0
    speed: int = 0
    level: int = 1
    exp: int = 0

    def get_level_up_requirements(self, rarity: BeastRarity) -> int:
        """Get XP required for next level based on rarity - NEW FORMULA"""
        base_requirements = {
            BeastRarity.COMMON: 100,
            BeastRarity.UNCOMMON: 150,
            BeastRarity.RARE: 200,
            BeastRarity.EPIC: 300,
            BeastRarity.LEGENDARY: 500,
            BeastRarity.MYTHIC: 800
        }
        base_req = base_requirements[rarity]
        return int(base_req * (1.5**(self.level - 1)))  # NEW: 1.5x progression

    def get_stat_gains(self,
                       rarity: BeastRarity) -> Dict[str, Tuple[int, int]]:
        """Get stat gain ranges based on rarity"""
        stat_ranges = {
            BeastRarity.COMMON: {
                'hp': (8, 15),
                'attack': (2, 5),
                'defense': (1, 3),
                'speed': (1, 3)
            },
            BeastRarity.UNCOMMON: {
                'hp': (10, 18),
                'attack': (3, 6),
                'defense': (2, 4),
                'speed': (1, 4)
            },
            BeastRarity.RARE: {
                'hp': (12, 22),
                'attack': (4, 8),
                'defense': (2, 5),
                'speed': (2, 5)
            },
            BeastRarity.EPIC: {
                'hp': (15, 28),
                'attack': (5, 10),
                'defense': (3, 7),
                'speed': (3, 7)
            },
            BeastRarity.LEGENDARY: {
                'hp': (20, 35),
                'attack': (7, 14),
                'defense': (4, 9),
                'speed': (4, 9)
            },
            BeastRarity.MYTHIC: {
                'hp': (25, 45),
                'attack': (10, 18),
                'defense': (6, 12),
                'speed': (6, 12)
            }
        }
        return stat_ranges[rarity]

    def get_bonus_stat_ranges(
            self, rarity: BeastRarity) -> Dict[str, Tuple[int, int]]:
        """Get bonus stat ranges for every 5 levels"""
        bonus_ranges = {
            BeastRarity.COMMON: {
                'hp': (5, 10),
                'attack': (1, 3),
                'defense': (1, 2),
                'speed': (1, 2)
            },
            BeastRarity.UNCOMMON: {
                'hp': (8, 15),
                'attack': (2, 4),
                'defense': (1, 3),
                'speed': (1, 3)
            },
            BeastRarity.RARE: {
                'hp': (12, 20),
                'attack': (3, 6),
                'defense': (2, 4),
                'speed': (2, 4)
            },
            BeastRarity.EPIC: {
                'hp': (15, 25),
                'attack': (4, 8),
                'defense': (3, 6),
                'speed': (3, 6)
            },
            BeastRarity.LEGENDARY: {
                'hp': (20, 35),
                'attack': (6, 12),
                'defense': (4, 8),
                'speed': (4, 8)
            },
            BeastRarity.MYTHIC: {
                'hp': (30, 50),
                'attack': (8, 16),
                'defense': (6, 12),
                'speed': (6, 12)
            }
        }
        return bonus_ranges[rarity]

    def level_up(self,
                 rarity: BeastRarity) -> Tuple[bool, bool, Dict[str, int]]:
        """Level up the beast if enough experience"""
        required_exp = self.get_level_up_requirements(rarity)
        if self.exp < required_exp:
            return False, False, {}

        self.level += 1
        self.exp -= required_exp

        stat_ranges = self.get_stat_gains(rarity)
        hp_gain = random.randint(*stat_ranges['hp'])
        attack_gain = random.randint(*stat_ranges['attack'])
        defense_gain = random.randint(*stat_ranges['defense'])
        speed_gain = random.randint(*stat_ranges['speed'])

        self.max_hp += hp_gain
        self.hp = self.max_hp
        self.attack += attack_gain
        self.defense += defense_gain
        self.speed += speed_gain

        stat_gains = {
            'hp': hp_gain,
            'attack': attack_gain,
            'defense': defense_gain,
            'speed': speed_gain
        }

        bonus_level = self.level % 5 == 0
        if bonus_level:
            bonus_ranges = self.get_bonus_stat_ranges(rarity)
            bonus_hp = random.randint(*bonus_ranges['hp'])
            bonus_attack = random.randint(*bonus_ranges['attack'])
            bonus_defense = random.randint(*bonus_ranges['defense'])
            bonus_speed = random.randint(*bonus_ranges['speed'])

            self.max_hp += bonus_hp
            self.hp = self.max_hp
            self.attack += bonus_attack
            self.defense += bonus_defense
            self.speed += bonus_speed

            stat_gains.update({
                'bonus_hp': bonus_hp,
                'bonus_attack': bonus_attack,
                'bonus_defense': bonus_defense,
                'bonus_speed': bonus_speed
            })

        return True, bonus_level, stat_gains

    def add_exp(
            self, amount: int,
            rarity: BeastRarity) -> List[Tuple[bool, bool, Dict[str, int]]]:
        """Add experience and handle multiple level ups"""
        self.exp += amount
        level_ups = []

        while True:
            leveled_up, bonus_level, stat_gains = self.level_up(rarity)
            if not leveled_up:
                break
            level_ups.append((leveled_up, bonus_level, stat_gains))
            if len(level_ups) >= 10:
                break

        return level_ups

    def get_total_exp_value(self, rarity: BeastRarity) -> int:
        """Calculate total XP value of this beast for sacrifice"""
        total_xp = self.exp
        for level in range(1, self.level):
            temp_stats = BeastStats(0, 0, 0, 0, 0, level, 0)
            total_xp += temp_stats.get_level_up_requirements(rarity)
        return total_xp

    def heal(self, amount: Optional[int] = None) -> int:
        """Heal the beast, return amount healed"""
        if amount is None:
            amount = self.max_hp
        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - old_hp


# ✅ FIXED: Beast class moved to module level (NOT nested in BeastTemplate)
@dataclass
class Beast:
    """Main beast class"""
    name: str
    rarity: BeastRarity
    tendency: str
    location: str
    stats: BeastStats
    description: str = ""
    caught_at: Optional[datetime.datetime] = None
    owner_id: Optional[int] = None
    unique_id: Optional[str] = None

    def __post_init__(self):
        if self.caught_at is None:
            self.caught_at = datetime.datetime.now()
        if self.unique_id is None:
            self.unique_id = f"{self.name}_{self.caught_at.timestamp()}_{random.randint(1000, 9999)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'rarity': self.rarity.value,
            'tendency': self.tendency,
            'location': self.location,
            'stats': asdict(self.stats),
            'description': self.description,
            'caught_at':
            self.caught_at.isoformat() if self.caught_at else None,
            'owner_id': self.owner_id,
            'unique_id': self.unique_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Beast':
        """Create beast from dictionary"""
        stats = BeastStats(**data['stats'])

        # Handle caught_at safely
        caught_at = None
        if 'caught_at' in data and data['caught_at']:
            caught_at = datetime.datetime.fromisoformat(data['caught_at'])

        return cls(name=data['name'],
                   rarity=BeastRarity(data['rarity']),
                   tendency=data['tendency'],
                   location=data['location'],
                   stats=stats,
                   description=data.get('description', ''),
                   caught_at=caught_at,
                   owner_id=data.get('owner_id'),
                   unique_id=data.get('unique_id'))

    # ✅ FIXED: power_level property moved to Beast class where it belongs
    @property
    def power_level(self) -> int:
        """Calculate overall power level"""
        return (self.stats.hp + self.stats.attack + self.stats.defense +
                self.stats.speed) * self.stats.level


def get_user_message_xp_amount(member: Optional[discord.Member],
                               config: BotConfig) -> int:
    """Calculate XP amount for messages based on user's roles"""
    base_xp = config.xp_per_message

    if not member:
        return base_xp

    user_role_ids = [role.id for role in member.roles]

    # Personal role gets highest bonus (60%)
    if config.personal_role_id in user_role_ids:
        return int(base_xp * config.message_xp_multiplier_personal)

    # Special roles get 30% bonus
    if any(role_id in user_role_ids for role_id in config.special_role_ids):
        return int(base_xp * config.message_xp_multiplier_special)

    # Normal users get base amount
    return base_xp


def get_boosted_xp_amount(base_xp: int, user: 'User') -> int:
    """Apply XP boost multiplier if active"""
    if (user.xp_boost_expires
            and datetime.datetime.now() < user.xp_boost_expires):
        return int(base_xp * 1.5)  # 50% boost
    return base_xp


@dataclass
class BeastTemplate:
    """Template for creating new beasts"""
    name: str
    rarity: BeastRarity
    tendency: str
    location: str
    base_hp_range: Tuple[int, int]
    base_attack_range: Tuple[int, int]
    description: str = ""

    def create_beast(
            self
    ) -> Beast:  # ✅ FIXED: No quotes needed, Beast is defined above
        """Create a new beast instance from this template"""
        base_hp = random.randint(*self.base_hp_range)
        base_attack = random.randint(*self.base_attack_range)

        stats = BeastStats(hp=base_hp,
                           max_hp=base_hp,
                           attack=base_attack,
                           defense=self.rarity.value * 5 +
                           random.randint(1, 10),
                           speed=random.randint(10, 50))

        return Beast(name=self.name,
                     rarity=self.rarity,
                     tendency=self.tendency,
                     location=self.location,
                     stats=stats,
                     description=self.description)


@dataclass
class User:
    """User data model"""
    user_id: int
    username: str
    spirit_stones: int = 1000
    last_daily: Optional[datetime.datetime] = None
    last_adopt: Optional[datetime.datetime] = None
    last_xp_gain: Optional[datetime.datetime] = None
    active_beast_id: Optional[int] = None
    total_catches: int = 0
    total_battles: int = 0
    wins: int = 0
    losses: int = 0
    battle_xp_day: Optional[str] = None
    battle_xp_count: int = 0
    created_at: Optional[datetime.datetime] = None
    has_used_adopt_legend: bool = False
    has_used_adopt_mythic: bool = False
    last_feed: Optional[datetime.datetime] = None
    last_groom: Optional[datetime.datetime] = None
    last_first_win: Optional[datetime.datetime] = None
    xp_boost_expires: Optional[datetime.datetime] = None
    spar_sessions_today: int = 0
    spar_day_date: Optional[str] = None
    sacrifice_day_date: Optional[str] = None
    sacrifice_count_today: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()

    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_battles == 0:
            return 0.0
        return (self.wins / self.total_battles) * 100

    def can_gain_xp(self, cooldown_seconds: int) -> bool:
        """Check if user can gain XP (cooldown check)"""
        if not self.last_xp_gain:
            return True
        time_since_last = datetime.datetime.now() - self.last_xp_gain
        return time_since_last.total_seconds() >= cooldown_seconds

    def can_spar(self) -> bool:
        """Check if user can spar today (max 15 sessions per day)"""
        today = datetime.datetime.now().date().isoformat()
        if self.spar_day_date != today:
            return True
        return self.spar_sessions_today < 15

    def record_spar(self):
        """Record a spar session"""
        today = datetime.datetime.now().date().isoformat()
        if self.spar_day_date != today:
            self.spar_day_date = today
            self.spar_sessions_today = 0
        self.spar_sessions_today += 1

    def can_get_stat_boosts(self) -> bool:
        """Check if user can receive stat boosts from sacrifice today"""
        today = datetime.datetime.now().date().isoformat()
        if self.sacrifice_day_date != today:
            self.sacrifice_day_date = today
            self.sacrifice_count_today = 0
        return self.sacrifice_count_today < 5

    def record_sacrifice_with_stats(self):
        """Record a sacrifice that gave stat boosts"""
        today = datetime.datetime.now().date().isoformat()
        if self.sacrifice_day_date != today:
            self.sacrifice_day_date = today
            self.sacrifice_count_today = 0
        self.sacrifice_count_today += 1


# Database Layer
class DatabaseInterface(ABC):
    """Abstract database interface"""

    @abstractmethod
    async def get_user(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    async def create_user(self, user: User) -> bool:
        pass

    @abstractmethod
    async def update_user(self, user: User) -> bool:
        pass

    @abstractmethod
    async def add_beast(self, beast: Beast) -> int:
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the database (create tables, setup schema, etc.)"""
        pass

    @abstractmethod
    async def get_user_beasts(
            self,
            user_id: int,
            limit: Optional[int] = None) -> List[Tuple[int, Beast]]:  # ✅ Fixed
        pass

    @abstractmethod
    async def update_beast(self, beast_id: int, beast: Beast) -> bool:
        pass

    @abstractmethod
    async def delete_beast(self, beast_id: int) -> bool:
        pass


class SQLiteDatabase(DatabaseInterface):
    """SQLite database implementation"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as e:
            print(
                f"Warning: Could not create directory {self.db_path.parent}: {e}"
            )
            # The directory might already exist or be created by the platform
            pass

    async def initialize(self) -> None:
        """Initialize database schema with proper tables"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL,
                    spirit_stones INTEGER DEFAULT 1000,
                    last_daily TIMESTAMP,
                    last_adopt TIMESTAMP,
                    last_xp_gain TIMESTAMP,
                    active_beast_id INTEGER,
                    total_catches INTEGER DEFAULT 0,
                    total_battles INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    has_used_adopt_legend BOOLEAN DEFAULT FALSE,
                    has_used_adopt_mythic BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    battle_xp_day TEXT,
                    battle_xp_count INTEGER DEFAULT 0,
                    last_feed TIMESTAMP,
                    last_groom TIMESTAMP,
                    last_first_win TIMESTAMP,
                    xp_boost_expires TIMESTAMP,
                    spar_sessions_today INTEGER DEFAULT 0,
                    spar_day_date TEXT,
                    sacrifice_day_date TEXT,
                    sacrifice_count_today INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS beasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id INTEGER NOT NULL,
                    beast_data TEXT NOT NULL,
                    is_favorite BOOLEAN DEFAULT FALSE,
                    caught_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (owner_id) REFERENCES users (user_id)
                );

                CREATE TABLE IF NOT EXISTS battle_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user1_id INTEGER NOT NULL,
                    user2_id INTEGER NOT NULL,
                    winner_id INTEGER,
                    battle_data TEXT,
                    battle_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user1_id) REFERENCES users (user_id),
                    FOREIGN KEY (user2_id) REFERENCES users (user_id)
                );

                -- NEW: Training cooldowns table
                CREATE TABLE IF NOT EXISTS training_cooldowns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    beast_id INTEGER NOT NULL,
                    last_training_time TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE(user_id, beast_id)
                );

                CREATE INDEX IF NOT EXISTS idx_beasts_owner ON beasts(owner_id);
                CREATE INDEX IF NOT EXISTS idx_battle_history_users ON battle_history(user1_id, user2_id);
                CREATE INDEX IF NOT EXISTS idx_users_active_beast ON users(active_beast_id);
                CREATE INDEX IF NOT EXISTS idx_training_cooldowns_user ON training_cooldowns(user_id);
                CREATE INDEX IF NOT EXISTS idx_training_cooldowns_beast ON training_cooldowns(user_id, beast_id);
            """)

            # Add new columns if they don't exist (safe migration)
            migration_columns = [
                "ALTER TABLE users ADD COLUMN battle_xp_day TEXT",
                "ALTER TABLE users ADD COLUMN battle_xp_count INTEGER DEFAULT 0",
                "ALTER TABLE users ADD COLUMN last_feed TIMESTAMP",
                "ALTER TABLE users ADD COLUMN last_groom TIMESTAMP",
                "ALTER TABLE users ADD COLUMN last_first_win TIMESTAMP",
                "ALTER TABLE users ADD COLUMN xp_boost_expires TIMESTAMP",
                "ALTER TABLE users ADD COLUMN spar_sessions_today INTEGER DEFAULT 0",
                "ALTER TABLE users ADD COLUMN spar_day_date TEXT",
                "ALTER TABLE users ADD COLUMN sacrifice_day_date TEXT",
                "ALTER TABLE users ADD COLUMN sacrifice_count_today INTEGER DEFAULT 0"
            ]

            for migration in migration_columns:
                try:
                    conn.execute(migration)
                except Exception:
                    pass  # Column already exists

            conn.commit()
        finally:
            conn.close()

    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        conn = self._get_connection()
        try:
            cursor = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                # Helper function to safely get values
                def safe_get(field_name, default=None):
                    try:
                        return row[field_name]
                    except (KeyError, IndexError):
                        return default

                def safe_datetime(field_name):
                    try:
                        value = row[field_name]
                        return datetime.datetime.fromisoformat(value) if value else None
                    except (KeyError, IndexError):
                        return None

                return User(
                    user_id=row['user_id'],
                    username=row['username'],
                    spirit_stones=row['spirit_stones'],
                    last_daily=safe_datetime('last_daily'),
                    last_adopt=safe_datetime('last_adopt'),
                    last_xp_gain=safe_datetime('last_xp_gain'),
                    active_beast_id=safe_get('active_beast_id'),
                    total_catches=row['total_catches'],
                    total_battles=row['total_battles'],
                    wins=row['wins'],
                    losses=row['losses'],
                    has_used_adopt_legend=bool(safe_get('has_used_adopt_legend', False)),
                    has_used_adopt_mythic=bool(safe_get('has_used_adopt_mythic', False)),
                    created_at=datetime.datetime.fromisoformat(row['created_at']),

                    # Optional fields with safe access
                    battle_xp_day=safe_get('battle_xp_day'),
                    battle_xp_count=safe_get('battle_xp_count', 0),
                    last_feed=safe_datetime('last_feed'),
                    last_groom=safe_datetime('last_groom'),
                    last_first_win=safe_datetime('last_first_win'),
                    xp_boost_expires=safe_datetime('xp_boost_expires'),
                    spar_sessions_today=safe_get('spar_sessions_today', 0),
                    spar_day_date=safe_get('spar_day_date'),
                    sacrifice_day_date=safe_get('sacrifice_day_date'),
                    sacrifice_count_today=safe_get('sacrifice_count_today', 0)
                )
            return None
        finally:
            conn.close()

    async def create_user(self, user: User) -> bool:
        """Create new user"""
        try:
            conn = self._get_connection()
            try:
                # ✅ FIXED: Handle None created_at safely
                created_at_str = user.created_at.isoformat(
                ) if user.created_at else datetime.datetime.now().isoformat()

                conn.execute(
                    """
                    INSERT OR IGNORE INTO users 
                    (user_id, username, spirit_stones, total_catches, total_battles, wins, losses, 
                     has_used_adopt_legend, has_used_adopt_mythic, created_at, battle_xp_day, battle_xp_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user.user_id, user.username, user.spirit_stones,
                      user.total_catches, user.total_battles, user.wins,
                      user.losses, user.has_used_adopt_legend,
                      user.has_used_adopt_mythic, created_at_str,
                      user.battle_xp_day, user.battle_xp_count))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logging.error(f"Failed to create user {user.user_id}: {e}")
            return False

    async def update_user(self, user: User) -> bool:
        """Update existing user"""
        try:
            conn = self._get_connection()
            try:
                # First, update core fields that should always exist
                conn.execute(
                    """
                    UPDATE users SET 
                    username=?, spirit_stones=?, last_daily=?, last_adopt=?, last_xp_gain=?,
                    active_beast_id=?, total_catches=?, total_battles=?, wins=?, losses=?,
                    has_used_adopt_legend=?, has_used_adopt_mythic=?
                    WHERE user_id=?
                    """,
                    (user.username, user.spirit_stones,
                     user.last_daily.isoformat() if user.last_daily else None,
                     user.last_adopt.isoformat() if user.last_adopt else None,
                     user.last_xp_gain.isoformat() if user.last_xp_gain else None, 
                     user.active_beast_id, user.total_catches,
                     user.total_battles, user.wins, user.losses,
                     user.has_used_adopt_legend, user.has_used_adopt_mythic,
                     user.user_id))

                # Then try to update optional fields one by one
                optional_fields = [
                    ('battle_xp_day', user.battle_xp_day),
                    ('battle_xp_count', user.battle_xp_count),
                    ('last_feed', user.last_feed.isoformat() if user.last_feed else None),
                    ('last_groom', user.last_groom.isoformat() if user.last_groom else None),
                    ('last_first_win', user.last_first_win.isoformat() if user.last_first_win else None),
                    ('xp_boost_expires', user.xp_boost_expires.isoformat() if user.xp_boost_expires else None),
                    ('spar_sessions_today', user.spar_sessions_today),
                    ('spar_day_date', user.spar_day_date),
                    ('sacrifice_day_date', user.sacrifice_day_date),
                    ('sacrifice_count_today', user.sacrifice_count_today)
                ]

                for field_name, field_value in optional_fields:
                    try:
                        conn.execute(f"UPDATE users SET {field_name}=? WHERE user_id=?", 
                                   (field_value, user.user_id))
                    except Exception as e:
                        # Log but don't fail - column might not exist yet
                        logging.warning(f"Failed to update {field_name} for user {user.user_id}: {e}")
                        continue

                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logging.error(f"Failed to update user {user.user_id}: {e}")
            return False

    async def add_beast(self, beast: Beast) -> int:
        """Add beast to database"""
        conn = self._get_connection()
        try:
            # ✅ FIXED: Handle None caught_at safely
            caught_at_str = beast.caught_at.isoformat(
            ) if beast.caught_at else datetime.datetime.now().isoformat()

            cursor = conn.execute(
                """
                INSERT INTO beasts (owner_id, beast_data, caught_at)
                VALUES (?, ?, ?)
            """, (beast.owner_id, json.dumps(beast.to_dict()), caught_at_str))
            conn.commit()

            # ✅ FIXED: Handle None lastrowid
            beast_id = cursor.lastrowid
            if beast_id is None:
                raise RuntimeError("Failed to get beast ID after insertion")
            return beast_id
        finally:
            conn.close()

    async def get_user_beasts(
        self,
        user_id: int,
        limit: Optional[int] = None
    ) -> List[Tuple[int, Beast]]:  # ✅ FIXED: Match abstract method signature
        """Get user's beasts"""
        query = "SELECT id, beast_data FROM beasts WHERE owner_id = ? ORDER BY caught_at DESC"
        params = [user_id]

        # ✅ FIXED: Handle None limit properly
        if limit is not None:  # Changed from "if limit:" to handle limit=0 correctly
            query += " LIMIT ?"
            params.append(limit)

        conn = self._get_connection()
        try:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            beasts = []
            for row in rows:
                try:
                    beast_data = json.loads(row['beast_data'])
                    beast = Beast.from_dict(beast_data)
                    beasts.append((row['id'], beast))
                except Exception as e:
                    logging.error(f"Failed to load beast {row['id']}: {e}")
            return beasts
        finally:
            conn.close()

    async def update_beast(self, beast_id: int, beast: Beast) -> bool:
        """Update beast data"""
        try:
            conn = self._get_connection()
            try:
                conn.execute("UPDATE beasts SET beast_data = ? WHERE id = ?",
                             (json.dumps(beast.to_dict()), beast_id))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logging.error(f"Failed to update beast {beast_id}: {e}")
            return False

    async def delete_beast(self, beast_id: int) -> bool:
        """Delete beast"""
        try:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM beasts WHERE id = ?", (beast_id, ))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logging.error(f"Failed to delete beast {beast_id}: {e}")
            return False

    async def backup_database(self,
                              backup_dir: str = "backups",
                              keep_count: int = 10) -> Optional[str]:
        """Create database backup with automatic cleanup"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"backup_{timestamp}.db"

        try:
            # Create new backup
            shutil.copy2(self.db_path, backup_file)
            logging.info(f"Database backup created: {backup_file}")

            # Clean up old backups - keep only latest N
            await self._cleanup_old_backups(backup_path, keep_count)

            return str(backup_file)
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return None

    async def _cleanup_old_backups(self, backup_path: Path, keep_count: int):
        """Remove old backup files, keeping only the latest ones"""
        try:
            # Get all backup files sorted by creation time (newest first)
            backup_files = sorted(backup_path.glob("backup_*.db"),
                                  key=lambda p: p.stat().st_mtime,
                                  reverse=True)

            # Remove old backups beyond keep_count
            removed_count = 0
            for old_backup in backup_files[keep_count:]:
                old_backup.unlink()
                removed_count += 1
                logging.info(f"Removed old backup: {old_backup}")

            if removed_count > 0:
                logging.info(f"Cleaned up {removed_count} old backup files")

        except Exception as e:
            logging.error(f"Backup cleanup failed: {e}")

    def get_storage_usage(self, backup_dir: str = "backups") -> Dict[str, Any]:
        """Get current backup storage usage"""
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'oldest_backup': None,
                'newest_backup': None
            }

        backup_files = list(backup_path.glob("backup_*.db"))
        total_size = sum(f.stat().st_size for f in backup_files)

        return {
            'total_files':
            len(backup_files),
            'total_size_mb':
            total_size / (1024 * 1024),
            'oldest_backup':
            min(backup_files, key=lambda f: f.stat().st_mtime)
            if backup_files else None,
            'newest_backup':
            max(backup_files, key=lambda f: f.stat().st_mtime)
            if backup_files else None,
        }

    async def get_training_cooldown(
            self, user_id: int, beast_id: int) -> Optional[datetime.datetime]:
        """Get last training time for specific user and beast"""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                'SELECT last_training_time FROM training_cooldowns WHERE user_id = ? AND beast_id = ?',
                (user_id, beast_id))
            row = cursor.fetchone()
            if row and row['last_training_time']:
                return datetime.datetime.fromisoformat(
                    row['last_training_time'])
            return None
        finally:
            conn.close()

    async def set_training_cooldown(self, user_id: int, beast_id: int,
                                    training_time: datetime.datetime) -> bool:
        """Set training cooldown for specific user and beast"""
        try:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO training_cooldowns (user_id, beast_id, last_training_time)
                    VALUES (?, ?, ?)
                """, (user_id, beast_id, training_time.isoformat()))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logging.error(f"Failed to set training cooldown: {e}")
            return False


# Battle System and Template Manager


class BattleEngine:
    """Handles beast battles"""

    @staticmethod
    def calculate_damage(attacker: Beast, defender: Beast) -> int:
        """Calculate damage dealt"""
        base_damage = attacker.stats.attack
        defense_reduction = defender.stats.defense * 0.5
        damage_variance = random.randint(-10, 10)
        final_damage = max(
            1, int(base_damage - defense_reduction + damage_variance))
        return final_damage

    @staticmethod
    def determine_turn_order(beast1: Beast,
                             beast2: Beast) -> Tuple[Beast, Beast]:
        """Determine which beast goes first based on speed"""
        if beast1.stats.speed > beast2.stats.speed:
            return beast1, beast2
        elif beast2.stats.speed > beast1.stats.speed:
            return beast2, beast1
        else:
            return random.choice([(beast1, beast2), (beast2, beast1)])

    async def simulate_battle(self, beast1: Beast,
                              beast2: Beast) -> Dict[str, Any]:
        """Simulate a battle between two beasts"""
        fighter1 = Beast.from_dict(beast1.to_dict())
        fighter2 = Beast.from_dict(beast2.to_dict())
        first, second = self.determine_turn_order(fighter1, fighter2)
        battle_log = []
        turn = 1
        max_turns = 50

        while fighter1.stats.hp > 0 and fighter2.stats.hp > 0 and turn <= max_turns:
            damage = self.calculate_damage(first, second)
            second.stats.hp = max(0, second.stats.hp - damage)
            battle_log.append({
                'turn': turn,
                'attacker': first.name,
                'defender': second.name,
                'damage': damage,
                'defender_hp': second.stats.hp
            })
            if second.stats.hp <= 0:
                break

            damage = self.calculate_damage(second, first)
            first.stats.hp = max(0, first.stats.hp - damage)
            battle_log.append({
                'turn': turn,
                'attacker': second.name,
                'defender': first.name,
                'damage': damage,
                'defender_hp': first.stats.hp
            })
            turn += 1

        if fighter1.stats.hp > 0 and fighter2.stats.hp <= 0:
            winner = beast1.name
            winner_original = beast1
            result = BattleResult.WIN if beast1 == fighter1 else BattleResult.LOSS
        elif fighter2.stats.hp > 0 and fighter1.stats.hp <= 0:
            winner = beast2.name
            winner_original = beast2
            result = BattleResult.LOSS if beast1 == fighter1 else BattleResult.WIN
        else:
            winner = None
            winner_original = None
            result = BattleResult.DRAW

        return {
            'winner': winner,
            'winner_beast': winner_original,
            'result': result,
            'turns': turn - 1,
            'battle_log': battle_log,
            'final_hp': {
                beast1.name: fighter1.stats.hp,
                beast2.name: fighter2.stats.hp
            }
        }


class CloudBackupManager:
    """Enhanced backup manager with cloud storage support"""

    def __init__(self, config: BotConfig):
        self.config = config
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv(
            'GITHUB_REPO')  # format: "username/repo-name"
        self.backup_branch = os.getenv('BACKUP_BRANCH', 'main')
        self.logger = logging.getLogger(__name__)

    async def create_backup_with_cloud_storage(self) -> Optional[str]:
        """Create backup and store both locally and in cloud"""
        try:
            # Create local backup first
            local_backup = await self._create_local_backup()
            if not local_backup:
                return None

            # Upload to GitHub if configured
            if self.github_token and self.github_repo:
                cloud_backup = await self._upload_to_github(local_backup)
                if cloud_backup:
                    self.logger.info(
                        f"Backup uploaded to GitHub: {cloud_backup}")
                else:
                    self.logger.warning("Failed to upload backup to GitHub")

            return local_backup

        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return None

    async def _create_local_backup(self) -> Optional[str]:
        """Create local database backup"""
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.db"

            # Copy database file
            if Path(self.config.database_path).exists():
                shutil.copy2(self.config.database_path, backup_file)
                self.logger.info(f"Local backup created: {backup_file}")

                # Clean up old local backups
                await self._cleanup_local_backups(backup_dir)
                return str(backup_file)
            else:
                self.logger.warning("Database file not found for backup")
                return None

        except Exception as e:
            self.logger.error(f"Local backup failed: {e}")
            return None

    async def _upload_to_github(self, backup_file_path: str) -> Optional[str]:
        """Upload backup to GitHub repository"""
        try:
            if not all([self.github_token, self.github_repo]):
                self.logger.info(
                    "GitHub credentials not configured, skipping cloud backup")
                return None

            # Read backup file
            with open(backup_file_path, 'rb') as f:
                file_content = f.read()

            # Encode to base64
            encoded_content = base64.b64encode(file_content).decode('utf-8')

            # Prepare GitHub API request
            filename = Path(backup_file_path).name
            api_url = f"https://api.github.com/repos/{self.github_repo}/contents/backups/{filename}"

            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Check if file already exists
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    file_exists = response.status == 200
                    existing_sha = None
                    if file_exists:
                        existing_data = await response.json()
                        existing_sha = existing_data.get('sha')

                # Prepare commit data
                commit_data = {
                    'message': f'Backup: {filename}',
                    'content': encoded_content,
                    'branch': self.backup_branch
                }

                if existing_sha:
                    commit_data['sha'] = existing_sha

                # Upload file
                async with session.put(api_url,
                                       headers=headers,
                                       json=commit_data) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        download_url = result['content']['download_url']
                        self.logger.info(
                            f"Backup uploaded to GitHub: {download_url}")
                        return download_url
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"GitHub upload failed: {response.status} - {error_text}"
                        )
                        return None

        except Exception as e:
            self.logger.error(f"GitHub backup upload failed: {e}")
            return None

    async def restore_from_cloud(self) -> bool:
        """Restore database from the latest cloud backup"""
        try:
            if not all([self.github_token, self.github_repo]):
                self.logger.error("GitHub credentials not configured")
                return False

            # List all backups in the repository
            api_url = f"https://api.github.com/repos/{self.github_repo}/contents/backups"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers=headers) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"Failed to list backups: {response.status}")
                        return False

                    files = await response.json()
                    backup_files = [
                        f for f in files if f['name'].startswith('backup_')
                        and f['name'].endswith('.db')
                    ]

                    if not backup_files:
                        self.logger.warning(
                            "No backup files found in repository")
                        return False

                    # Sort by name (timestamp) to get the latest
                    latest_backup = sorted(backup_files,
                                           key=lambda x: x['name'])[-1]

                    # Download the latest backup
                    download_url = latest_backup['download_url']
                    async with session.get(download_url) as download_response:
                        if download_response.status == 200:
                            backup_content = await download_response.read()

                            # Ensure database directory exists
                            db_path = Path(self.config.database_path)
                            db_path.parent.mkdir(parents=True, exist_ok=True)

                            # Write the backup to database location
                            with open(db_path, 'wb') as f:
                                f.write(backup_content)

                            self.logger.info(
                                f"Database restored from cloud backup: {latest_backup['name']}"
                            )
                            return True
                        else:
                            self.logger.error(
                                f"Failed to download backup: {download_response.status}"
                            )
                            return False

        except Exception as e:
            self.logger.error(f"Cloud restore failed: {e}")
            return False

    async def _cleanup_local_backups(self, backup_dir: Path):
        """Clean up old local backups"""
        try:
            backup_files = sorted(backup_dir.glob("backup_*.db"),
                                  key=lambda p: p.stat().st_mtime,
                                  reverse=True)

            # Keep only the configured number of backups
            for old_backup in backup_files[self.config.
                                           backup_retention_count:]:
                old_backup.unlink()
                self.logger.info(f"Removed old local backup: {old_backup}")

        except Exception as e:
            self.logger.error(f"Local backup cleanup failed: {e}")


class BeastTemplateManager:
    """Manages beast templates and spawning"""

    def __init__(self, data_file: str = "data/beast_templates.yaml"):
        self.data_file = Path(data_file)
        self.templates: Dict[str, BeastTemplate] = {}
        self._ensure_data_directory()
        self._load_templates()

    def _ensure_data_directory(self):
        """Ensure the data directory exists"""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_templates(self):
        """Load beast templates from file"""
        if not self.data_file.exists():
            self._create_default_templates()
            return

        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not data:
                self._create_default_templates()
                return

            for name, template_data in data.items():
                try:
                    self.templates[name] = BeastTemplate(
                        name=name,
                        rarity=BeastRarity(template_data['rarity']),
                        tendency=template_data['tendency'],
                        location=template_data['location'],
                        base_hp_range=tuple(template_data['base_hp_range']),
                        base_attack_range=tuple(
                            template_data['base_attack_range']),
                        description=template_data.get('description', ''))
                except Exception as e:
                    logging.error(
                        f"Failed to load beast template '{name}': {e}")

            logging.info(f"Loaded {len(self.templates)} beast templates")
        except Exception as e:
            logging.error(
                f"Failed to load beast templates from {self.data_file}: {e}")
            self._create_default_templates()

    def _create_default_templates(self):
        """Create default beast templates with expanded variety"""
        default_data = {
            # ★ COMMON BEASTS
            "Flood Dragonling": {
                "rarity": 1,
                "tendency": "n/a",
                "location": "Fishing",
                "base_hp_range": [80, 120],
                "base_attack_range": [15, 25],
                "description": "A young dragon with control over floods"
            },

            # ★★ UNCOMMON BEASTS
            "Shenghuang": {
                "rarity": 2,
                "tendency": "n/a",
                "location": "Hell Market",
                "base_hp_range": [120, 180],
                "base_attack_range": [25, 35],
                "description": "A celestial phoenix of divine nature"
            },
            "Pi Xiu": {
                "rarity": 2,
                "tendency": "n/a",
                "location": "Hell Market",
                "base_hp_range": [115, 175],
                "base_attack_range": [22, 32],
                "description": "A mystical creature that brings fortune"
            },
            "Chi Wen": {
                "rarity": 2,
                "tendency": "n/a",
                "location": "Heaven Sect Market",
                "base_hp_range": [125, 185],
                "base_attack_range": [26, 36],
                "description": "A dragon son known for its water control"
            },
            "Bi'an": {
                "rarity": 2,
                "tendency": "n/a",
                "location": "Heaven Sect Market",
                "base_hp_range": [118, 178],
                "base_attack_range": [24, 34],
                "description": "A righteous beast that judges good and evil"
            },

            # ★★★ RARE BEASTS
            "Red Phoenix": {
                "rarity": 3,
                "tendency": "Divine Ephor or Tai Bai +8%",
                "location": "Hell Market",
                "base_hp_range": [180, 250],
                "base_attack_range": [35, 50],
                "description": "A crimson phoenix burning with divine flames"
            },
            "Azure Phoenix": {
                "rarity": 3,
                "tendency": "Fu Xi or Yandi +8%",
                "location": "Monthly Event",
                "base_hp_range": [185, 255],
                "base_attack_range": [37, 52],
                "description": "A blue phoenix with ancient wisdom"
            },
            "Baize": {
                "rarity": 3,
                "tendency": "Dragon Princess or Princess Longji +8%",
                "location": "Hell Market",
                "base_hp_range": [175, 245],
                "base_attack_range": [33, 48],
                "description": "A wise beast that knows all creatures"
            },
            "Xiezhi": {
                "rarity": 3,
                "tendency": "Ne Zha, Yang Jian, or Xing Tian +8%",
                "location": "Monthly Event",
                "base_hp_range": [190, 260],
                "base_attack_range": [38, 53],
                "description": "A unicorn-like beast of justice"
            },
            "Ninetails": {
                "rarity": 3,
                "tendency": "Yao Ji, Da Ji, or Lord TongTian +8%",
                "location": "Spirit Jade Pavilion",
                "base_hp_range": [195, 265],
                "base_attack_range": [40, 55],
                "description": "A mystical fox with nine tails"
            },
            "Shenpa": {
                "rarity": 3,
                "tendency": "n/a",
                "location": "Cultivator Starter",
                "base_hp_range": [170, 240],
                "base_attack_range": [32, 47],
                "description": "A cultivator's first companion beast"
            },
            "Tianhun": {
                "rarity": 3,
                "tendency": "n/a",
                "location": "Cultivator Starter",
                "base_hp_range": [178, 248],
                "base_attack_range": [34, 49],
                "description": "A spirit beast of the heavens"
            },
            "Bixin": {
                "rarity": 3,
                "tendency": "n/a",
                "location": "Cultivator Starter",
                "base_hp_range": [172, 242],
                "base_attack_range": [33, 48],
                "description": "A beast with a pure heart"
            },
            "Snow": {
                "rarity": 3,
                "tendency": "n/a",
                "location": "Cultivator Starter",
                "base_hp_range": [180, 250],
                "base_attack_range": [35, 50],
                "description": "A frost beast of eternal winter"
            },
            "Shadow Monster": {
                "rarity": 3,
                "tendency": "n/a",
                "location": "Tales of Demons&Gods Event",
                "base_hp_range": [188, 258],
                "base_attack_range": [37, 52],
                "description": "A creature born from pure shadow"
            },
            "Cyan Phoenix": {
                "rarity": 3,
                "tendency": "n/a",
                "location": "Tales of Demons&Gods Event",
                "base_hp_range": [185, 255],
                "base_attack_range": [36, 51],
                "description": "A cyan-feathered phoenix of legends"
            },

            # ★★★★ EPIC BEASTS
            "Kyrin": {
                "rarity": 4,
                "tendency": "Eastern Immortal +12%",
                "location": "Spirit Jade Pavilion",
                "base_hp_range": [300, 450],
                "base_attack_range": [60, 90],
                "description": "A majestic unicorn from the eastern realms"
            },
            "Azure Dragon": {
                "rarity": 4,
                "tendency": "Primordial Land +12%",
                "location": "Hell Market",
                "base_hp_range": [320, 470],
                "base_attack_range": [65, 95],
                "description": "A mighty dragon of the eastern skies"
            },
            "Jiuying": {
                "rarity": 4,
                "tendency": "Skyshine Continent +12%",
                "location": "Hell Market",
                "base_hp_range": [310, 460],
                "base_attack_range": [62, 92],
                "description": "A nine-headed water serpent"
            },
            "ZhuYan": {
                "rarity": 4,
                "tendency": "Deification Domain +12%",
                "location": "Void Treasure Quintessence",
                "base_hp_range": [305, 455],
                "base_attack_range": [61, 91],
                "description": "A fire ape that brings war"
            },
            "Hell Steed": {
                "rarity": 4,
                "tendency": "Heaven Palace +12%",
                "location": "Treasure Shop",
                "base_hp_range": [315, 465],
                "base_attack_range": [63, 93],
                "description": "A demonic horse from the underworld"
            },
            "Yuan Chu": {
                "rarity": 4,
                "tendency": "Heaven Palace +12%",
                "location": "Treasure Shop",
                "base_hp_range": [318, 468],
                "base_attack_range": [64, 94],
                "description": "A primordial beast of creation"
            },
            "Jue Ru": {
                "rarity": 4,
                "tendency": "Luo Shen, Shi Ji, Xi He, or Queen Mother +12%",
                "location": "Treasure Shop",
                "base_hp_range": [312, 462],
                "base_attack_range": [62, 92],
                "description": "A beast of absolute power"
            },
            "Monster Nian": {
                "rarity": 4,
                "tendency": "Deification Domain +12%",
                "location": "New Year Event",
                "base_hp_range": [325, 475],
                "base_attack_range": [65, 95],
                "description": "The legendary beast of the new year"
            },
            "LoveBird": {
                "rarity": 4,
                "tendency": "Skyshine Continent +12%",
                "location": "Valentine's Day Event",
                "base_hp_range": [300, 450],
                "base_attack_range": [60, 90],
                "description": "A pair of birds representing eternal love"
            },
            "Phoenix": {
                "rarity": 4,
                "tendency": "Primordial Land +12%",
                "location": "Women's Day Event",
                "base_hp_range": [330, 480],
                "base_attack_range": [66, 96],
                "description": "The legendary bird of rebirth"
            },
            "QingYuan": {
                "rarity": 4,
                "tendency": "Eastern Immortal +12%",
                "location": "Event",
                "base_hp_range": [308, 458],
                "base_attack_range": [62, 92],
                "description": "A beast from the eastern immortal realm"
            },
            "Devour": {
                "rarity": 4,
                "tendency": "Deification Domain +12%",
                "location": "Monthly Event",
                "base_hp_range": [322, 472],
                "base_attack_range": [64, 94],
                "description": "A beast that devours all in its path"
            },
            "Thunder": {
                "rarity": 4,
                "tendency": "Skyshine Continent +12%",
                "location": "Thunder Valley",
                "base_hp_range": [315, 465],
                "base_attack_range": [63, 93],
                "description": "A beast embodying the power of thunder"
            },
            "Fly Dragon": {
                "rarity": 4,
                "tendency": "Primordial Land +12%",
                "location": "Sky Realm",
                "base_hp_range": [320, 470],
                "base_attack_range": [64, 94],
                "description": "A dragon that soars through the heavens"
            },
            "Koi": {
                "rarity": 4,
                "tendency": "Eastern Immortal +12%",
                "location": "Celestial Pond",
                "base_hp_range": [310, 460],
                "base_attack_range": [62, 92],
                "description": "A sacred fish that brings good fortune"
            },
            "Picapica": {
                "rarity": 4,
                "tendency": "Deification Domain +12%",
                "location": "Divine Forest",
                "base_hp_range": [305, 455],
                "base_attack_range": [61, 91],
                "description": "A small but mighty divine creature"
            },
            "Greenbull": {
                "rarity": 4,
                "tendency": "Skyshine Continent +12%",
                "location": "Emerald Plains",
                "base_hp_range": [335, 485],
                "base_attack_range": [67, 97],
                "description": "A powerful bull with emerald horns"
            },
            "Jade Rabbit": {
                "rarity": 4,
                "tendency": "Eastern Immortal +12%",
                "location": "Moon Palace",
                "base_hp_range": [300, 450],
                "base_attack_range": [60, 90],
                "description": "The moon goddess's companion"
            },
            "Dragon Horse": {
                "rarity": 4,
                "tendency": "Deification Domain +12%",
                "location": "Divine Stables",
                "base_hp_range": [318, 468],
                "base_attack_range": [64, 94],
                "description": "A mystical horse with dragon heritage"
            },
            "Black Kyrin": {
                "rarity": 4,
                "tendency": "Primordial Land +12%",
                "location": "Shadow Realm",
                "base_hp_range": [325, 475],
                "base_attack_range": [65, 95],
                "description": "A dark unicorn of immense power"
            },
            "Reindeer": {
                "rarity": 4,
                "tendency": "Skyshine Continent +12%",
                "location": "Frozen Tundra",
                "base_hp_range": [310, 460],
                "base_attack_range": [62, 92],
                "description": "A magical reindeer from the north"
            },
            "Panda": {
                "rarity": 4,
                "tendency": "Primordial Land +12%",
                "location": "Bamboo Grove",
                "base_hp_range": [330, 480],
                "base_attack_range": [66, 96],
                "description": "A gentle but powerful bear"
            },
            "Shadow Lord": {
                "rarity": 4,
                "tendency": "Skyshine Continent +12%",
                "location": "Tales of Demons&Gods Event",
                "base_hp_range": [340, 490],
                "base_attack_range": [68, 98],
                "description": "Lord of all shadows and darkness"
            },
            "Elephant": {
                "rarity": 4,
                "tendency": "Deification Domain +12%",
                "location": "Sacred Grove",
                "base_hp_range": [350, 500],
                "base_attack_range": [70, 100],
                "description": "A wise and powerful elephant"
            },
            "Sky Phoenix": {
                "rarity": 4,
                "tendency": "Eastern Immortal +12%",
                "location": "Tales of Demons&Gods Event",
                "base_hp_range": [335, 485],
                "base_attack_range": [67, 97],
                "description": "A phoenix that rules the skies"
            },

            # ★★★★★ LEGENDARY BEASTS
            "Heavenly Kun": {
                "rarity": 5,
                "tendency": "Luo Shen, Shi Ji, Xi He, or Queen Mother +15%",
                "location": "Event",
                "base_hp_range": [500, 700],
                "base_attack_range": [100, 140],
                "description": "A colossal fish from the heavenly seas"
            },
            "Nine-Hell Steed": {
                "rarity": 5,
                "tendency": "Heaven Palace +15%",
                "location": "Treasure Shop",
                "base_hp_range": [520, 720],
                "base_attack_range": [104, 144],
                "description": "A demonic horse from the nine hells"
            },
            "Tien Yuanchu": {
                "rarity": 5,
                "tendency": "Heaven Palace +15%",
                "location": "Treasure Shop",
                "base_hp_range": [510, 710],
                "base_attack_range": [102, 142],
                "description": "The primordial beast of heaven"
            },
            "Snowy Jue Ru": {
                "rarity": 5,
                "tendency": "Luo Shen, Shi Ji, Xi He, or Queen Mother +15%",
                "location": "Christmas Event",
                "base_hp_range": [495, 695],
                "base_attack_range": [99, 139],
                "description": "A frost-covered beast of absolute power"
            },
            "Immortal Eater": {
                "rarity": 5,
                "tendency": "Deification Domain +15%",
                "location": "Treasure Shop",
                "base_hp_range": [530, 730],
                "base_attack_range": [106, 146],
                "description": "A fearsome beast that devours immortals"
            },
            "QueenBird": {
                "rarity": 5,
                "tendency": "Skyshine Continent +15%",
                "location": "Valentine's Day Event",
                "base_hp_range": [505, 705],
                "base_attack_range": [101, 141],
                "description": "The sovereign of all flying creatures"
            },
            "Deity Phoenix": {
                "rarity": 5,
                "tendency": "Primordial Land +15%",
                "location": "Women's Day Event",
                "base_hp_range": [525, 725],
                "base_attack_range": [105, 145],
                "description": "A phoenix elevated to deity status"
            },
            "Deity QingYuan": {
                "rarity": 5,
                "tendency": "Eastern Immortal +15%",
                "location": "Event",
                "base_hp_range": [515, 715],
                "base_attack_range": [103, 143],
                "description": "A deified beast from the eastern realm"
            },
            "Dark Devourer": {
                "rarity": 5,
                "tendency": "Deification Domain +15%",
                "location": "Monthly Event",
                "base_hp_range": [535, 735],
                "base_attack_range": [107, 147],
                "description": "A creature that consumes darkness itself"
            },
            "Purple Thunder": {
                "rarity": 5,
                "tendency": "Skyshine Continent +15%",
                "location": "Storm Peak",
                "base_hp_range": [520, 720],
                "base_attack_range": [104, 144],
                "description": "A beast wreathed in purple lightning"
            },
            "Lucky Fly Dragon": {
                "rarity": 5,
                "tendency": "Primordial Land +15%",
                "location": "Fortune Valley",
                "base_hp_range": [515, 715],
                "base_attack_range": [103, 143],
                "description": "A dragon that brings incredible luck"
            },
            "King Koi": {
                "rarity": 5,
                "tendency": "Eastern Immortal +15%",
                "location": "Royal Pond",
                "base_hp_range": [500, 700],
                "base_attack_range": [100, 140],
                "description": "The king of all koi fish"
            },
            "Star Picapica": {
                "rarity": 5,
                "tendency": "Deification Domain +15%",
                "location": "Stellar Grove",
                "base_hp_range": [510, 710],
                "base_attack_range": [102, 142],
                "description": "A star-blessed divine creature"
            },
            "Qing Greenbull": {
                "rarity": 5,
                "tendency": "Skyshine Continent +15%",
                "location": "Mystic Forest",
                "base_hp_range": [545, 745],
                "base_attack_range": [109, 149],
                "description": "A legendary bull with emerald horns"
            },
            "Moon Rabbit": {
                "rarity": 5,
                "tendency": "Eastern Immortal +15%",
                "location": "Lunar Palace",
                "base_hp_range": [495, 695],
                "base_attack_range": [99, 139],
                "description": "The moon's eternal guardian"
            },
            "Buddha Dragon Horse": {
                "rarity": 5,
                "tendency": "Deification Domain +15%",
                "location": "Sacred Temple",
                "base_hp_range": [525, 725],
                "base_attack_range": [105, 145],
                "description": "A horse blessed by Buddha himself"
            },
            "Snowy Reindeer": {
                "rarity": 5,
                "tendency": "Skyshine Continent +15%",
                "location": "Winter Event",
                "base_hp_range": [505, 705],
                "base_attack_range": [101, 141],
                "description": "A magical reindeer from eternal winter"
            },
            "Might Panda": {
                "rarity": 5,
                "tendency": "Primordial Land +15%",
                "location": "Ancient Grove",
                "base_hp_range": [540, 740],
                "base_attack_range": [108, 148],
                "description": "A panda of incredible strength"
            },
            "Shadow Immortal": {
                "rarity": 5,
                "tendency": "Skyshine Continent +15%",
                "location": "Tales of Demons&Gods Event",
                "base_hp_range": [530, 730],
                "base_attack_range": [106, 146],
                "description": "An immortal being born from shadows"
            },
            "Immortal Elephant": {
                "rarity": 5,
                "tendency": "Deification Domain +15%",
                "location": "Divine Sanctuary",
                "base_hp_range": [550, 750],
                "base_attack_range": [110, 150],
                "description": "An elephant that achieved immortality"
            },
            "Heaventy Phoenix": {
                "rarity": 5,
                "tendency": "Eastern Immortal +15%",
                "location": "Tales of Demons&Gods Event",
                "base_hp_range": [535, 735],
                "base_attack_range": [107, 147],
                "description": "A phoenix blessed by the heavens"
            },

            # ★★★★★★ MYTHIC BEASTS
            "Yinglong": {
                "rarity": 6,
                "tendency": "Leizu, Xianle, or Taiyuan +18%",
                "location": "Mythical Realm",
                "base_hp_range": [800, 1200],
                "base_attack_range": [150, 200],
                "description": "A legendary winged dragon of immense power"
            },
            "River God": {
                "rarity": 6,
                "tendency": "Leizu, Xianle, or Taiyuan +18%",
                "location": "Sacred River",
                "base_hp_range": [820, 1220],
                "base_attack_range": [155, 205],
                "description": "The divine ruler of all rivers and streams"
            },
            "Immortal Panda": {
                "rarity": 6,
                "tendency": "Primordial Land +18%",
                "location": "Spring Festival Event",
                "base_hp_range": [850, 1250],
                "base_attack_range": [160, 210],
                "description": "A panda that transcended mortality"
            },
            "Dragon Ancestor": {
                "rarity": 6,
                "tendency": "Leizu, Xianle, or Taiyuan +20%",
                "location": "Primordial Void",
                "base_hp_range": [900, 1300],
                "base_attack_range": [170, 220],
                "description": "The first dragon, ancestor of all dragonkind"
            },
            "Dragon Chi Wen": {
                "rarity": 6,
                "tendency": "Skyshine Continent +17%",
                "location": "Operation Events",
                "base_hp_range": [830, 1230],
                "base_attack_range": [157, 207],
                "description": "The most powerful of the dragon sons"
            }
        }

        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_data, f, default_flow_style=False)
            logging.info(
                f"Created expanded beast templates in {self.data_file}")
            # Load the templates we just created
            self._load_templates()
        except Exception as e:
            logging.error(f"Failed to create default templates: {e}")

    def get_random_template_by_rarity(
            self, rarity: BeastRarity) -> Optional[BeastTemplate]:
        """Get random template of specific rarity"""
        rarity_templates = [
            template for template in self.templates.values()
            if template.rarity == rarity
        ]
        return random.choice(rarity_templates) if rarity_templates else None

    def get_random_template_up_to_rarity(
        self,
        max_rarity: BeastRarity,
        rarity_weights: Optional[Dict[BeastRarity, float]] = None
    ) -> BeastTemplate:
        """Get random template up to specified rarity with weights"""
        if not self.templates:
            raise ValueError("No beast templates available")

        if rarity_weights is None:
            rarity_weights = {
                BeastRarity.COMMON: 27,
                BeastRarity.UNCOMMON: 40,
                BeastRarity.RARE: 27,
                BeastRarity.EPIC: 5,
                BeastRarity.LEGENDARY: 1
            }

        available_templates = []
        weights = []
        for template in self.templates.values():
            if template.rarity.value <= max_rarity.value and template.rarity in rarity_weights:
                available_templates.append(template)
                weights.append(rarity_weights[template.rarity])

        # Use weighted selection if we have both templates and weights
        if available_templates and weights:
            return random.choices(available_templates, weights=weights)[0]

        # Fallback to unweighted selection
        fallback_templates = [
            t for t in self.templates.values()
            if t.rarity.value <= max_rarity.value
        ]

        if fallback_templates:
            return random.choice(fallback_templates)

        # Ultimate fallback - return any template
        return random.choice(list(self.templates.values()))

    def get_template_by_name(self, name: str) -> Optional[BeastTemplate]:
        """Get specific template by name"""
        return self.templates.get(name)


async def migrate_all_beasts_to_new_xp_curve(bot):
    """One-time migration to recalculate all beast levels using new XP curve"""
    print("🔄 Starting XP curve migration...")

    try:
        # Get all users
        conn = bot.db._get_connection()
        user_cursor = conn.execute('SELECT user_id FROM users')
        all_users = user_cursor.fetchall()
        conn.close()

        total_beasts_migrated = 0
        users_processed = 0

        for user_row in all_users:
            try:
                user_id = user_row['user_id']
                user_beasts = await bot.db.get_user_beasts(user_id)

                for beast_id, beast in user_beasts:
                    # Calculate total XP this beast has earned using OLD formula
                    total_xp = beast.stats.exp  # Current XP

                    # Add XP from all previous levels (using OLD curve)
                    for level in range(1, beast.stats.level):
                        temp_stats = BeastStats(0, 0, 0, 0, 0, level, 0)
                        total_xp += int(beast.rarity.value * 100 *
                                        (level**1.2))  # OLD formula

                    # Now recalculate using NEW formula
                    beast.stats.level = 1
                    beast.stats.exp = 0

                    # Level up using new curve
                    while True:
                        # NEW formula: base × (1.5 ^ (level - 1))
                        base_requirements = {
                            BeastRarity.COMMON: 100,
                            BeastRarity.UNCOMMON: 150,
                            BeastRarity.RARE: 200,
                            BeastRarity.EPIC: 300,
                            BeastRarity.LEGENDARY: 500,
                            BeastRarity.MYTHIC: 800
                        }
                        base_req = base_requirements[beast.rarity]
                        required_xp = int(base_req *
                                          (1.5**(beast.stats.level - 1)))

                        if total_xp >= required_xp:
                            beast.stats.level += 1
                            total_xp -= required_xp
                        else:
                            break

                    # Set remaining XP
                    beast.stats.exp = total_xp

                    # Update beast in database
                    await bot.db.update_beast(beast_id, beast)
                    total_beasts_migrated += 1

                users_processed += 1
                if users_processed % 10 == 0:
                    print(
                        f"📊 Processed {users_processed} users, {total_beasts_migrated} beasts migrated"
                    )

            except Exception as e:
                print(f"❌ Error processing user {user_row['user_id']}: {e}")
                continue

        print(
            f"✅ Migration complete! {total_beasts_migrated} beasts migrated across {users_processed} users"
        )
        return True

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False


class UserRoleManager:
    """Manages user roles and permissions"""

    def __init__(self, config: BotConfig):
        self.config = config

    def get_user_role(self, member: discord.Member) -> UserRole:
        """Determine user's role type based on Discord roles"""
        user_role_ids = [role.id for role in member.roles]
        if self.config.personal_role_id in user_role_ids:
            return UserRole.PERSONAL
        # Check for special roles (list)
        elif any(role_id in user_role_ids
                 for role_id in self.config.special_role_ids):
            return UserRole.SPECIAL
        else:
            return UserRole.NORMAL

    def get_beast_limit(self, user_role: UserRole) -> int:
        """Get beast inventory limit based on user role"""
        limits = {
            UserRole.NORMAL: self.config.normal_user_beast_limit,
            UserRole.SPECIAL: self.config.special_role_beast_limit,
            UserRole.PERSONAL: self.config.personal_role_beast_limit
        }
        return limits[user_role]

    def can_use_adopt_legend(self, user_role: UserRole) -> bool:
        """Check if user can use adopt legend command"""
        return user_role in [UserRole.SPECIAL, UserRole.PERSONAL]

    def can_use_adopt_mythic(self, user_role: UserRole) -> bool:
        """Check if user can use adopt mythic command"""
        return user_role == UserRole.PERSONAL


async def select_beast_for_battle(
        ctx,
        user: discord.Member,
        beasts: List[Tuple[int, Beast]],  # Fix this type
        pronoun: str = "your") -> Optional[Tuple[int, Beast]]:
    """Helper function to let a user select a beast for battle"""
    if not beasts:
        return None

    embed = discord.Embed(
        title=f"Select {pronoun.title()} Beast for Battle",
        description=
        f"{user.mention}, choose a beast by reacting with the corresponding number:",
        color=0x00AAFF)

    options = beasts[:10]
    number_emojis = [
        '1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟'
    ]

    for i, (beast_id, beast) in enumerate(options):
        embed.add_field(
            name=
            f"{number_emojis[i]} #{beast_id} {beast.name} {beast.rarity.emoji}",
            value=
            f"Level {beast.stats.level} | HP: {beast.stats.hp}/{beast.stats.max_hp} | Power: {beast.power_level}",
            inline=False)

    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    for i in range(len(options)):
        await ctx.bot.safe_add_reaction(message, number_emojis[i])

    def check(reaction, react_user):
        return (react_user == user
                and str(reaction.emoji) in number_emojis[:len(options)]
                and reaction.message.id == message.id)

    try:
        reaction, _ = await ctx.bot.wait_for('reaction_add',
                                             timeout=30.0,
                                             check=check)
        selected_index = number_emojis.index(str(reaction.emoji))
        selected_beast = options[selected_index]
        await message.delete()
        return selected_beast
    except asyncio.TimeoutError:
        await message.delete()
        return None


def _today_str():
    return datetime.datetime.now().strftime('%Y-%m-%d')


def _check_and_reset_battle_day(user: User):
    today = _today_str()
    if user.battle_xp_day != today:
        user.battle_xp_day = today
        user.battle_xp_count = 0


class ImmortalBeastsBot(commands.Bot):
    """Main bot class with time-based and message-based XP systems"""

    def __init__(self, config: BotConfig):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.prefix,
                         intents=intents,
                         help_command=None)

        self.api_rate_limiter = RateLimiter(
            max_calls=30,
            period=60)  # Reduced from 60 to 30 requests per minute
        #self.activity_tracker = UserActivityTracker()
        self.config = config
        self.db: DatabaseInterface = SQLiteDatabase(config.database_path)
        self.template_manager = BeastTemplateManager()
        self.role_manager = UserRoleManager(config)
        self.battle_engine = BattleEngine()
        self.backup_manager = CloudBackupManager(config)
        self.spawn_channel_id = config.spawn_channel_id
        self.current_spawned_beast: Optional[Beast] = None
        self.catch_attempts: Dict[int, int] = {}  # user_id -> attempt_count
        self.max_catch_attempts = 3
        self.guild_rate_manager = GuildRateLimitManager()

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('bot.log'),
                      logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    async def safe_api_call(self,
                            api_func,
                            guild_id: Optional[int] = None,
                            *args,
                            **kwargs):
        """Enhanced safe API call with better rate limiting"""

        # Use guild-specific rate limiter if guild_id is provided
        if guild_id:
            try:
                guild_rate_limiter = await self.guild_rate_manager.get_api_rate_limiter(
                    guild_id)
                rate_limiter = guild_rate_limiter
            except Exception as e:
                self.logger.warning(
                    f"Failed to get guild rate limiter for {guild_id}: {e}")
                rate_limiter = self.api_rate_limiter
        else:
            rate_limiter = self.api_rate_limiter

        for attempt in range(3):  # REDUCED from 5 to 3 attempts
            try:
                # INCREASED wait time before each call
                wait_time = await rate_limiter.wait_if_needed()
                if wait_time:
                    self.logger.info(
                        f"Rate limited (Guild: {guild_id}), waited {wait_time:.1f}s"
                    )

                result = await api_func(*args, **kwargs)
                return result

            except Exception as e:
                if hasattr(e, "status") and getattr(e, "status", None) == 429:
                    # EXPONENTIAL BACKOFF - much more conservative
                    wait = min(300, 30 * (2**attempt))  # Cap at 5 minutes
                    self.logger.warning(
                        f"429 error, backing off for {wait} seconds (attempt {attempt+1})"
                    )
                    await asyncio.sleep(wait)
                else:
                    self.logger.error(f"API call failed: {e}")
                    if attempt < 2:  # Only retry for non-429 errors once
                        await asyncio.sleep(5)
                    else:
                        break

        self.logger.error("Max retries exceeded for API call")
        return None

    async def safe_send_message(self, channel, *args, **kwargs):
        """Send message with rate limiting"""
        guild_id = getattr(channel, 'guild', None)
        guild_id = guild_id.id if guild_id else None

        return await self.safe_api_call(channel.send, guild_id, *args,
                                        **kwargs)

    async def safe_add_reaction(self, message, emoji):
        """Add reaction with rate limiting"""
        guild_id = getattr(message, 'guild', None)
        guild_id = guild_id.id if guild_id else None

        return await self.safe_api_call(message.add_reaction, guild_id, emoji)

    async def safe_edit_message(self, message, *args, **kwargs):
        """Edit message with rate limiting"""
        guild_id = getattr(message, 'guild', None)
        guild_id = guild_id.id if guild_id else None

        return await self.safe_api_call(message.edit, guild_id, *args,
                                        **kwargs)

    async def auto_select_active_beast(
            self, user_id: int) -> Optional[Tuple[int, Beast]]:
        """Automatically select the best beast as active (fallback only)"""
        user_beasts = await self.db.get_user_beasts(user_id)

        if not user_beasts:
            return None

        # If only 1 beast, select it
        if len(user_beasts) == 1:
            return user_beasts[0]

        # Find highest rarity
        max_rarity = max(beast.rarity.value for _, beast in user_beasts)

        # Filter for highest rarity beasts
        highest_rarity_beasts = [(beast_id, beast)
                                 for beast_id, beast in user_beasts
                                 if beast.rarity.value == max_rarity]

        # If only one of highest rarity, select it
        if len(highest_rarity_beasts) == 1:
            return highest_rarity_beasts[0]

        # Multiple of same highest rarity - select most powerful
        # Primary: Power level, Secondary: Level, Tertiary: Beast ID (for consistency)
        best_beast = max(highest_rarity_beasts,
                         key=lambda x: (x[1].power_level, x[1].stats.level, x))

        return best_beast

    async def setup_hook(self):
        """Enhanced setup with automatic restoration - MESSAGE XP ONLY"""
        # Check if database exists, if not try to restore from cloud
        db_path = Path(self.config.database_path)
        if not db_path.exists() or db_path.stat().st_size == 0:
            self.logger.info(
                "Database not found or empty, attempting cloud restore...")
            restored = await self.backup_manager.restore_from_cloud()
            if restored:
                self.logger.info("✅ Database restored from cloud backup!")
            else:
                self.logger.info(
                    "No cloud backup found, starting with fresh database")

        startup_delay = random.uniform(5, 15)  # Random delay 5-15 seconds
        self.logger.info(
            f"Waiting {startup_delay:.1f}s before initializing...")
        await asyncio.sleep(startup_delay)

        await self.db.initialize()
        self.logger.info("Database initialized")

        await self.add_cog(RateLimitConfig(self, self.guild_rate_manager))
        self.logger.info("Rate limiting cog loaded")

        # Update backup task interval based on config
        self.enhanced_backup_task.change_interval(
            hours=self.config.backup_interval_hours)
        if self.config.backup_enabled:
            self.enhanced_backup_task.start()
            self.logger.info(
                f"Enhanced backup task started (every {self.config.backup_interval_hours}h, keep {self.config.backup_retention_count})"
            )
        else:
            self.logger.info("Backup task disabled")

        # Start background tasks (NO XP DISTRIBUTION TASK!)
        # self.xp_distribution_task.start()  # ← REMOVE THIS LINE!
        self.spawn_task.start()
        self.logger.info("Background tasks started (message-based XP only)")

    async def on_ready(self):
        """Called when bot is ready"""
        self.logger.info(f'{self.user} has connected to Discord!')
        activity = discord.Game(
            name=f"Immortal Beasts | {self.config.prefix}help")
        await self.change_presence(activity=activity)

    async def on_message(self, message):
        """Handle messages for XP gain with smart auto-selection fallback"""
        if message.author.bot:
            await self.process_commands(message)
            return

        # Handle message-based XP only
        if (message.channel.id in self.config.xp_chat_channel_ids
                and hasattr(message, 'content') and len(message.content) > 3):

            user = await self.get_or_create_user(message.author.id,
                                                 str(message.author))

            # AUTO-SELECT ACTIVE BEAST if none set (fallback only)
            if not user.active_beast_id:
                best_beast = await self.auto_select_active_beast(user.user_id)
                if best_beast:
                    beast_id, beast = best_beast
                    user.active_beast_id = beast_id
                    await self.db.update_user(user)

                    # Optional: Log for debugging (silent to users)
                    self.logger.info(
                        f"Auto-selected active beast for {message.author}: {beast.name} (ID: {beast_id})"
                    )

            # Process XP if active beast is set (either manual or auto-selected)
            if user.active_beast_id:
                await self.handle_message_xp_gain(message)

        await self.process_commands(message)

    async def handle_message_xp_gain(self, message):
        """Handle XP gain from messages with anti-spam protection"""
        try:
            user = await self.get_or_create_user(message.author.id,
                                                 str(message.author))

            # Check cooldown (now 60 seconds)
            if not user.can_gain_xp(self.config.xp_cooldown_seconds):
                return

            if not user.active_beast_id:
                return

            # Enhanced anti-spam validation
            if not self._is_valid_xp_message(message):
                return

            # Get active beast
            user_beasts = await self.db.get_user_beasts(user.user_id)
            active_beast = None
            active_beast_id = None

            for beast_id, beast in user_beasts:
                if beast_id == user.active_beast_id:
                    active_beast = beast
                    active_beast_id = beast_id
                    break

            if not active_beast or active_beast_id is None:
                return

            # Calculate role-based XP amount with bonuses AND boost
            base_xp = get_user_message_xp_amount(message.author, self.config)
            xp_amount = get_boosted_xp_amount(base_xp, user)

            # Add XP and check for level ups
            level_ups = active_beast.stats.add_exp(xp_amount,
                                                   active_beast.rarity)

            # Update database
            user.last_xp_gain = datetime.datetime.now()
            await self.db.update_user(user)
            await self.db.update_beast(active_beast_id, active_beast)

            # Send level up notification (only for level ups, not every XP gain)
            if level_ups:
                for leveled_up, bonus_level, stat_gains in level_ups:
                    embed = discord.Embed(
                        title="🎉 Level Up!",
                        description=
                        f"{message.author.mention}'s **{active_beast.name}** leveled up!",
                        color=active_beast.rarity.color)

                    embed.add_field(name="New Level",
                                    value=f"Level {active_beast.stats.level}",
                                    inline=True)
                    embed.add_field(name="Power Level",
                                    value=active_beast.power_level,
                                    inline=True)

                    # Regular stat gains
                    gain_text = []
                    for stat, gain in stat_gains.items():
                        if not stat.startswith('bonus_'):
                            gain_text.append(f"{stat.title()}: +{gain}")

                    embed.add_field(name="Stat Gains",
                                    value="\n".join(gain_text),
                                    inline=False)

                    # Bonus stats for level 5 multiples
                    if bonus_level:
                        bonus_text = []
                        for stat, gain in stat_gains.items():
                            if stat.startswith('bonus_'):
                                clean_stat = stat.replace('bonus_', '')
                                bonus_text.append(
                                    f"{clean_stat.title()}: +{gain}")

                        embed.add_field(
                            name="🌟 Bonus Stats (Level 5 Multiple)!",
                            value="\n".join(bonus_text),
                            inline=False)

                    embed.set_footer(text=f"Message XP: +{xp_amount}")

                    # Add boost notification if active
                    if xp_amount > base_xp:
                        embed.add_field(
                            name="🔥 XP Boost Active!",
                            value=f"Base: {base_xp} → Boosted: {xp_amount}",
                            inline=False)

                    await message.channel.send(embed=embed)

        except Exception as e:
            self.logger.error(f"Error in handle_message_xp_gain: {e}")

    def _is_valid_xp_message(self, message) -> bool:
        """Comprehensive anti-spam validation for XP messages"""
        content = message.content.strip()

        # Basic length check (minimum 5 characters)
        if len(content) < 5:
            return False

        # Check for spam patterns
        if self._is_spam_pattern(content):
            return False

        # Check for repetitive content
        if self._is_repetitive_content(message.author.id, content):
            return False

        # Check message frequency (additional rate limiting)
        if self._is_message_spam(message.author.id):
            return False

        return True

    def _is_spam_pattern(self, content: str) -> bool:
        """Detect common spam patterns"""
        content_lower = content.lower()

        # Single character repeated
        if len(set(content.replace(' ', ''))) <= 2 and len(content) > 10:
            return True

        # Common spam phrases
        spam_phrases = ['xp', 'exp', 'experience', 'level up', 'lvl', 'farm']

        # If message is just spam phrases
        words = content_lower.split()
        if len(words) <= 3 and any(phrase in content_lower
                                   for phrase in spam_phrases):
            return True

        # Excessive punctuation or caps
        punct_ratio = sum(
            1 for c in content if c in '!@#$%^&*().,?;:') / len(content)
        if punct_ratio > 0.3:
            return True

        caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
        if caps_ratio > 0.7 and len(content) > 10:
            return True

        return False

    def _is_repetitive_content(self, user_id: int, content: str) -> bool:
        """Check if user is repeating the same content"""
        if not hasattr(self, '_user_message_history'):
            self._user_message_history = {}

        if user_id not in self._user_message_history:
            self._user_message_history[user_id] = []

        user_history = self._user_message_history[user_id]

        # Keep only last 5 messages
        if len(user_history) >= 5:
            user_history.pop(0)

        # Check similarity with recent messages
        similar_count = 0
        for old_msg in user_history:
            similarity = self._calculate_similarity(content.lower(),
                                                    old_msg.lower())
            if similarity > 0.8:  # 80% similar
                similar_count += 1

        # Add current message to history
        user_history.append(content)

        # If 3+ similar messages in recent history, it's spam
        return similar_count >= 2

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if len(str1) == 0 and len(str2) == 0:
            return 1.0
        if len(str1) == 0 or len(str2) == 0:
            return 0.0

        # Simple character-based similarity
        set1, set2 = set(str1), set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _is_message_spam(self, user_id: int) -> bool:
        """Check if user is sending messages too frequently"""
        if not hasattr(self, '_user_message_timestamps'):
            self._user_message_timestamps = {}

        now = datetime.datetime.now()

        if user_id not in self._user_message_timestamps:
            self._user_message_timestamps[user_id] = []

        timestamps = self._user_message_timestamps[user_id]

        # Remove timestamps older than 2 minutes
        timestamps[:] = [
            ts for ts in timestamps if (now - ts).total_seconds() < 120
        ]

        # If more than 6 messages in 2 minutes, it's spam
        if len(timestamps) >= 6:
            return True

        # Add current timestamp
        timestamps.append(now)

        return False

    async def on_command_error(self, ctx, error):
        """Global error handler"""
        if isinstance(error, commands.CommandNotFound):
            return

        if isinstance(error, commands.MissingPermissions):
            embed = discord.Embed(
                title="❌ Missing Permissions",
                description="You don't have permission to use this command.",
                color=0xFF0000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)

        self.logger.error(f"Command error in {ctx.command}: {error}")
        embed = discord.Embed(
            title="❌ Error",
            description="An error occurred while processing your command.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    async def get_or_create_user(self, user_id: int, username: str) -> User:
        """Get user or create if doesn't exist"""
        user = await self.db.get_user(user_id)
        if not user:
            user = User(user_id=user_id,
                        username=username,
                        spirit_stones=self.config.starting_beast_stones)
            await self.db.create_user(user)
        return user

    @tasks.loop(hours=6)  # This will be updated in setup_hook
    async def enhanced_backup_task(self):
        """Enhanced backup task with cloud storage"""
        if not self.config.backup_enabled:
            return

        try:
            backup_file = await self.safe_api_call(
                self.backup_manager.create_backup_with_cloud_storage)
            if backup_file:
                self.logger.info(f"Enhanced backup completed: {backup_file}")
            else:
                self.logger.error("Enhanced backup failed")

        except Exception as e:
            self.logger.error(f"Enhanced backup task failed: {e}")

    @tasks.loop(minutes=1)
    async def spawn_task(self):
        """Improved beast spawning with single channel"""
        try:
            if not self.spawn_channel_id:
                self.logger.debug("No spawn channel configured")
                return

            # Check if there's already a beast spawned
            if self.current_spawned_beast is not None:
                return  # Don't spawn if there's already one

            # Check if it's time to spawn
            spawn_interval = self.config.fixed_spawn_interval_minutes  # ✅ Use config
            if not hasattr(self, '_next_spawn_time'):
                self._next_spawn_time = datetime.datetime.now(
                ) + datetime.timedelta(minutes=spawn_interval)
                self.logger.info(
                    f"Next beast spawn scheduled in {spawn_interval} minutes")
                return

            if datetime.datetime.now() < self._next_spawn_time:
                return  # Not time yet

            # Time to spawn!
            channel = self.get_channel(self.spawn_channel_id)
            if channel and isinstance(channel, discord.TextChannel):
                await self.spawn_beast(channel)
                self.logger.info(
                    f"Beast spawned in channel: {channel.name} ({self.spawn_channel_id})"
                )

                # Schedule next spawn
                self._next_spawn_time = datetime.datetime.now(
                ) + datetime.timedelta(minutes=spawn_interval)
                self.logger.info(
                    f"Next beast spawn scheduled in {spawn_interval} minutes")
            elif channel:
                # ✅ Enhanced: Channel exists but wrong type
                self.logger.warning(
                    f"Spawn channel {self.spawn_channel_id} is not a text channel (type: {type(channel).__name__})"
                )
            else:
                # ✅ Enhanced: Channel doesn't exist
                self.logger.warning(
                    f"Spawn channel {self.spawn_channel_id} not found")

        except Exception as e:
            self.logger.error(f"Spawn task failed: {e}")

    async def spawn_beast(self, channel: discord.TextChannel):
        """Spawn a beast in the given channel"""
        try:
            rarity_weights = {
                BeastRarity.COMMON: 50,
                BeastRarity.UNCOMMON: 30,
                BeastRarity.RARE: 17,
                BeastRarity.EPIC: 2,
                BeastRarity.LEGENDARY: 1,
                BeastRarity.MYTHIC: 0.1
            }

            template = self.template_manager.get_random_template_up_to_rarity(
                BeastRarity.MYTHIC, rarity_weights)
            beast = template.create_beast()

            # Set the current spawned beast and reset catch attempts
            self.current_spawned_beast = beast
            self.catch_attempts.clear()

            embed = discord.Embed(
                title="🌟 A Wild Beast Appeared! 🌟",
                description=
                f"**{beast.name}** has appeared!\n{beast.rarity.emoji}\n\n"
                f"Quick! Use `{self.config.prefix}catch` to capture it!\n"
                f"**Each user gets {self.max_catch_attempts} attempts!**",
                color=beast.rarity.color)

            embed.add_field(name="Rarity",
                            value=beast.rarity.emoji,
                            inline=True)
            embed.add_field(name="Level", value=beast.stats.level, inline=True)
            embed.add_field(name="Power", value=beast.power_level, inline=True)
            embed.add_field(name="Tendency",
                            value=beast.tendency or "None",
                            inline=False)
            embed.add_field(name="Location",
                            value=beast.location or "Unknown",
                            inline=False)

            if beast.description:
                embed.add_field(name="Description",
                                value=beast.description,
                                inline=False)

            await channel.send(embed=embed)

            # Beast disappears after 5 minutes
            await asyncio.sleep(300)
            if self.current_spawned_beast == beast:  # Still the same beast
                self.current_spawned_beast = None
                self.catch_attempts.clear()
                embed = discord.Embed(
                    title="💨 Beast Fled",
                    description=f"The {beast.name} has disappeared...",
                    color=0x808080)
                await channel.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error spawning beast: {e}")


def require_channel(channel_type: str):
    """Decorator to restrict commands to specific channels"""

    def decorator(func):

        @functools.wraps(func)
        async def wrapper(ctx, *args, **kwargs):
            config = ctx.bot.config

            if channel_type == "battle":
                # Ensure type consistency
                battle_channel_ids = [
                    int(ch_id) for ch_id in config.battle_channel_ids
                ]
                current_channel_id = int(ctx.channel.id)

                if current_channel_id not in battle_channel_ids:
                    embed = discord.Embed(
                        title="❌ Wrong Channel",
                        description=
                        "This command can only be used in designated battle channels!",
                        color=0xFF0000)
                    await ctx.bot.safe_send_message(ctx.channel, embed=embed)
                    return

            elif channel_type == "adopt":
                if int(ctx.channel.id) != int(config.adopt_channel_id):
                    embed = discord.Embed(
                        title="❌ Wrong Channel",
                        description=
                        "This command can only be used in the adoption channel!",
                        color=0xFF0000)
                    await ctx.bot.safe_send_message(ctx.channel, embed=embed)
                    return

            elif channel_type == "spawn":
                if int(ctx.channel.id) != int(config.spawn_channel_id):
                    embed = discord.Embed(
                        title="❌ Wrong Channel",
                        description=
                        "This command can only be used in the beast spawn channel!",
                        color=0xFF0000)
                    await ctx.bot.safe_send_message(ctx.channel, embed=embed)
                    return

            return await func(ctx, *args, **kwargs)

        return wrapper

    return decorator


# Commands


@commands.command(name='stone')
async def daily_stone_reward(ctx):
    """Claim your daily beast stone reward"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    now = datetime.datetime.now()
    if user.last_daily:
        time_since_last = now - user.last_daily
        if time_since_last.total_seconds() < 24 * 3600:
            remaining_time = datetime.timedelta(hours=24) - time_since_last
            hours = int(remaining_time.total_seconds() // 3600)
            minutes = int((remaining_time.total_seconds() % 3600) // 60)

            embed = discord.Embed(
                title="⏰ Daily Beast Stones Already Claimed",
                description=
                f"You can claim your next daily beast stones in {hours}h {minutes}m",
                color=0xFF8000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

    daily_reward = 100
    user.spirit_stones += daily_reward
    user.last_daily = now
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🔮 Daily Beast Stones Claimed!",
        description=
        f"**{ctx.author.display_name}** received **{daily_reward} Beast Stones**!",
        color=0x9932CC)

    embed.add_field(name="💎 Reward",
                    value=f"{daily_reward} beast stones",
                    inline=True)
    embed.add_field(name="💰 Total Beast Stones",
                    value=f"{user.spirit_stones:,} stones",
                    inline=True)
    embed.add_field(
        name="⏰ Next Claim",
        value=f"<t:{int((now + datetime.timedelta(hours=24)).timestamp())}:R>",
        inline=False)

    embed.set_footer(text="Come back tomorrow for another 100 beast stones!")
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# SPAWN MANAGEMENT COMMANDS - ADD THESE TO YOUR COMMANDS SECTION
@commands.command(name='forcespawn')
@commands.has_permissions(administrator=True)
async def force_spawn(ctx):
    """Manually force a beast to spawn (admin only)"""
    # OLD CODE: Check if channel is in spawn_channels
    # NEW CODE: Check if this is the designated spawn channel
    if ctx.channel.id != ctx.bot.spawn_channel_id:
        embed = discord.Embed(
            title="❌ Not the Spawn Channel",
            description="This is not the designated beast spawn channel.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check if there's already a beast spawned
    if ctx.bot.current_spawned_beast:
        embed = discord.Embed(
            title="❌ Beast Already Spawned",
            description=
            f"There's already a {ctx.bot.current_spawned_beast.name} waiting to be caught!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    try:
        await ctx.bot.spawn_beast(ctx.channel)
        embed = discord.Embed(
            title="✅ Beast Force Spawned",
            description="A wild beast has been manually spawned!",
            color=0x00FF00)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
    except Exception as e:
        embed = discord.Embed(title="❌ Spawn Failed",
                              description=f"Failed to spawn beast: {str(e)}",
                              color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='spawninfo')
@commands.has_permissions(administrator=True)
async def spawn_info(ctx):
    """Show spawn system information (admin only)"""
    embed = discord.Embed(title="📊 Beast Spawn Information", color=0x00AAFF)

    # OLD CODE: Multiple spawn channels
    # NEW CODE: Single spawn channel
    if ctx.bot.spawn_channel_id:
        channel = ctx.bot.get_channel(ctx.bot.spawn_channel_id)
        if channel:
            embed.add_field(name="🎯 Spawn Channel",
                            value=f"#{channel.name}",
                            inline=False)
        else:
            embed.add_field(name="🎯 Spawn Channel",
                            value=f"Unknown ({ctx.bot.spawn_channel_id})",
                            inline=False)
    else:
        embed.add_field(name="🎯 Spawn Channel",
                        value="Not configured",
                        inline=False)

    # Timing info - KEEP THIS UNCHANGED
    embed.add_field(
        name="⏰ Spawn Interval",
        value=
        f"{ctx.bot.config.spawn_interval_min}-{ctx.bot.config.spawn_interval_max} minutes",
        inline=True)

    # Next spawn time - KEEP THIS UNCHANGED
    if hasattr(ctx.bot, '_next_spawn_time'):
        time_until_spawn = ctx.bot._next_spawn_time - datetime.datetime.now()
        minutes_left = int(time_until_spawn.total_seconds() / 60)
        if minutes_left > 0:
            embed.add_field(name="⏱️ Next Spawn",
                            value=f"In {minutes_left} minutes",
                            inline=True)
        else:
            embed.add_field(name="⏱️ Next Spawn",
                            value="Due now!",
                            inline=True)
    else:
        embed.add_field(name="⏱️ Next Spawn",
                        value="Not scheduled",
                        inline=True)

    # OLD CODE: Current spawned beasts in multiple channels
    # NEW CODE: Single current spawned beast
    if ctx.bot.current_spawned_beast:
        beast = ctx.bot.current_spawned_beast
        embed.add_field(name="🐉 Current Wild Beast",
                        value=f"{beast.name} {beast.rarity.emoji}",
                        inline=False)

        # NEW: Show catch attempts for this beast
        attempts_info = []
        for user_id, attempts in ctx.bot.catch_attempts.items():
            user = ctx.bot.get_user(user_id)
            if user:
                attempts_info.append(
                    f"{user.display_name}: {attempts}/{ctx.bot.max_catch_attempts}"
                )

        if attempts_info:
            embed.add_field(name="🎯 Catch Attempts",
                            value="\n".join(attempts_info[:5]) +
                            (f"\n... and {len(attempts_info)-5} more"
                             if len(attempts_info) > 5 else ""),
                            inline=False)
        else:
            embed.add_field(name="🎯 Catch Attempts",
                            value="No attempts yet",
                            inline=False)
    else:
        embed.add_field(name="🐉 Current Wild Beast",
                        value="None",
                        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='sacrifice', aliases=['sac'])
async def sacrifice_beast(ctx, beast_id: int):
    """Sacrifice a beast to gain experience for your active beast"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)

    # Locate the target beast
    target_beast = None
    for bid, beast in user_beasts:
        if bid == beast_id:
            target_beast = beast
            break

    # Error: No such beast
    if not target_beast:
        embed = discord.Embed(color=0xFF1744)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 🚫 ⚡ IMMORTAL BEAST SHRINE ⚡ 🚫\n"
                        "## ❌ BEAST NOT FOUND ❌\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🔍 Beast Registry Search",
                        value=(f"```\n"
                               f"- Beast ID: #{beast_id}\n"
                               f"- Owner: {ctx.author.display_name}\n"
                               f"- Status: NOT FOUND\n"
                               f"- Collection: {len(user_beasts)} beasts\n"
                               f"```"),
                        inline=False)
        embed.add_field(
            name="💡 Available Actions",
            value=(
                f"📋 `{ctx.bot.config.prefix}beasts` - View your collection\n"
                f"🔍 `{ctx.bot.config.prefix}beast <id>` - Check beast details\n"
                f"⚡ Use a valid beast ID from your collection"),
            inline=False)
        embed.set_footer(
            text="⚡ IMMORTAL BEAST SHRINE • Double-check your beast ID!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Error: Attempting to sacrifice the active beast
    if beast_id == user.active_beast_id:
        embed = discord.Embed(color=0xFF6D00)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 🛡️ ⚡ IMMORTAL BEAST SHRINE ⚡ 🛡️\n"
                        "## ⚠️ ACTIVE BEAST PROTECTED ⚠️\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🟢 Active Beast Status",
                        value=(f"```\n"
                               f"Protected Beast: {target_beast.name}\n"
                               f"Beast ID: #{beast_id}\n"
                               f"Status: CURRENTLY ACTIVE\n"
                               f"Protection: SHRINE GUARDIAN\n"
                               f"```"),
                        inline=False)
        embed.add_field(
            name="🔄 Solution Required",
            value=
            (f"**Step 1:** Choose a different active beast\n"
             f"**Step 2:** Use `{ctx.bot.config.prefix}active <other_beast_id>`\n"
             f"**Step 3:** Return here to perform sacrifice\n\n"
             "🛡️ The shrine protects your active companion!"),
            inline=False)
        embed.set_footer(
            text=
            "⚡ IMMORTAL BEAST SHRINE • Your active beast cannot be sacrificed!"
        )
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Calculate the rewards
    sacrifice_xp = target_beast.stats.get_total_exp_value(target_beast.rarity)
    beast_stones_reward = target_beast.stats.level * 10 + target_beast.rarity.value * 50

    # Find active beast (not the sacrificial one)
    active_beast = None
    if user.active_beast_id and user.active_beast_id != beast_id:
        for bid, beast in user_beasts:
            if bid == user.active_beast_id:
                active_beast = beast
                break

    # Confirmation embed
    embed = discord.Embed(color=0xFF3D00)
    power_bar = "🔴" * min(10, target_beast.power_level // 1000) or "⬛"
    rarity_glow = "✨" * target_beast.rarity.value
    embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    value="# 🔥 ⚡ IMMORTAL BEAST SHRINE ⚡ 🔥\n"
                    "## ⚠️ SACRIFICE RITUAL INITIATED ⚠️\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    inline=False)
    embed.add_field(
        name="🐉 SACRIFICE CANDIDATE",
        value=
        (f"### {rarity_glow} {target_beast.name} {rarity_glow}\n"
         f"```\n"
         f"Beast ID: #{beast_id}\n"
         f"Rarity: {target_beast.rarity.name.title()} {target_beast.rarity.emoji}\n"
         f"Level: {target_beast.stats.level}\n"
         f"Power: {target_beast.power_level:,}\n"
         f"```\n"
         f"**Power Level:** {power_bar}\n"
         f"**Origin:** {target_beast.location}"),
        inline=False)
    stone_value_bars = "💎" * min(8, beast_stones_reward // 100) or "⬛"
    xp_value_bars = "⚡" * min(8, sacrifice_xp // 500) or "⬛"
    embed.add_field(
        name="💰 SHRINE REWARDS",
        value=(f"```\n"
               f"Beast Stones: {beast_stones_reward:,}\n"
               f"Value Tier:  {stone_value_bars}\n"
               f"Formula:     (Level × 10) + (Rarity × 50)\n"
               f"```\n"
               f"💎 {beast_stones_reward:,} Beast Stones will be granted!"),
        inline=True)
    if active_beast:
        embed.add_field(name="📈 SOUL TRANSFER",
                        value=(f"```\n"
                               f"Experience:  {sacrifice_xp:,} XP\n"
                               f"Power Tier:  {xp_value_bars}\n"
                               f"Recipient:   {active_beast.name}\n"
                               f"```\n"
                               f"⚡ Soul energy flows to {active_beast.name}!"),
                        inline=True)
    else:
        embed.add_field(name="💀 LOST ESSENCE",
                        value=(f"```\n"
                               f"- Experience: {sacrifice_xp:,} XP\n"
                               f"- Status: NO ACTIVE BEAST\n"
                               f"- Result: ESSENCE DISPERSES\n"
                               f"```\n"
                               f"💀 Soul energy will be lost forever!"),
                        inline=True)

    embed.add_field(
        name="⚠️ RITUAL WARNING",
        value=
        (f"```\n"
         f"- Beast Stones: PERMANENT GAIN\n"
         f"{'+ Soul Transfer: TO ACTIVE BEAST' if active_beast else '- Soul Energy: LOST FOREVER'}\n"
         f"- Beast Loss: IRREVERSIBLE\n"
         f"- Shrine Decision: FINAL\n"
         f"```"),
        inline=False)
    embed.add_field(
        name="🔥 SHRINE RITUAL COMMANDS",
        value=
        (f"### ✅ COMPLETE SACRIFICE\n"
         f"```\n"
         f"▶ Accept the shrine's power\n"
         f"▶ Claim beast stones\n"
         f"{'▶ Transfer soul to active beast' if active_beast else '▶ Release soul to the void'}\n"
         f"```\n"
         f"### ❌ ABANDON RITUAL\n"
         f"```\n"
         f"▶ Preserve your beast\n"
         f"▶ Leave shrine unchanged\n"
         f"```"),
        inline=False)
    embed.add_field(
        name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        value="⚡ IMMORTAL BEAST SHRINE • Ancient powers await your decision\n"
        "🔥 *React within 30 seconds or the ritual will expire...*",
        inline=False)
    embed.set_author(name=f"Ritual Master: {ctx.author.display_name}",
                     icon_url=ctx.author.display_avatar.url if hasattr(
                         ctx.author, 'display_avatar') else None)
    embed.timestamp = discord.utils.utcnow()
    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)
    await ctx.bot.safe_add_reaction(message, "✅")
    await ctx.bot.safe_add_reaction(message, "❌")

    def check(reaction, react_user):
        return (react_user == ctx.author and str(reaction.emoji) in ["✅", "❌"]
                and reaction.message.id == message.id)

    try:
        reaction, _ = await ctx.bot.wait_for('reaction_add',
                                             timeout=30.0,
                                             check=check)

        if str(reaction.emoji) == "✅":
            success = await ctx.bot.db.delete_beast(beast_id)
            if not success:
                error_embed = discord.Embed(color=0xFF1744)
                error_embed.add_field(
                    name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    value="# 💥 ⚡ RITUAL FAILURE ⚡ 💥\n"
                    "## ❌ SHRINE REJECTED SACRIFICE ❌\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    inline=False)
                error_embed.add_field(
                    name="🔧 System Error",
                    value=
                    "```\nDatabase operation failed\nPlease try again\n```",
                    inline=False)
                await ctx.bot.safe_edit_message(message, embed=error_embed)
                return

            # Add rewards and process sacrifice
            user.spirit_stones += beast_stones_reward

            # Transfer XP to active beast if available
            if active_beast:
                level_ups = active_beast.stats.add_exp(sacrifice_xp,
                                                       active_beast.rarity)
                await ctx.bot.db.update_beast(user.active_beast_id,
                                              active_beast)

            # --- Stat boost eligibility and application ---
            grant_stat_boosts = False
            if user.can_get_stat_boosts():
                if has_sufficient_bond(target_beast):
                    grant_stat_boosts = True

            if grant_stat_boosts and active_beast:
                stat_boosts = calculate_percentage_stat_boosts(target_beast)
                for stat, boost_value in stat_boosts.items():
                    if hasattr(active_beast.stats, stat):
                        setattr(
                            active_beast.stats, stat,
                            getattr(active_beast.stats, stat) + boost_value)
                await ctx.bot.db.update_beast(user.active_beast_id,
                                              active_beast)
                user.record_sacrifice_with_stats()

            await ctx.bot.db.update_user(user)

            # Success feedback embed
            success_embed = discord.Embed(color=0x00FF00)
            success_embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                    value="# ✅ ⚡ SACRIFICE COMPLETE ⚡ ✅\n"
                                    "## 🌟 SHRINE ACCEPTS OFFERING 🌟\n"
                                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                    inline=False)

            # Basic rewards
            reward_text = f"💎 **{beast_stones_reward:,} Beast Stones** added!\n"
            if active_beast:
                reward_text += f"⚡ **{sacrifice_xp:,} XP** transferred to {active_beast.name}"
            else:
                reward_text += f"💀 **{sacrifice_xp:,} XP** dispersed (no active beast)"

            success_embed.add_field(
                name="💰 Rewards Granted",
                value=reward_text,
                inline=False)

            # Show stat boosts if they were granted
            if grant_stat_boosts and active_beast:
                stat_boosts = calculate_percentage_stat_boosts(target_beast)
                boost_text = []
                for stat, boost_value in stat_boosts.items():
                    if stat == 'hp':
                        boost_text.append(f"❤️ **Max HP:** +{boost_value}")
                    elif stat == 'attack':
                        boost_text.append(f"⚔️ **Attack:** +{boost_value}")
                    elif stat == 'defense':
                        boost_text.append(f"🛡️ **Defense:** +{boost_value}")
                    elif stat == 'speed':
                        boost_text.append(f"💨 **Speed:** +{boost_value}")

                success_embed.add_field(
                    name="🎯 STAT BOOSTS APPLIED!",
                    value=f"**{active_beast.name}** received permanent stat bonuses!\n" + "\n".join(boost_text),
                    inline=False)

                # Show updated power level
                success_embed.add_field(
                    name="📊 Power Update",
                    value=f"**New Power Level:** {active_beast.power_level:,}\n"
                          f"🌟 **Permanent gains from sacrifice!**",
                    inline=True)

            elif active_beast and has_sufficient_bond(target_beast) and not user.can_get_stat_boosts():
                # Beast had sufficient bond but daily limit reached
                success_embed.add_field(
                    name="⏰ Daily Limit Reached",
                    value=f"**{target_beast.name}** had sufficient bond (12+ hours)\n"
                          f"But you've used all 5 daily stat boosts!\n"
                          f"✅ Still received XP and beast stones",
                    inline=False)

            elif active_beast and not has_sufficient_bond(target_beast):
                # Beast didn't have sufficient bond
                owned_time = datetime.datetime.now() - target_beast.caught_at if target_beast.caught_at else datetime.timedelta(0)
                hours_owned = owned_time.total_seconds() / 3600
                hours_needed = max(0, 12 - hours_owned)

                success_embed.add_field(
                    name="🕐 Insufficient Bond",
                    value=f"**{target_beast.name}** needed 12+ hours ownership for stat boosts\n"
                          f"⏰ Was owned for: {hours_owned:.1f} hours\n"
                          f"❓ Needed: {hours_needed:.1f} more hours\n"
                          f"✅ Still received XP and beast stones",
                    inline=False)

            # Show remaining daily stat boosts
            if active_beast:
                remaining_boosts = 5 - user.sacrifice_count_today if user.can_get_stat_boosts() else 0
                success_embed.add_field(
                    name="📅 Daily Status",
                    value=f"**Stat Boosts Remaining Today:** {remaining_boosts}/5",
                    inline=True)

            await ctx.bot.safe_edit_message(message, embed=success_embed)

        else:
            # Ritual abandoned
            abandoned_embed = discord.Embed(color=0x808080)
            abandoned_embed.add_field(
                name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                value="# 🛡️ ⚡ RITUAL ABANDONED ⚡ 🛡️\n"
                "## 💙 BEAST PRESERVED 💙\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                inline=False)
            abandoned_embed.add_field(
                name="🌙 Wise Choice",
                value=
                f"**{target_beast.name}** remains safely in your collection.\n"
                "The shrine respects your decision.",
                inline=False)

            await ctx.bot.safe_edit_message(message, embed=abandoned_embed)
    except asyncio.TimeoutError:
        timeout_embed = discord.Embed(color=0x616161)
        timeout_embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                value="# ⏳ ⚡ RITUAL EXPIRED ⚡ ⏳\n"
                                "## 🌙 SHRINE GROWS SILENT 🌙\n"
                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                inline=False)
        timeout_embed.add_field(
            name="⏰ Time's Judgment",
            value=
            (f"```\n"
             f"Ritual Duration: 30 seconds\n"
             f"Response: NONE RECEIVED\n"
             f"Shrine Status: DORMANT\n"
             f"Beast Status: SAFE\n"
             f"```\n"
             f"🌙 The shrine's power fades... {target_beast.name} remains protected"
             ),
            inline=False)
        timeout_embed.add_field(
            name="🔮 Ancient Wisdom",
            value=
            "*The shrine only accepts offerings from those who act with certainty*\n"
            "💫 Return when you're ready to make a decisive choice",
            inline=False)
        await ctx.bot.safe_edit_message(message, embed=timeout_embed)


@commands.command(name='release', aliases=['free'])
async def release_beast(ctx, beast_id: int):
    """Release a beast without any rewards (quick inventory management)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)

    # Check if user owns the beast
    target_beast = None
    for bid, beast in user_beasts:
        if bid == beast_id:
            target_beast = beast
            break

    if not target_beast:
        embed = discord.Embed(color=0xFF5722)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 🌿 🕊️ **SANCTUARY GATES** 🕊️ 🌿\n"
                        "## ❌ **BEAST NOT FOUND** ❌\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🔍 **Sanctuary Search**",
                        value=f"```diff\n"
                        f"- Seeking: Beast #{beast_id}\n"
                        f"- Owner: {ctx.author.display_name}\n"
                        f"- Result: NOT IN COLLECTION\n"
                        f"```",
                        inline=False)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check if trying to release active beast
    if beast_id == user.active_beast_id:
        embed = discord.Embed(color=0xFF8F00)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 🛡️ 🕊️ **SANCTUARY GATES** 🕊️ 🛡️\n"
                        "## ⚠️ **ACTIVE COMPANION PROTECTED** ⚠️\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="💙 **Bonded Guardian**",
                        value=f"```yaml\n"
                        f"Active Beast: {target_beast.name}\n"
                        f"Bond Status: CURRENTLY ACTIVE\n"
                        f"Protection: SANCTUARY GUARDIAN\n"
                        f"```\n"
                        f"🛡️ **Your active companion cannot be released!**",
                        inline=False)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Cool Release Confirmation
    embed = discord.Embed(color=0x4CAF50)

    # Nature-themed header
    embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    value="# 🌿 🕊️ **SANCTUARY RELEASE** 🕊️ 🌿\n"
                    "## 💚 **RETURN TO THE WILD** 💚\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    inline=False)

    # Beast showcase with nature theme
    freedom_visual = "🍃" * target_beast.rarity.value
    embed.add_field(
        name="🐉 **BEAST SEEKING FREEDOM**",
        value=f"### {freedom_visual} {target_beast.name} {freedom_visual}\n"
        f"```ansi\n"
        f"\u001b[1;32mBeast ID:\u001b[0m #{beast_id}\n"
        f"\u001b[1;36mRarity:\u001b[0m {target_beast.rarity.name.title()} {target_beast.rarity.emoji}\n"
        f"\u001b[1;33mLevel:\u001b[0m {target_beast.stats.level}\n"
        f"\u001b[1;35mOrigin:\u001b[0m {target_beast.location}\n"
        f"```\n"
        f"🌲 **Yearns to return to {target_beast.location}**",
        inline=False)

    # Release terms with clear styling
    embed.add_field(name="📜 **SANCTUARY TERMS**",
                    value="```diff\n"
                    "- NO Beast Stones Granted\n"
                    "- NO Experience Transfer\n"
                    "- NO Shrine Rewards\n"
                    "+ Instant Inventory Space\n"
                    "+ Peaceful Liberation\n"
                    "+ Beast Returns to Wild\n"
                    "```",
                    inline=True)

    embed.add_field(name="🌍 **Freedom Declaration**",
                    value="```yaml\n"
                    "Release Type: Peaceful\n"
                    "Destination: Natural Habitat\n"
                    "Status: Immediate Liberation\n"
                    "Rewards: None (Pure Freedom)\n"
                    "```",
                    inline=True)

    # Action options with nature styling
    embed.add_field(name="🕊️ **SANCTUARY CHOICE**",
                    value="### ✅ **GRANT FREEDOM**\n"
                    "```ansi\n"
                    "\u001b[1;32m▶ Release to natural habitat\u001b[0m\n"
                    "\u001b[1;32m▶ No rewards granted\u001b[0m\n"
                    "\u001b[1;32m▶ Instant liberation\u001b[0m\n"
                    "```\n"
                    "### ❌ **MAINTAIN BOND**\n"
                    "```ansi\n"
                    "\u001b[1;31m▶ Keep in collection\u001b[0m\n"
                    "\u001b[1;31m▶ Continue companionship\u001b[0m\n"
                    "```",
                    inline=False)

    # Quick footer
    embed.add_field(
        name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        value=
        "🕊️ **SANCTUARY GATES** • Choose quickly, the gates close in 15 seconds\n"
        "🌿 *True freedom requires no rewards, only compassion*",
        inline=False)

    embed.set_author(name=f"Sanctuary Keeper: {ctx.author.display_name}",
                     icon_url=ctx.author.display_avatar.url if hasattr(
                         ctx.author, 'display_avatar') else None)
    embed.timestamp = discord.utils.utcnow()

    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)
    await ctx.bot.safe_add_reaction(message, "✅")
    await ctx.bot.safe_add_reaction(message, "❌")

    def check(reaction, react_user):
        return (react_user == ctx.author and str(reaction.emoji) in ["✅", "❌"]
                and reaction.message.id == message.id)

    try:
        reaction, _ = await ctx.bot.wait_for('reaction_add',
                                             timeout=15.0,
                                             check=check)

        if str(reaction.emoji) == "✅":
            success = await ctx.bot.db.delete_beast(beast_id)
            if success:
                if user.active_beast_id == beast_id:
                    # Auto-select new active beast from remaining collection
                    new_active = await ctx.bot.auto_select_active_beast(
                        user.user_id)
                    user.active_beast_id = new_active[0] if new_active else None
                    await ctx.bot.db.update_user(user)

                    if new_active:
                        _, new_beast = new_active
                        # Add to your existing Beautiful success embed
                        embed.add_field(
                            name="🔄 **SANCTUARY AUTO-SELECTION**",
                            value=
                            f"New guardian chosen: **{new_beast.name}** {new_beast.rarity.emoji}\n"
                            f"🛡️ Your strongest companion steps forward to serve",
                            inline=False)
                else:
                    await ctx.bot.db.update_user(user)

                # Beautiful success embed
                embed = discord.Embed(color=0x66BB6A)
                embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                value="# 🌟 🕊️ **FREEDOM GRANTED** 🕊️ 🌟\n"
                                "## 💚 **SANCTUARY BLESSING** 💚\n"
                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                inline=False)
                embed.add_field(
                    name="🌿 **Liberation Complete**",
                    value=f"```ansi\n"
                    f"\u001b[1;32m{target_beast.name}\u001b[0m soars toward freedom\n"
                    f"Destination: {target_beast.location}\n"
                    f"Status: SUCCESSFULLY RELEASED\n"
                    f"```\n"
                    f"🕊️ **{target_beast.name} spreads wings toward {target_beast.location}**\n"
                    f"🌲 **The sanctuary gates close gently behind them**",
                    inline=False)
                embed.add_field(name="📊 **Collection Update**",
                                value=f"```yaml\n"
                                f"Remaining Beasts: {len(user_beasts) - 1}\n"
                                f"Freedom Granted: Peacefully\n"
                                f"Sanctuary Status: Mission Complete\n"
                                f"```",
                                inline=False)
            else:
                embed = discord.Embed(color=0xFF5722)
                embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                value="# ❌ 🕊️ **RELEASE FAILED** 🕊️ ❌\n"
                                "## 🚫 **SANCTUARY ERROR** 🚫\n"
                                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                                inline=False)
                embed.add_field(name="🔧 **System Failure**",
                                value="```diff\n"
                                "- Gate mechanism jammed\n"
                                "- Beast remains in collection\n"
                                "- Please try again\n"
                                "```",
                                inline=False)
        else:
            # Peaceful cancellation
            embed = discord.Embed(color=0x81C784)
            embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                            value="# 💙 🕊️ **BOND PRESERVED** 🕊️ 💙\n"
                            "## 🤝 **COMPANIONSHIP CONTINUES** 🤝\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                            inline=False)
            embed.add_field(
                name="💝 **Wise Choice**",
                value=f"```ansi\n"
                f"\u001b[1;36m{target_beast.name}\u001b[0m remains by your side\n"
                f"Your bond grows stronger\n"
                f"No separation occurred\n"
                f"```\n"
                f"💙 **{target_beast.name} appreciates your loyalty**",
                inline=False)

    except asyncio.TimeoutError:
        embed = discord.Embed(color=0xA5D6A7)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# ⏰ 🕊️ **GATES CLOSED** 🕊️ ⏰\n"
                        "## 🌙 **SANCTUARY RESTS** 🌙\n"
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(
            name="🌙 **Time's Wisdom**",
            value=f"```yaml\n"
            f"Decision Window: 15 seconds\n"
            f"Response: NONE RECEIVED\n"
            f"Beast Status: SAFELY PRESERVED\n"
            f"```\n"
            f"🕊️ **{target_beast.name} remains under your protection**",
            inline=False)

    embed.set_author(name=f"Sanctuary Record: {ctx.author.display_name}",
                     icon_url=ctx.author.display_avatar.url if hasattr(
                         ctx.author, 'display_avatar') else None)
    embed.timestamp = discord.utils.utcnow()

    await ctx.bot.safe_edit_message(message, embed=embed)

    # ADD THIS POWERFUL DEBUGGING COMMAND:

    from typing import Optional

    @commands.command(name='addxpchannel')
    @commands.has_permissions(administrator=True)
    async def add_xp_channel(ctx,
                             channel: Optional[discord.TextChannel] = None
                             ):  # ✅ Fixed type annotation
        """Add a channel to XP gain list (admin only)"""
        if channel is None:
            channel = ctx.channel
            # ✅ ADDED: Ensure ctx.channel is actually a TextChannel
            if not isinstance(channel, discord.TextChannel):
                embed = discord.Embed(
                    title="❌ Invalid Channel",
                    description=
                    "This command can only be used in text channels.",
                    color=0xFF0000)
                await ctx.bot.safe_send_message(ctx.channel, embed=embed)
                return

        if channel.id in ctx.bot.config.xp_chat_channel_ids:
            embed = discord.Embed(
                title="⚠️ Already Added",
                description=
                f"{channel.mention} is already in the XP channel list",
                color=0xFFAA00)
        else:
            ctx.bot.config.xp_chat_channel_ids.append(channel.id)
            embed = discord.Embed(
                title="✅ XP Channel Added",
                description=f"{channel.mention} has been added to XP channels",
                color=0x00FF00)
            embed.add_field(
                name="⚠️ Note",
                value=
                "This change is temporary. Update environment variables for permanent change.",
                inline=False)

        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# ADD THESE COMMANDS if you don't have them:


@commands.command(name='xpconfig')
@commands.has_permissions(administrator=True)
async def xp_config(ctx):
    """Show detailed XP system configuration (admin only)"""
    embed = discord.Embed(title="⚙️ XP System Configuration",
                          description="Detailed XP system settings",
                          color=0x00AAFF)

    # Basic settings
    embed.add_field(
        name="📊 Basic Settings",
        value=f"**XP per Message:** {ctx.bot.config.xp_per_message}\n"
        f"**Cooldown:** {ctx.bot.config.xp_cooldown_seconds} seconds\n"
        f"**Min Message Length:** 4 characters",
        inline=False)

    # XP Channels with validation
    valid_channels = []
    invalid_channels = []

    for channel_id in ctx.bot.config.xp_chat_channel_ids:
        channel = ctx.bot.get_channel(channel_id)
        if channel:
            valid_channels.append(f"✅ #{channel.name} ({channel_id})")
        else:
            invalid_channels.append(f"❌ Unknown Channel ({channel_id})")

    if valid_channels:
        embed.add_field(name="📺 Valid XP Channels",
                        value="\n".join(valid_channels[:10]) +
                        (f"\n... and {len(valid_channels)-10} more"
                         if len(valid_channels) > 10 else ""),
                        inline=False)

    if invalid_channels:
        embed.add_field(name="⚠️ Invalid XP Channels",
                        value="\n".join(invalid_channels[:5]) +
                        (f"\n... and {len(invalid_channels)-5} more"
                         if len(invalid_channels) > 5 else ""),
                        inline=False)

    # Statistics
    try:
        conn = ctx.bot.db._get_connection()

        # Count users with active beasts
        cursor = conn.execute(
            'SELECT COUNT(*) as count FROM users WHERE active_beast_id IS NOT NULL'
        )
        active_users = cursor.fetchone()['count']

        # Count total users
        cursor = conn.execute('SELECT COUNT(*) as count FROM users')
        total_users = cursor.fetchone()['count']

        # Count users who have gained XP
        cursor = conn.execute(
            'SELECT COUNT(*) as count FROM users WHERE last_xp_gain IS NOT NULL'
        )
        xp_users = cursor.fetchone()['count']

        conn.close()

        embed.add_field(
            name="📈 Usage Statistics",
            value=f"**Total Users:** {total_users}\n"
            f"**Users with Active Beasts:** {active_users}\n"
            f"**Users Who Gained XP:** {xp_users}\n"
            f"**Active Beast Rate:** {(active_users/total_users*100):.1f}%"
            if total_users > 0 else "**Active Beast Rate:** 0%",
            inline=True)

    except Exception as e:
        embed.add_field(name="📈 Usage Statistics",
                        value=f"Error retrieving stats: {str(e)}",
                        inline=True)

    embed.add_field(name="🔧 Troubleshooting",
                    value="**Common Issues:**\n"
                    "• Users don't have active beasts set\n"
                    "• Cooldown too restrictive (60s default)\n"
                    "• Invalid channel configuration\n"
                    "• Bot permissions in XP channels",
                    inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='feed')
async def feed_beast(ctx):
    """Feed your active beast for XP (24h cooldown)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    # Check cooldown
    if user.last_feed:
        time_since = datetime.datetime.now() - user.last_feed
        if time_since.total_seconds() < 24 * 3600:
            remaining = datetime.timedelta(hours=24) - time_since
            hours = int(remaining.total_seconds() // 3600)
            minutes = int((remaining.total_seconds() % 3600) // 60)

            embed = discord.Embed(
                title="⏰ Feed Cooldown Active",
                description=
                f"You can feed your beast again in {hours}h {minutes}m",
                color=0xFF8000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

    # Get active beast
    if not user.active_beast_id:
        embed = discord.Embed(
            title="❌ No Active Beast",
            description=
            "You need an active beast to feed! Use `!active <beast_id>`",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    active_beast = None
    active_beast_id = None

    for bid, beast in user_beasts:
        if bid == user.active_beast_id:
            active_beast = beast
            active_beast_id = bid
            break

    if not active_beast:
        embed = discord.Embed(
            title="❌ Active Beast Not Found",
            description="Your active beast couldn't be found!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Award XP
    feed_xp = 300
    boosted_xp = get_boosted_xp_amount(feed_xp, user)
    level_ups = active_beast.stats.add_exp(boosted_xp, active_beast.rarity)

    # Update database
    user.last_feed = datetime.datetime.now()
    await ctx.bot.db.update_user(user)
    await ctx.bot.db.update_beast(active_beast_id, active_beast)

    embed = discord.Embed(
        title="🍖 Beast Fed!",
        description=f"**{active_beast.name}** enjoyed a hearty meal!",
        color=active_beast.rarity.color)
    embed.add_field(name="XP Gained", value=f"+{boosted_xp} XP", inline=True)
    embed.add_field(name="New Level",
                    value=active_beast.stats.level,
                    inline=True)

    if boosted_xp > feed_xp:
        embed.add_field(name="🔥 XP Boost!",
                        value="50% bonus applied!",
                        inline=True)

    if level_ups:
        embed.add_field(name="🎉 Level Up!",
                        value=f"Gained {len(level_ups)} level(s)!",
                        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='groom')
async def groom_beast(ctx):
    """Groom your active beast for XP (24h cooldown)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    # Check cooldown
    if user.last_groom:
        time_since = datetime.datetime.now() - user.last_groom
        if time_since.total_seconds() < 24 * 3600:
            remaining = datetime.timedelta(hours=24) - time_since
            hours = int(remaining.total_seconds() // 3600)
            minutes = int((remaining.total_seconds() % 3600) // 60)

            embed = discord.Embed(
                title="⏰ Groom Cooldown Active",
                description=
                f"You can groom your beast again in {hours}h {minutes}m",
                color=0xFF8000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

    # Get active beast
    if not user.active_beast_id:
        embed = discord.Embed(
            title="❌ No Active Beast",
            description=
            "You need an active beast to groom! Use `!active <beast_id>`",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    active_beast = None
    active_beast_id = None

    for bid, beast in user_beasts:
        if bid == user.active_beast_id:
            active_beast = beast
            active_beast_id = bid
            break

    if not active_beast:
        embed = discord.Embed(
            title="❌ Active Beast Not Found",
            description="Your active beast couldn't be found!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Award XP
    groom_xp = 300
    boosted_xp = get_boosted_xp_amount(groom_xp, user)
    level_ups = active_beast.stats.add_exp(boosted_xp, active_beast.rarity)

    # Update database
    user.last_groom = datetime.datetime.now()
    await ctx.bot.db.update_user(user)
    await ctx.bot.db.update_beast(active_beast_id, active_beast)

    embed = discord.Embed(
        title="✨ Beast Groomed!",
        description=f"**{active_beast.name}** looks magnificent!",
        color=active_beast.rarity.color)
    embed.add_field(name="XP Gained", value=f"+{boosted_xp} XP", inline=True)
    embed.add_field(name="New Level",
                    value=active_beast.stats.level,
                    inline=True)

    if boosted_xp > groom_xp:
        embed.add_field(name="🔥 XP Boost!",
                        value="50% bonus applied!",
                        inline=True)

    if level_ups:
        embed.add_field(name="🎉 Level Up!",
                        value=f"Gained {len(level_ups)} level(s)!",
                        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='spar')
async def spar_training(ctx, beast_id: int, training_type: str = "strength"):
    """Train your beast with sparring (2h cooldown per beast, 15 sessions per day)

    Training Types:
    - strength: +1-3 Attack
    - endurance: +10-20 Max HP (and heal to full)
    - agility: +1-3 Speed  
    - defense: +1-3 Defense
    """
    training_types = {
        "strength": {
            "xp": 300,
            "stat": "attack",
            "emoji": "💪",
            "gain_range": (1, 3)
        },
        "endurance": {
            "xp": 300,
            "stat": "hp",
            "emoji": "❤️",
            "gain_range": (10, 20)
        },
        "agility": {
            "xp": 300,
            "stat": "speed",
            "emoji": "💨",
            "gain_range": (1, 3)
        },
        "defense": {
            "xp": 300,
            "stat": "defense",
            "emoji": "🛡️",
            "gain_range": (1, 3)
        }
    }

    if training_type.lower() not in training_types:
        embed = discord.Embed(
            title="❌ Invalid Training Type",
            description=f"**Available Training Types:**\n" +
            f"🏋️ `strength` - Increases Attack (+1-3)\n" +
            f"🏃 `endurance` - Increases Max HP (+10-20) and heals\n" +
            f"💨 `agility` - Increases Speed (+1-3)\n" +
            f"🛡️ `defense` - Increases Defense (+1-3)\n\n" +
            f"**Usage:** `!spar {beast_id} <type>`",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    # Check global daily limit FIRST
    if not user.can_spar():
        embed = discord.Embed(
            title="⏰ Daily Spar Limit Reached",
            description=
            f"You've used all 15 spar sessions today!\nResets at midnight.",
            color=0xFF8000)
        embed.add_field(name="Sessions Used",
                        value=f"{user.spar_sessions_today}/15",
                        inline=True)
        embed.add_field(name="Reset Time", value="Midnight UTC", inline=True)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check beast-specific cooldown using SQLite
    last_training = await ctx.bot.db.get_training_cooldown(
        ctx.author.id, beast_id)
    if last_training:
        time_since = datetime.datetime.now() - last_training
        if time_since.total_seconds() < 2 * 3600:  # 2 hour cooldown
            remaining = datetime.timedelta(hours=2) - time_since
            hours = int(remaining.total_seconds() // 3600)
            minutes = int((remaining.total_seconds() % 3600) // 60)

            embed = discord.Embed(
                title="⏰ Training Cooldown Active",
                description=
                f"Beast #{beast_id} can train again in {hours}h {minutes}m",
                color=0xFF8000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

    # Get beast
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    target_beast = None
    target_beast_id = None

    for bid, beast in user_beasts:
        if bid == beast_id:
            target_beast = beast
            target_beast_id = bid
            break

    if not target_beast:
        embed = discord.Embed(title="❌ Beast Not Found",
                              description=f"You don't own beast #{beast_id}",
                              color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Award XP and training
    training = training_types[training_type.lower()]
    boosted_xp = get_boosted_xp_amount(training["xp"], user)
    level_ups = target_beast.stats.add_exp(boosted_xp, target_beast.rarity)

    # Apply stat gain based on training type
    def apply_training_stat_gain(beast, training_stat: str, gain_range: tuple):
        """Apply random stat increase to beast based on training type"""
        gain = random.randint(*gain_range)

        if training_stat == 'hp':
            beast.stats.max_hp += gain
            beast.stats.hp += gain  # Also heal current HP
        elif training_stat == 'attack':
            beast.stats.attack += gain
        elif training_stat == 'defense':
            beast.stats.defense += gain
        elif training_stat == 'speed':
            beast.stats.speed += gain

        return gain

    stat_gain = apply_training_stat_gain(target_beast, training['stat'],
                                         training['gain_range'])

    # Update cooldowns using SQLite
    now = datetime.datetime.now()
    await ctx.bot.db.set_training_cooldown(ctx.author.id, beast_id, now)
    user.record_spar()
    await ctx.bot.db.update_user(user)
    await ctx.bot.db.update_beast(target_beast_id, target_beast)

    embed = discord.Embed(
        title=f"{training['emoji']} {training_type.title()} Training Complete!",
        description=
        f"**{target_beast.name}** completed {training_type} training!",
        color=target_beast.rarity.color)
    embed.add_field(name="💫 XP Gained", value=f"+{boosted_xp} XP", inline=True)
    embed.add_field(name="📈 Level",
                    value=f"Level {target_beast.stats.level}",
                    inline=True)
    embed.add_field(name="📊 Sessions Today",
                    value=f"{user.spar_sessions_today}/15",
                    inline=True)

    # Show the stat improvement
    stat_name = training['stat'].title()
    if training['stat'] == 'hp':
        embed.add_field(name="💪 Stat Improved",
                        value=f"+{stat_gain} Max HP\n🩹 HP fully restored!",
                        inline=True)
    else:
        embed.add_field(name="💪 Stat Improved",
                        value=f"+{stat_gain} {stat_name}",
                        inline=True)

    # Show current stats for reference
    embed.add_field(
        name="📋 Current Stats",
        value=f"HP: {target_beast.stats.hp}/{target_beast.stats.max_hp}\n"
        f"ATK: {target_beast.stats.attack} | DEF: {target_beast.stats.defense}\n"
        f"SPD: {target_beast.stats.speed} | PWR: {target_beast.power_level:,}",
        inline=False)

    if boosted_xp > training["xp"]:
        embed.add_field(name="🔥 XP Boost Active!",
                        value="50% bonus applied!",
                        inline=False)

    if level_ups:
        embed.add_field(
            name="🎉 Level Up Bonus!",
            value=
            f"Gained {len(level_ups)} level(s) from XP!\n🌟 Additional stat bonuses from leveling!",
            inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='dailies')
async def daily_status(ctx):
    """Check your daily quest status"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    embed = discord.Embed(
        title=f"📋 {ctx.author.display_name}'s Daily Quest Status",
        description="Track your daily quest progress",
        color=0x00AAFF)

    now = datetime.datetime.now()

    # Feed status
    if user.last_feed:
        time_since_feed = now - user.last_feed
        if time_since_feed.total_seconds() < 24 * 3600:
            remaining_feed = datetime.timedelta(hours=24) - time_since_feed
            hours = int(remaining_feed.total_seconds() // 3600)
            minutes = int((remaining_feed.total_seconds() % 3600) // 60)
            feed_status = f"❌ {hours}h {minutes}m remaining"
        else:
            feed_status = "✅ Available"
    else:
        feed_status = "✅ Available"

    # Groom status
    if user.last_groom:
        time_since_groom = now - user.last_groom
        if time_since_groom.total_seconds() < 24 * 3600:
            remaining_groom = datetime.timedelta(hours=24) - time_since_groom
            hours = int(remaining_groom.total_seconds() // 3600)
            minutes = int((remaining_groom.total_seconds() % 3600) // 60)
            groom_status = f"❌ {hours}h {minutes}m remaining"
        else:
            groom_status = "✅ Available"
    else:
        groom_status = "✅ Available"

    # Spar status
    spar_available = 15 - user.spar_sessions_today if user.can_spar() else 0

    # XP Boost status
    if user.xp_boost_expires and now < user.xp_boost_expires:
        boost_remaining = user.xp_boost_expires - now
        boost_minutes = int(boost_remaining.total_seconds() // 60)
        boost_status = f"🔥 Active ({boost_minutes}m remaining)"
    else:
        boost_status = "⚪ Inactive"

    embed.add_field(name="🍖 Feed Beast (300 XP)",
                    value=feed_status,
                    inline=True)
    embed.add_field(name="✨ Groom Beast (300 XP)",
                    value=groom_status,
                    inline=True)
    embed.add_field(name="⚔️ Spar Training (600 XP + Stats)",
                    value=f"✅ {spar_available} sessions left",
                    inline=True)
    embed.add_field(name="🔥 XP Boost (50%)", value=boost_status, inline=False)

    # Calculate potential XP today
    potential_xp = 0
    if "Available" in feed_status:
        potential_xp += 300
    if "Available" in groom_status:
        potential_xp += 300
    potential_xp += spar_available * 600

    if potential_xp > 0:
        embed.add_field(name="📊 Remaining XP Today",
                        value=f"{potential_xp:,} XP available",
                        inline=False)

    # Training guide
    embed.add_field(
        name="🏋️ Training Guide",
        value="**Spar Training Types:**\n" + "💪 `strength` - +Attack\n" +
        "❤️ `endurance` - +Max HP & heal\n" + "💨 `agility` - +Speed\n" +
        "🛡️ `defense` - +Defense",
        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='sacrificestatus')
async def sacrifice_status(ctx):
    """Check your sacrifice restrictions status"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    embed = discord.Embed(title="⚔️ Sacrifice Status",
                          description="Check your sacrifice limitations",
                          color=0x00AAFF)

    # Daily stat boost limit
    stat_boosts_left = 5 - user.sacrifice_count_today if user.can_get_stat_boosts(
    ) else 0
    embed.add_field(name="📊 Daily Stat Boosts",
                    value=f"**{stat_boosts_left}/5** remaining today",
                    inline=True)

    embed.add_field(name="🕐 Bond Requirement",
                    value="**12 hours** minimum ownership",
                    inline=True)

    embed.add_field(
        name="💡 Note",
        value=
        "You can always sacrifice for XP + stones!\nStat boosts require meeting both restrictions.",
        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='xpstatus', aliases=['xpcheck'])
async def xp_status(ctx):
    """Check your XP gain status and anti-spam information"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    embed = discord.Embed(
        title="⚡ XP System Status",
        description=f"XP gain analysis for {ctx.author.display_name}",
        color=0x00AAFF)

    # Channel check
    current_channel_valid = ctx.channel.id in ctx.bot.config.xp_chat_channel_ids
    embed.add_field(
        name="📍 Current Channel",
        value=
        f"**{ctx.channel.name}**\nXP Enabled: {'✅ Yes' if current_channel_valid else '❌ No'}",
        inline=True)

    # Active beast check
    if user.active_beast_id:
        user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
        active_beast = None
        for beast_id, beast in user_beasts:
            if beast_id == user.active_beast_id:
                active_beast = beast
                break

        if active_beast:
            embed.add_field(
                name="🐉 Active Beast",
                value=
                f"**{active_beast.name}** {active_beast.rarity.emoji}\nLevel {active_beast.stats.level} | ID: #{user.active_beast_id}",
                inline=True)
        else:
            embed.add_field(
                name="🐉 Active Beast",
                value=
                "❌ **INVALID ID**\nYour active beast ID doesn't match any beast!",
                inline=True)
    else:
        embed.add_field(
            name="🐉 Active Beast",
            value="❌ **NONE SET**\nSet an active beast to gain XP!",
            inline=True)

    # Cooldown check
    cooldown_remaining = 0
    if user.last_xp_gain:
        time_since_last = datetime.datetime.now() - user.last_xp_gain
        cooldown_remaining = max(
            0, ctx.bot.config.xp_cooldown_seconds -
            time_since_last.total_seconds())

        if cooldown_remaining > 0:
            embed.add_field(
                name="⏰ XP Cooldown",
                value=
                f"❌ **{int(cooldown_remaining)}s remaining**\nNext XP available in {int(cooldown_remaining)} seconds",
                inline=False)
        else:
            embed.add_field(name="⏰ XP Cooldown",
                            value="✅ **Ready to gain XP**\nNo cooldown active",
                            inline=False)
    else:
        embed.add_field(name="⏰ XP Cooldown",
                        value="✅ **Ready to gain XP**\nFirst time gaining XP",
                        inline=False)

    # Anti-spam information
    embed.add_field(
        name="🛡️ Anti-Spam Rules",
        value=
        f"**Minimum Length:** {ctx.bot.config.min_message_length} characters\n"
        f"**No Repetitive Messages:** Max {ctx.bot.config.max_similar_messages} similar\n"
        f"**Rate Limit:** Max {ctx.bot.config.message_frequency_limit} messages/2min\n"
        f"**No Spam Patterns:** caps, symbols, farming words",
        inline=False)

    # XP settings
    embed.add_field(
        name="⚙️ XP Settings",
        value=f"**XP per Message:** {ctx.bot.config.xp_per_message}\n"
        f"**Cooldown:** {ctx.bot.config.xp_cooldown_seconds}s\n"
        f"**Role Bonuses:** Special +30%, Personal +60%",
        inline=True)

    # Recommendations
    recommendations = []
    if not user.active_beast_id:
        recommendations.append(
            f"🎯 Use `{ctx.bot.config.prefix}active <beast_id>` to set active beast"
        )
    if not current_channel_valid:
        recommendations.append("📍 Move to an XP-enabled channel")
    if cooldown_remaining > 0:
        recommendations.append(
            f"⏰ Wait {int(cooldown_remaining)}s for cooldown")

    recommendations.append("💬 **Write meaningful messages** - avoid spam!")
    recommendations.append("📝 **Minimum 5 characters** per message")
    recommendations.append("🚫 **Avoid repetitive content** or farming phrases")

    if recommendations:
        embed.add_field(name="💡 Recommendations",
                        value="\n".join(recommendations),
                        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='removexp')
@commands.has_permissions(administrator=True)
async def remove_xp_channel(ctx,
                            channel: Optional[
                                discord.TextChannel] = None):  # ✅ Fixed
    """Remove a channel from XP gain list (admin only)"""
    if channel is None:
        channel = ctx.channel
        # ✅ ADDED: Ensure ctx.channel is actually a TextChannel
        if not isinstance(channel, discord.TextChannel):
            embed = discord.Embed(
                title="❌ Invalid Channel",
                description="This command can only be used in text channels.",
                color=0xFF0000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

    if channel.id in ctx.bot.config.xp_chat_channel_ids:
        ctx.bot.config.xp_chat_channel_ids.remove(channel.id)
        embed = discord.Embed(
            title="✅ XP Channel Removed",
            description=f"{channel.mention} has been removed from XP channels",
            color=0x00FF00)
    else:
        embed = discord.Embed(
            title="⚠️ Not Found",
            description=f"{channel.mention} is not in the XP channel list",
            color=0xFFAA00)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# REPLACE the existing fixcooldown command with this enhanced version:


@commands.command(name='fixcooldown', aliases=['setcooldown'])
@commands.has_permissions(administrator=True)
async def fix_xp_cooldown(ctx, new_cooldown: Optional[int] = None):  # ✅ Fixed
    """Temporarily adjust XP cooldown (admin only)"""

    # If no cooldown provided, show current setting
    if new_cooldown is None:
        embed = discord.Embed(
            title="⚙️ Current XP Cooldown",
            description=
            f"Current XP cooldown: **{ctx.bot.config.xp_cooldown_seconds} seconds**",
            color=0x00AAFF)
        embed.add_field(
            name="Usage",
            value=
            f"`{ctx.bot.config.prefix}fixcooldown <seconds>`\nExample: `{ctx.bot.config.prefix}fixcooldown 10`",
            inline=False)
        embed.add_field(name="Valid Range",
                        value="0 to 300 seconds",
                        inline=True)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Validate cooldown range
    if new_cooldown < 0 or new_cooldown > 300:
        embed = discord.Embed(
            title="❌ Invalid Cooldown",
            description=
            f"Cooldown must be between **0** and **300** seconds.\nYou entered: **{new_cooldown}**",
            color=0xFF0000)
        embed.add_field(
            name="Examples",
            value=
            "• `!fixcooldown 5` - Very fast XP gain\n• `!fixcooldown 15` - Balanced XP gain\n• `!fixcooldown 60` - Slow XP gain",
            inline=False)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    try:
        old_cooldown = ctx.bot.config.xp_cooldown_seconds
        ctx.bot.config.xp_cooldown_seconds = new_cooldown

        embed = discord.Embed(
            title="✅ XP Cooldown Updated Successfully!",
            description=
            f"XP cooldown changed from **{old_cooldown}s** to **{new_cooldown}s**",
            color=0x00FF00)

        # Show the impact
        if new_cooldown < old_cooldown:
            impact = "⚡ **Faster XP gain** - Users can gain XP more frequently"
        elif new_cooldown > old_cooldown:
            impact = "🐌 **Slower XP gain** - Users must wait longer between XP"
        else:
            impact = "🔄 **No change** - Same cooldown as before"

        embed.add_field(name="Impact", value=impact, inline=False)
        embed.add_field(
            name="⚠️ Important Note",
            value=
            "This change is **temporary** and will reset when the bot restarts.\nUpdate your environment variables for permanent change.",
            inline=False)

        # Show next steps
        embed.add_field(
            name="💡 Test the Change",
            value=
            f"• Use `{ctx.bot.config.prefix}xpstatus` to check XP status\n• Have users chat in XP channels to test\n• Look for ⚡ reactions on messages",
            inline=False)

        await ctx.bot.safe_send_message(ctx.channel, embed=embed)

        # Log the change
        print(
            f"XP cooldown changed by {ctx.author} from {old_cooldown}s to {new_cooldown}s"
        )

    except Exception as e:
        embed = discord.Embed(
            title="❌ Error Updating Cooldown",
            description=
            f"An error occurred while updating the cooldown: {str(e)}",
            color=0xFF0000)
        embed.add_field(
            name="Debug Info",
            value=
            f"Old cooldown: {getattr(ctx.bot.config, 'xp_cooldown_seconds', 'Unknown')}\nNew cooldown: {new_cooldown}",
            inline=False)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# ALSO ADD this simple command to check current settings:


@commands.command(name='cooldown', aliases=['getcooldown'])
async def check_cooldown(ctx):
    """Check current XP cooldown setting"""
    try:
        current_cooldown = ctx.bot.config.xp_cooldown_seconds

        embed = discord.Embed(
            title="⏰ XP Cooldown Status",
            description=f"Current XP cooldown: **{current_cooldown} seconds**",
            color=0x00AAFF)

        # Provide context for the cooldown
        if current_cooldown <= 5:
            speed_rating = "🔥 **Very Fast** - Almost instant XP gain"
        elif current_cooldown <= 15:
            speed_rating = "⚡ **Fast** - Quick XP gain"
        elif current_cooldown <= 30:
            speed_rating = "⚖️ **Balanced** - Moderate XP gain"
        elif current_cooldown <= 60:
            speed_rating = "🐌 **Slow** - Patient XP gain"
        else:
            speed_rating = "🛑 **Very Slow** - Rare XP gain"

        embed.add_field(name="Speed Rating", value=speed_rating, inline=False)
        embed.add_field(
            name="What This Means",
            value=
            f"Users must wait **{current_cooldown} seconds** between XP gains from chatting",
            inline=False)

        if ctx.author.guild_permissions.administrator:
            embed.add_field(
                name="🔧 Admin Options",
                value=
                f"`{ctx.bot.config.prefix}fixcooldown <seconds>` - Change cooldown\n`{ctx.bot.config.prefix}fixcooldown` - Show current setting",
                inline=False)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Could not retrieve cooldown setting: {str(e)}",
            color=0xFF0000)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# ADD this command to test if commands are working:


@commands.command(name='ping')
async def ping_command(ctx):
    """Simple ping command to test bot responsiveness"""
    import time
    start_time = time.time()

    embed = discord.Embed(title="🏓 Pong!",
                          description="Bot is responding normally",
                          color=0x00FF00)

    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    end_time = time.time()
    response_time = (end_time - start_time) * 1000

    embed.add_field(name="Response Time",
                    value=f"{response_time:.0f}ms",
                    inline=True)

    embed.add_field(name="Bot Latency",
                    value=f"{ctx.bot.latency * 1000:.0f}ms",
                    inline=True)

    await ctx.bot.safe_edit_message(message, embed=embed)


@commands.command(name='nextspawn')
async def next_spawn_time(ctx):
    """Check when the next beast will spawn"""
    if not hasattr(ctx.bot, '_next_spawn_time'):
        embed = discord.Embed(
            title="⏰ Next Beast Spawn",
            description=
            "Spawn timing not initialized yet. Please wait a moment.",
            color=0xFFAA00)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    time_until_spawn = ctx.bot._next_spawn_time - datetime.datetime.now()
    minutes_left = int(time_until_spawn.total_seconds() / 60)

    if minutes_left <= 0:
        embed = discord.Embed(title="🎯 Next Beast Spawn",
                              description="A beast should spawn very soon!",
                              color=0x00FF00)
    else:
        embed = discord.Embed(
            title="⏰ Next Beast Spawn",
            description=
            f"Next wild beast will spawn in approximately **{minutes_left} minutes**",
            color=0x00AAFF)

        # OLD CODE: Multiple spawn channels
        # NEW CODE: Single spawn channel
        if ctx.bot.spawn_channel_id:
            channel = ctx.bot.get_channel(ctx.bot.spawn_channel_id)
            if channel:
                embed.add_field(name="📍 Spawn Location",
                                value=f"#{channel.name}",
                                inline=True)
            else:
                embed.add_field(name="📍 Spawn Location",
                                value="Channel not found",
                                inline=True)
        else:
            embed.add_field(name="📍 Spawn Location",
                            value="Not configured",
                            inline=True)

        embed.add_field(name="🎲 Spawn Interval", value="45 min", inline=True)

    # NEW: Show if there's currently a beast to catch
    if ctx.bot.current_spawned_beast:
        beast = ctx.bot.current_spawned_beast
        embed.add_field(
            name="🐉 Current Beast Available",
            value=f"{beast.name} {beast.rarity.emoji} - Go catch it!",
            inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='cloudbackup')
@commands.has_permissions(administrator=True)
async def manual_cloud_backup(ctx):
    """Create a manual backup with cloud storage"""
    embed = discord.Embed(
        title="☁️ Creating Cloud Backup...",
        description="Creating backup and uploading to cloud storage.",
        color=0xFFAA00)
    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    try:
        backup_file = await ctx.bot.safe_api_call(
            ctx.bot.backup_manager.create_backup_with_cloud_storage)
        if backup_file:
            embed = discord.Embed(
                title="✅ Cloud Backup Created",
                description="Database backup created and uploaded to cloud!",
                color=0x00FF00)
            embed.add_field(name="📁 Local File",
                            value=f"`{Path(backup_file).name}`",
                            inline=True)

            if ctx.bot.backup_manager.github_repo:
                embed.add_field(name="☁️ Cloud Storage",
                                value="✅ Uploaded to GitHub",
                                inline=True)
            else:
                embed.add_field(name="☁️ Cloud Storage",
                                value="❌ Not configured",
                                inline=True)
        else:
            embed = discord.Embed(
                title="❌ Backup Failed",
                description="Failed to create backup. Check logs for details.",
                color=0xFF0000)
    except Exception as e:
        embed = discord.Embed(title="❌ Backup Error",
                              description=f"An error occurred: {str(e)}",
                              color=0xFF0000)

    await ctx.bot.safe_edit_message(message, embed=embed)


@commands.command(name='restorebackup')
@commands.has_permissions(administrator=True)
async def restore_from_cloud(ctx):
    """Restore database from cloud backup"""
    embed = discord.Embed(
        title="⚠️ Restore Confirmation",
        description=
        "This will replace your current database with the latest cloud backup.\n"
        "**ALL CURRENT DATA WILL BE LOST!**\n\n"
        "React with ✅ to confirm or ❌ to cancel.",
        color=0xFF8800)
    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)
    await ctx.bot.safe_add_reaction(message, "✅")
    await ctx.bot.safe_add_reaction(message, "❌")

    def check(reaction, user):
        return (user == ctx.author and str(reaction.emoji) in ["✅", "❌"]
                and reaction.message.id == message.id)

    try:
        reaction, user = await ctx.bot.wait_for('reaction_add',
                                                timeout=30.0,
                                                check=check)

        if str(reaction.emoji) == "✅":
            embed = discord.Embed(
                title="⬇️ Restoring from Cloud...",
                description=
                "Downloading and restoring backup from cloud storage.",
                color=0xFFAA00)
            await ctx.bot.safe_edit_message(message, embed=embed)

            success = await ctx.bot.safe_api_call(
                ctx.bot.backup_manager.restore_from_cloud)
            if success:
                embed = discord.Embed(
                    title="✅ Restore Complete",
                    description="Database has been restored from cloud backup!\n"
                    "**Bot restart recommended.**",
                    color=0x00FF00)
            else:
                embed = discord.Embed(
                    title="❌ Restore Failed",
                    description=
                    "Failed to restore from cloud backup. Check logs for details.",
                    color=0xFF0000)
        else:
            embed = discord.Embed(
                title="❌ Restore Cancelled",
                description="Database restore has been cancelled.",
                color=0x808080)

    except asyncio.TimeoutError:
        embed = discord.Embed(
            title="⏰ Timeout",
            description="Restore confirmation timed out. Operation cancelled.",
            color=0x808080)

    await ctx.bot.safe_edit_message(message, embed=embed)


@commands.command(name='adopt')
@require_channel("adopt")
async def adopt_beast(ctx):
    """Adopt a random beast (available every 2 days)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=
            f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    now = datetime.datetime.now()
    if user.last_adopt:
        cooldown_hours = ctx.bot.config.adopt_cooldown_hours
        time_since_last = now - user.last_adopt
        if time_since_last.total_seconds() < cooldown_hours * 3600:
            remaining_time = datetime.timedelta(
                hours=cooldown_hours) - time_since_last
            hours = int(remaining_time.total_seconds() // 3600)
            minutes = int((remaining_time.total_seconds() % 3600) // 60)

            embed = discord.Embed(
                title="⏰ Adopt Cooldown",
                description=
                f"You can adopt another beast in {hours}h {minutes}m",
                color=0xFF8000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

    template = ctx.bot.template_manager.get_random_template_up_to_rarity(
        BeastRarity.LEGENDARY)
    beast = template.create_beast()
    beast.owner_id = ctx.author.id

    beast_id = await ctx.bot.db.add_beast(beast)

    user.last_adopt = now
    user.total_catches += 1
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🎉 Beast Adopted!",
        description=
        f"**{ctx.author.display_name}** adopted **{beast.name}**!\n{beast.rarity.emoji}",
        color=beast.rarity.color)
    embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
    embed.add_field(name="Level", value=beast.stats.level, inline=True)
    embed.add_field(name="Power Level", value=beast.power_level, inline=True)
    embed.add_field(name="Tendency",
                    value=beast.tendency or "None",
                    inline=False)
    embed.add_field(
        name="Next Adopt",
        value=
        f"<t:{int((now + datetime.timedelta(hours=ctx.bot.config.adopt_cooldown_hours)).timestamp())}:R>",
        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='beasts', aliases=['inventory', 'inv'])
async def show_beasts(ctx, page: int = 1):
    """Show your beast collection"""
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    if not user_beasts:
        embed = discord.Embed(
            title="📦 Empty Beast Collection",
            description=
            f"You don't have any beasts yet!\nUse `{ctx.bot.config.prefix}adopt` or catch wild beasts to start your collection.",
            color=0x808080)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Pagination
    per_page = 5
    total_pages = (len(user_beasts) + per_page - 1) // per_page
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_beasts = user_beasts[start_idx:end_idx]

    embed = discord.Embed(
        title=f"🏛️ {ctx.author.display_name}'s Beast Collection",
        description=
        f"**{len(user_beasts)}/{beast_limit}** beasts | Page {page}/{total_pages}",
        color=0x00AAFF)

    for beast_id, beast in page_beasts:
        active_indicator = "🟢 " if beast_id == user.active_beast_id else ""
        embed.add_field(
            name=
            f"{active_indicator}#{beast_id} {beast.name} {beast.rarity.emoji}",
            value=
            f"Level {beast.stats.level} | HP: {beast.stats.hp}/{beast.stats.max_hp}\n"
            f"Power: {beast.power_level} | Location: {beast.location}",
            inline=False)

    embed.set_footer(
        text=
        f"💰 {user.spirit_stones:,} Beast Stones | Use {ctx.bot.config.prefix}beast <id> for details"
    )
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='help')
async def help_command(ctx):
    """Show comprehensive help information"""
    embed = discord.Embed(
        title="🐉 Immortal Beasts Bot - Complete Guide",
        description=
        "Collect, train, and battle mythical beasts in an epic adventure!",
        color=0x00AAFF)

    # 📦 BEAST COLLECTION
    embed.add_field(
        name="📦 **Beast Collection**",
        value=
        f"`{ctx.bot.config.prefix}adopt` - Adopt a random beast (48h cooldown)\n"
        f"`{ctx.bot.config.prefix}catch` - Catch wild spawned beasts\n"
        f"`{ctx.bot.config.prefix}beasts` - View your beast collection\n"
        f"`{ctx.bot.config.prefix}beast <id>` - Detailed beast information\n"
        f"`{ctx.bot.config.prefix}active <id>` - Set active beast for XP gain",
        inline=False)

    # ⚔️ BATTLE SYSTEM
    embed.add_field(
        name="⚔️ **Battle & Combat**",
        value=
        f"`{ctx.bot.config.prefix}battle @user` - Challenge another trainer\n"
        f"`{ctx.bot.config.prefix}heal <id>` - Heal a specific beast (50 stones)\n"
        f"`{ctx.bot.config.prefix}healall` - Heal all your beasts\n"
        f"`{ctx.bot.config.prefix}sacrifice <id>` - Sacrifice a beast for beast stones, XP, and (with restrictions) stat boosts!\n"
        f"`{ctx.bot.config.prefix}release <id>` - Release beast peacefully",
        inline=False)

    # 🎯 NEW: DAILY TRAINING SYSTEM
    embed.add_field(
        name="🎯 **Daily Training & Quests**",
        value=
        f"`{ctx.bot.config.prefix}feed` - Feed your active beast (+300 XP, 24h cooldown)\n"
        f"`{ctx.bot.config.prefix}groom` - Groom your active beast (+300 XP, 24h cooldown)\n"
        f"`{ctx.bot.config.prefix}spar <beast_id> <type>` - Train beast (+300 XP + stats, 2h cooldown)\n"
        f"`{ctx.bot.config.prefix}dailies` - Check your daily quest progress\n"
        f"💡 **Spar Types:** strength, endurance, agility, defense",
        inline=False)

    # 🔥 NEW: XP BOOST SYSTEM
    embed.add_field(
        name="🔥 **XP Boost System**",
        value=
        f"🏆 **First Win Bonus:** Win your first battle of the day for 50% XP boost (2 hours)\n"
        f"⚡ **Boost applies to:** Feed, Groom, Spar, and Chat XP\n"
        f"🎯 **Strategy:** Win a battle first, then do daily training for maximum XP!\n"
        f"📊 **Daily Potential:** 4,800 XP normal | 7,200 XP with boost",
        inline=False)

    # 💰 ECONOMY & REWARDS
    embed.add_field(
        name="💰 **Economy & Resources**",
        value=
        f"`{ctx.bot.config.prefix}stone` - Daily beast stones (24h cooldown)\n"
        f"`{ctx.bot.config.prefix}balance` - Check your beast stones\n"
        f"💎 **Beast Stones Uses:** Healing beasts (50 per heal)",
        inline=False)

    # ⭐ SPECIAL ADOPTIONS
    embed.add_field(
        name="⭐ **Special Adoptions** (Role-Based)",
        value=
        f"`{ctx.bot.config.prefix}adoptlegend` - Guaranteed legendary (special roles)\n"
        f"`{ctx.bot.config.prefix}adoptmythic` - Guaranteed mythic (personal role)\n"
        f"`{ctx.bot.config.prefix}adoptstatus` - Check adoption cooldowns",
        inline=False)

    # 📊 STATS & LEADERBOARDS
    embed.add_field(
        name="📊 **Statistics & Rankings**",
        value=
        f"`{ctx.bot.config.prefix}userstats [@user]` - View player statistics\n"
        f"`{ctx.bot.config.prefix}leaderboard [category]` - Top players\n"
        f"`{ctx.bot.config.prefix}serverbeasts` - Server beast distribution\n"
        f"`{ctx.bot.config.prefix}checkbeasts @user` - Check user's beasts (admin)",
        inline=False)

    # ⚡ UPDATED: XP SYSTEM
    embed.add_field(
        name="⚡ **XP & Training System**",
        value=f"`{ctx.bot.config.prefix}xpstatus` - Check your XP gain status\n"
        f"`{ctx.bot.config.prefix}cooldown` - View XP cooldown remaining\n"
        f"💬 **Message XP:** 25 XP per message (60s cooldown)\n"
        f"🎯 **Role Bonuses:** Special roles get XP multipliers!\n"
        f"🔥 **NEW:** XP boost system for enhanced training",
        inline=False)

    # 🔧 ADMIN COMMANDS
    embed.add_field(
        name="🔧 **Admin Commands**",
        value=f"`{ctx.bot.config.prefix}forcespawn` - Force beast spawn\n"
        f"`{ctx.bot.config.prefix}nextspawn` - Check spawn timer\n"
        f"`{ctx.bot.config.prefix}addxpchannel [#channel]` - Add XP channel\n"
        f"`{ctx.bot.config.prefix}removexp [#channel]` - Remove XP channel\n"
        f"`{ctx.bot.config.prefix}fixcooldown <seconds>` - Adjust XP cooldown\n"
        f"`{ctx.bot.config.prefix}antispam <setting> <value>` - Configure anti-spam\n"
        f"`{ctx.bot.config.prefix}backup` - Manual database backup\n"
        f"`{ctx.bot.config.prefix}channels` - Show channel configuration",
        inline=False)

    # 🌟 BEAST RARITIES
    embed.add_field(name="🌟 **Beast Rarity System**",
                    value="⭐ **Common** - Starting companions\n"
                    "⭐⭐ **Uncommon** - Reliable partners\n"
                    "⭐⭐⭐ **Rare** - Powerful allies\n"
                    "⭐⭐⭐⭐ **Epic** - Elite warriors\n"
                    "⭐⭐⭐⭐⭐ **Legendary** - Mythical beings\n"
                    "⭐⭐⭐⭐⭐⭐ **Mythic** - Divine creatures",
                    inline=False)

    # 💡 UPDATED: TIPS & TRICKS
    embed.add_field(
        name="💡 **Pro Tips & Strategy**",
        value="🎯 Set an active beast to gain XP from chatting\n"
        "🏆 **WIN A BATTLE FIRST** for 50% XP boost on daily training\n"
        "📅 Complete daily quests: Feed → Groom → Spar training\n"
        "⚔️ Heal your beasts after battles (50 stones each)\n"
        "📈 Higher rarity = better stats and XP scaling\n"
        "💎 Use `!dailies` to track your progress\n"
        "🔥 Time your training during XP boost for maximum gains",
        inline=False)

    # UPDATED FOOTER INFORMATION
    embed.set_footer(
        text=
        f"📱 Total Commands: 30+ | 🎮 Spawn every 45min | 💬 Daily XP: 4,800-7,200 | 🔥 First win = 50% boost!"
    )
    embed.timestamp = discord.utils.utcnow()

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='beast', aliases=['info'])
async def beast_info(ctx, beast_id: int):
    """Show detailed information about a specific beast"""
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    target_beast = None
    for bid, beast in user_beasts:
        if bid == beast_id:
            target_beast = beast
            break

    if not target_beast:
        embed = discord.Embed(
            title="❌ Beast Not Found",
            description=f"You don't own a beast with ID #{beast_id}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    embed = discord.Embed(
        title=f"🐉 {target_beast.name} {target_beast.rarity.emoji}",
        description=target_beast.description or "A mysterious beast",
        color=target_beast.rarity.color)

    # Status indicator
    status = "🟢 Active" if beast_id == user.active_beast_id else "⚪ Inactive"
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
    embed.add_field(name="Rarity",
                    value=target_beast.rarity.name.title(),
                    inline=True)

    # Stats
    embed.add_field(name="Level", value=target_beast.stats.level, inline=True)
    embed.add_field(
        name="Experience",
        value=
        f"{target_beast.stats.exp}/{target_beast.stats.get_level_up_requirements(target_beast.rarity)}",
        inline=True)
    embed.add_field(name="Power Level",
                    value=target_beast.power_level,
                    inline=True)

    # Detailed stats
    embed.add_field(
        name="📊 Combat Stats",
        value=f"**HP:** {target_beast.stats.hp}/{target_beast.stats.max_hp}\n"
        f"**Attack:** {target_beast.stats.attack}\n"
        f"**Defense:** {target_beast.stats.defense}\n"
        f"**Speed:** {target_beast.stats.speed}",
        inline=True)

    # Location and tendency
    embed.add_field(name="🌍 Origin",
                    value=f"**Location:** {target_beast.location}\n"
                    f"**Tendency:** {target_beast.tendency}",
                    inline=True)

    # Caught date
    caught_date = target_beast.caught_at.strftime("%Y-%m-%d %H:%M")
    embed.add_field(name="📅 Caught", value=caught_date, inline=True)

    embed.set_footer(
        text=
        f"Use {ctx.bot.config.prefix}active {beast_id} to set as active beast")
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='active')
async def set_active_beast(ctx, beast_id: Optional[int] = None):
    """Set a beast as your active beast for XP gain, or see current active beast"""
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    # If no beast_id provided, show current active + auto-selection info
    if beast_id is None:
        if not user_beasts:
            embed = discord.Embed(
                title="📦 No Beasts Available",
                description=
                "You don't have any beasts yet! Use `!adopt` or catch wild beasts.",
                color=0xFF0000)
        elif not user.active_beast_id:
            # Show what would be auto-selected
            auto_beast = await ctx.bot.auto_select_active_beast(ctx.author.id)
            if auto_beast:
                beast_id_auto, beast = auto_beast
                embed = discord.Embed(
                    title="🤖 No Active Beast Set",
                    description=
                    (f"**Auto-Selection Available:** {beast.name} {beast.rarity.emoji}\n"
                     f"This beast will be automatically selected when you chat in XP channels.\n\n"
                     f"Use `!active {beast_id_auto}` to set it manually now, or chat to auto-select!"
                     ),
                    color=0xFFAA00)
            else:
                embed = discord.Embed(
                    title="❌ No Active Beast",
                    description="No beasts available for selection.",
                    color=0xFF0000)
        else:
            # Show current active beast
            embed = discord.Embed(
                title="❌ No Active Beast Found",
                description=
                "Your active beast ID doesn't match any beast in your collection.",
                color=0xFF0000)

            for bid, beast in user_beasts:
                if bid == user.active_beast_id:
                    embed = discord.Embed(
                        title="🟢 Current Active Beast",
                        description=f"**{beast.name}** {beast.rarity.emoji}",
                        color=beast.rarity.color)
                    embed.add_field(name="Beast ID",
                                    value=f"#{bid}",
                                    inline=True)
                    embed.add_field(name="Level",
                                    value=beast.stats.level,
                                    inline=True)
                    embed.add_field(name="Power",
                                    value=beast.power_level,
                                    inline=True)
                    break

        embed.add_field(name="💡 Usage",
                        value=(f"`!active <beast_id>` - Set specific beast\n"
                               f"`!beasts` - View all your beasts"),
                        inline=False)

        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Rest of the existing manual selection logic...
    target_beast = None
    for bid, beast in user_beasts:
        if bid == beast_id:
            target_beast = beast
            break

    if not target_beast:
        embed = discord.Embed(
            title="❌ Beast Not Found",
            description=f"You don't own a beast with ID #{beast_id}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    user.active_beast_id = beast_id
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🟢 Active Beast Set!",
        description=(f"**{target_beast.name}** is now your active beast!\n"
                     f"It will gain XP when you chat in XP channels"),
        color=target_beast.rarity.color)
    embed.add_field(
        name="Beast",
        value=f"#{beast_id} {target_beast.name} {target_beast.rarity.emoji}",
        inline=True)
    embed.add_field(name="Level", value=target_beast.stats.level, inline=True)
    embed.add_field(name="XP per Message",
                    value=f"+{ctx.bot.config.xp_per_message}",
                    inline=True)
    embed.add_field(
        name="🤖 Auto-Selection",
        value=
        "Manual selection overrides auto-selection. Your choice will be remembered!",
        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='balance', aliases=['stones'])
async def show_balance(ctx):
    """Show your beast stone balance"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    embed = discord.Embed(
        title="💰 Beast Stone Balance",
        description=
        f"**{ctx.author.display_name}** has **{user.spirit_stones:,}** Beast Stones",
        color=0x9932CC)

    # Next daily claim time
    if user.last_daily:
        next_daily = user.last_daily + datetime.timedelta(hours=24)
        if datetime.datetime.now() < next_daily:
            embed.add_field(name="⏰ Next Daily",
                            value=f"<t:{int(next_daily.timestamp())}:R>",
                            inline=True)
        else:
            embed.add_field(name="✅ Daily Available",
                            value="Use !stone to claim",
                            inline=True)
    else:
        embed.add_field(name="✅ Daily Available",
                        value="Use !stone to claim",
                        inline=True)

    embed.add_field(
        name="📊 Stats",
        value=
        f"Total Catches: {user.total_catches}\nBattles: {user.total_battles}\nWin Rate: {user.win_rate:.1f}%",
        inline=True)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# Add this new heal command to your commands section


@commands.command(name='heal')
async def heal_beast(ctx, beast_id: int):
    """Heal a beast to full HP for 50 beast stones"""
    HEAL_COST = 50

    # Get user data
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    # Check if user has enough beast stones
    if user.spirit_stones < HEAL_COST:
        embed = discord.Embed(
            title="❌ Insufficient Beast Stones",
            description=
            f"You need **{HEAL_COST} beast stones** to heal a beast!\n"
            f"You currently have **{user.spirit_stones} stones**.\n"
            f"Use `{ctx.bot.config.prefix}stone` to get daily stones.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Get user's beasts
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)

    # Find the target beast
    target_beast = None
    target_beast_id = None
    for bid, beast in user_beasts:
        if bid == beast_id:
            target_beast = beast
            target_beast_id = bid
            break

    if not target_beast:
        embed = discord.Embed(
            title="❌ Beast Not Found",
            description=f"You don't own a beast with ID #{beast_id}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check if beast is already at full HP
    if target_beast.stats.hp >= target_beast.stats.max_hp:
        embed = discord.Embed(
            title="💚 Beast Already Healthy",
            description=f"**{target_beast.name}** is already at full HP!\n"
            f"HP: {target_beast.stats.hp}/{target_beast.stats.max_hp}",
            color=0x00FF00)
        embed.add_field(name="💰 Beast Stones",
                        value=f"No stones spent: {user.spirit_stones}",
                        inline=True)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Calculate healing amount
    old_hp = target_beast.stats.hp
    healing_amount = target_beast.stats.heal()  # Heal to full HP

    # Deduct beast stones
    user.spirit_stones -= HEAL_COST

    # Update database
    await ctx.bot.db.update_user(user)
    await ctx.bot.db.update_beast(target_beast_id, target_beast)

    # Create success embed
    embed = discord.Embed(
        title="✨ Beast Healed!",
        description=
        f"**{target_beast.name}** has been fully healed!\n{target_beast.rarity.emoji}",
        color=target_beast.rarity.color)

    embed.add_field(name="🐉 Beast Info",
                    value=f"**Name:** {target_beast.name}\n"
                    f"**ID:** #{beast_id}\n"
                    f"**Level:** {target_beast.stats.level}",
                    inline=True)

    embed.add_field(
        name="❤️ HP Restored",
        value=f"**Before:** {old_hp}/{target_beast.stats.max_hp}\n"
        f"**After:** {target_beast.stats.hp}/{target_beast.stats.max_hp}\n"
        f"**Healed:** +{healing_amount} HP",
        inline=True)

    embed.add_field(name="💰 Cost",
                    value=f"**Spent:** {HEAL_COST} stones\n"
                    f"**Remaining:** {user.spirit_stones} stones",
                    inline=True)

    embed.set_footer(
        text=
        f"Use {ctx.bot.config.prefix}beast {beast_id} to view full beast details"
    )
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# Also add this enhanced heal_all command as a bonus feature


@commands.command(name='healall')
async def heal_all_beasts(ctx):
    """Heal ALL your beasts to full HP (costs 50 stones per damaged beast)"""
    HEAL_COST = 50

    # Get user data
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)

    if not user_beasts:
        embed = discord.Embed(title="📦 No Beasts",
                              description="You don't have any beasts to heal!",
                              color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Find damaged beasts
    damaged_beasts = []
    for beast_id, beast in user_beasts:
        if beast.stats.hp < beast.stats.max_hp:
            damaged_beasts.append((beast_id, beast))

    if not damaged_beasts:
        embed = discord.Embed(
            title="💚 All Beasts Healthy",
            description="All your beasts are already at full HP!",
            color=0x00FF00)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Calculate total cost
    total_cost = len(damaged_beasts) * HEAL_COST

    # Check if user has enough stones
    if user.spirit_stones < total_cost:
        embed = discord.Embed(
            title="❌ Insufficient Beast Stones",
            description=
            f"You have **{len(damaged_beasts)} damaged beasts** requiring **{total_cost} stones** to heal.\n"
            f"You only have **{user.spirit_stones} stones**.\n\n"
            f"Use `{ctx.bot.config.prefix}heal <beast_id>` to heal individual beasts.",
            color=0xFF0000)

        # Show damaged beasts
        damaged_list = []
        for beast_id, beast in damaged_beasts[:5]:  # Show first 5
            damaged_list.append(
                f"#{beast_id} {beast.name} ({beast.stats.hp}/{beast.stats.max_hp} HP)"
            )

        embed.add_field(name="🩹 Damaged Beasts",
                        value="\n".join(damaged_list) +
                        (f"\n... and {len(damaged_beasts)-5} more"
                         if len(damaged_beasts) > 5 else ""),
                        inline=False)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Heal all damaged beasts
    total_healing = 0
    healed_beasts = []

    for beast_id, beast in damaged_beasts:
        old_hp = beast.stats.hp
        healing_amount = beast.stats.heal()
        total_healing += healing_amount
        healed_beasts.append((beast_id, beast, old_hp, healing_amount))
        await ctx.bot.db.update_beast(beast_id, beast)

    # Deduct total cost
    user.spirit_stones -= total_cost
    await ctx.bot.db.update_user(user)

    # Create success embed
    embed = discord.Embed(
        title="✨ Mass Healing Complete!",
        description=f"Healed **{len(damaged_beasts)} beasts** to full HP!",
        color=0x00FF00)

    embed.add_field(name="📊 Healing Summary",
                    value=f"**Beasts Healed:** {len(damaged_beasts)}\n"
                    f"**Total HP Restored:** {total_healing}\n"
                    f"**Cost:** {total_cost} stones",
                    inline=True)

    embed.add_field(name="💰 Remaining Stones",
                    value=f"{user.spirit_stones} stones",
                    inline=True)

    # Show first few healed beasts
    if len(healed_beasts) <= 3:
        for beast_id, beast, old_hp, healing in healed_beasts:
            embed.add_field(
                name=f"#{beast_id} {beast.name}",
                value=f"{old_hp}→{beast.stats.max_hp} HP (+{healing})",
                inline=True)
    else:
        # Show summary for many beasts
        embed.add_field(
            name="🩹 Healed Beasts",
            value=
            f"All {len(healed_beasts)} damaged beasts restored to full health!",
            inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# USER STATS AND LEADERBOARD COMMANDS
# Add these commands to your main.py file (around line 2600, after your heal commands)


@commands.command(name='checkbeasts', aliases=['userbeasts'])
@commands.has_permissions(administrator=True)
async def check_user_beasts(ctx, user: discord.Member):
    """Check another user's beast collection (admin only)"""
    try:
        # Get user's beasts
        user_beasts = await ctx.bot.db.get_user_beasts(user.id)
        user_data = await ctx.bot.get_or_create_user(user.id, str(user))
        user_role = ctx.bot.role_manager.get_user_role(user)
        beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

        if not user_beasts:
            embed = discord.Embed(
                title="📦 Empty Beast Collection",
                description=f"{user.display_name} doesn't have any beasts yet.",
                color=0x808080)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Create detailed embed
        embed = discord.Embed(
            title=f"🔍 {user.display_name}'s Beast Collection (Admin View)",
            description=
            f"**{len(user_beasts)}/{beast_limit}** beasts | Requested by {ctx.author.display_name}",
            color=0x00AAFF)

        # Show up to 10 beasts in detail
        display_beasts = user_beasts[:10]
        for beast_id, beast in display_beasts:
            active_indicator = "🟢 " if beast_id == user_data.active_beast_id else ""
            embed.add_field(
                name=
                f"{active_indicator}#{beast_id} {beast.name} {beast.rarity.emoji}",
                value=
                f"**Level:** {beast.stats.level} | **HP:** {beast.stats.hp}/{beast.stats.max_hp}\n"
                f"**Power:** {beast.power_level} | **Location:** {beast.location}\n"
                f"**Caught:** {beast.caught_at.strftime('%Y-%m-%d')}",
                inline=False)

        if len(user_beasts) > 10:
            embed.add_field(
                name="📋 Additional Beasts",
                value=f"... and {len(user_beasts) - 10} more beasts",
                inline=False)

        # User stats summary
        embed.add_field(
            name="📊 User Stats",
            value=f"**Beast Stones:** {user_data.spirit_stones:,}\n"
            f"**Total Catches:** {user_data.total_catches}\n"
            f"**Battles:** {user_data.total_battles} | **Win Rate:** {user_data.win_rate:.1f}%",
            inline=True)

        # Special privileges
        privileges = []
        if user_data.has_used_adopt_legend:
            privileges.append("✅ Used Legend Adopt")
        else:
            privileges.append(
                "⭐ Legend Adopt Available" if ctx.bot.role_manager.
                can_use_adopt_legend(user_role) else "❌ No Legend Access")

        if user_data.has_used_adopt_mythic:
            privileges.append("✅ Used Mythic Adopt")
        else:
            privileges.append(
                "🔥 Mythic Adopt Available" if ctx.bot.role_manager.
                can_use_adopt_mythic(user_role) else "❌ No Mythic Access")

        embed.add_field(name="🎯 Special Privileges",
                        value="\n".join(privileges),
                        inline=True)

        embed.set_footer(
            text=f"Admin command used by {ctx.author.display_name}")
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to retrieve user beast data: {str(e)}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='userstats')
async def user_stats(ctx, user: Optional[discord.Member] = None):
    """Check public stats for a user"""
    if user is None:
        user = ctx.author

    # ✅ FIXED: Add assertion to ensure user is not None
    if user is None:
        await ctx.bot.safe_send_message(ctx.channel,
                                        "❌ Unable to determine user.")
        return

    try:
        # Now pyright knows user is definitely not None
        user_data = await ctx.bot.get_or_create_user(user.id,
                                                     str(user))  # ✅ Safe
        user_beasts = await ctx.bot.db.get_user_beasts(user.id)  # ✅ Safe
        user_role = ctx.bot.role_manager.get_user_role(user)
        beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

        embed = discord.Embed(
            title=f"📊 {user.display_name}'s Public Stats",  # ✅ Safe
            color=0x00AAFF)

        # Beast collection stats
        embed.add_field(
            name="🐉 Beast Collection",
            value=f"**Total Beasts:** {len(user_beasts)}/{beast_limit}\n"
            f"**Total Catches:** {user_data.total_catches}",
            inline=True)

        # Battle stats
        embed.add_field(
            name="⚔️ Battle Record",
            value=f"**Total Battles:** {user_data.total_battles}\n"
            f"**Wins:** {user_data.wins} | **Losses:** {user_data.losses}\n"
            f"**Win Rate:** {user_data.win_rate:.1f}%",
            inline=True)

        # Economy stats
        embed.add_field(name="💰 Economy",
                        value=f"**Beast Stones:** {user_data.spirit_stones:,}",
                        inline=True)

        # Rarity breakdown
        if user_beasts:
            rarity_counts = {}
            total_power = 0
            highest_level = 0

            for _, beast in user_beasts:
                rarity_name = beast.rarity.name.title()
                rarity_counts[rarity_name] = rarity_counts.get(rarity_name,
                                                               0) + 1
                total_power += beast.power_level
                highest_level = max(highest_level, beast.stats.level)

            rarity_text = []
            for rarity in [
                    'Common', 'Uncommon', 'Rare', 'Epic', 'Legendary', 'Mythic'
            ]:
                count = rarity_counts.get(rarity, 0)
                if count > 0:
                    stars = '⭐' * ([
                        'Common', 'Uncommon', 'Rare', 'Epic', 'Legendary',
                        'Mythic'
                    ].index(rarity) + 1)
                    rarity_text.append(f"{stars} {count}")

            embed.add_field(
                name="🌟 Collection Breakdown",
                value="\n".join(rarity_text) if rarity_text else "No beasts",
                inline=True)

            embed.add_field(
                name="📈 Power Stats",
                value=f"**Total Power:** {total_power:,}\n"
                f"**Highest Level:** {highest_level}\n"
                f"**Average Power:** {total_power // len(user_beasts) if user_beasts else 0:,}",
                inline=True)

        # Account age
        account_age = datetime.datetime.now() - user_data.created_at
        embed.add_field(
            name="📅 Account Info",
            value=
            f"**Playing Since:** {user_data.created_at.strftime('%Y-%m-%d')}\n"
            f"**Days Active:** {account_age.days}",
            inline=True)

        # Active beast
        if user_data.active_beast_id:
            for beast_id, beast in user_beasts:
                if beast_id == user_data.active_beast_id:
                    embed.add_field(
                        name="🟢 Active Beast",
                        value=f"#{beast_id} {beast.name} {beast.rarity.emoji}\n"
                        f"Level {beast.stats.level}",
                        inline=False)
                    break

        embed.set_footer(text=f"Requested by {ctx.author.display_name}")
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to retrieve user stats: {str(e)}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='leaderboard', aliases=['lb', 'top'])
async def leaderboard(ctx,
                      category: str = "beasts",
                      limit: int = 10):  # ✅ Added missing 'self'
    """Show various leaderboards

    Categories: beasts, stones, battles, wins, catches, power
    """
    if limit > 20:
        limit = 20
    elif limit < 3:
        limit = 3

    valid_categories = [
        "beasts", "stones", "battles", "wins", "catches", "power"
    ]
    if category.lower() not in valid_categories:
        embed = discord.Embed(
            title="❌ Invalid Category",
            description=f"Valid categories: {', '.join(valid_categories)}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # ✅ FIXED: Initialize variables with defaults
    title = "📊 Leaderboard"
    value_key = "beast_count"
    value_suffix = " items"

    try:
        # Get all users from database
        conn = ctx.bot.db._get_connection()
        cursor = conn.execute('SELECT * FROM users ORDER BY created_at')
        all_users = cursor.fetchall()
        conn.close()

        if not all_users:
            embed = discord.Embed(
                title="📊 Empty Leaderboard",
                description="No users found in the database.",
                color=0x808080)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Calculate leaderboard data
        leaderboard_data = []

        for user_row in all_users:
            try:
                # Get Discord user object
                discord_user = ctx.bot.get_user(user_row['user_id'])
                if not discord_user:
                    try:
                        discord_user = await ctx.bot.fetch_user(
                            user_row['user_id'])
                    except:
                        continue

                user_data = User(user_id=user_row['user_id'],
                                 username=user_row['username'],
                                 spirit_stones=user_row['spirit_stones'],
                                 total_catches=user_row['total_catches'],
                                 total_battles=user_row['total_battles'],
                                 wins=user_row['wins'],
                                 losses=user_row['losses'],
                                 created_at=datetime.datetime.fromisoformat(
                                     user_row['created_at']))

                # Get beast count and total power for this user
                user_beasts = await ctx.bot.db.get_user_beasts(
                    user_data.user_id)
                beast_count = len(user_beasts)
                total_power = sum(beast.power_level
                                  for _, beast in user_beasts)

                entry = {
                    'user': discord_user,
                    'user_data': user_data,
                    'beast_count': beast_count,
                    'total_power': total_power
                }
                leaderboard_data.append(entry)
            except Exception as e:
                continue  # Skip problematic users

        if not leaderboard_data:
            embed = discord.Embed(
                title="📊 Empty Leaderboard",
                description="No valid users found for leaderboard.",
                color=0x808080)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Sort based on category
        category = category.lower()
        if category == "beasts":
            leaderboard_data.sort(key=lambda x: x['beast_count'], reverse=True)
            title = "🐉 Beast Collection Leaderboard"
            value_key = 'beast_count'
            value_suffix = " beasts"
        elif category == "stones":
            leaderboard_data.sort(key=lambda x: x['user_data'].spirit_stones,
                                  reverse=True)
            title = "💰 Beast Stones Leaderboard"
            value_key = 'spirit_stones'
            value_suffix = " stones"
        elif category == "battles":
            leaderboard_data.sort(key=lambda x: x['user_data'].total_battles,
                                  reverse=True)
            title = "⚔️ Total Battles Leaderboard"
            value_key = 'total_battles'
            value_suffix = " battles"
        elif category == "wins":
            leaderboard_data.sort(key=lambda x: x['user_data'].wins,
                                  reverse=True)
            title = "🏆 Battle Wins Leaderboard"
            value_key = 'wins'
            value_suffix = " wins"
        elif category == "catches":
            leaderboard_data.sort(key=lambda x: x['user_data'].total_catches,
                                  reverse=True)
            title = "🎯 Total Catches Leaderboard"
            value_key = 'total_catches'
            value_suffix = " catches"
        elif category == "power":
            leaderboard_data.sort(key=lambda x: x['total_power'], reverse=True)
            title = "💪 Total Power Leaderboard"
            value_key = 'total_power'
            value_suffix = " power"
        else:
            # ✅ ADDED: Safety else clause (should never be reached due to validation)
            embed = discord.Embed(
                title="❌ Unexpected Error",
                description="Invalid category detected after validation.",
                color=0xFF0000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Create leaderboard embed
        embed = discord.Embed(
            title=title,  # ✅ Now guaranteed to be defined
            description=f"Top {min(limit, len(leaderboard_data))} players",
            color=0xFFD700)

        # Add leaderboard entries
        medals = ["🥇", "🥈", "🥉"]

        for i, entry in enumerate(leaderboard_data[:limit]):
            rank = i + 1
            medal = medals[i] if i < 3 else f"{rank}."

            # ✅ FIXED: Initialize value with default
            value = 0

            if value_key == 'spirit_stones':
                value = entry['user_data'].spirit_stones
            elif value_key == 'total_battles':
                value = entry['user_data'].total_battles
            elif value_key == 'wins':
                value = entry['user_data'].wins
            elif value_key == 'total_catches':
                value = entry['user_data'].total_catches
            elif value_key == 'beast_count':
                value = entry['beast_count']
            elif value_key == 'total_power':
                value = entry['total_power']

            # Add win rate for battle-related leaderboards
            if category in ["battles", "wins"]:
                win_rate = entry['user_data'].win_rate
                embed.add_field(
                    name=f"{medal} {entry['user'].display_name}",
                    value=
                    f"**{value:,}**{value_suffix}\n*{win_rate:.1f}% win rate*",  # ✅ Now safe
                    inline=True)
            else:
                embed.add_field(
                    name=f"{medal} {entry['user'].display_name}",
                    value=f"**{value:,}**{value_suffix}",  # ✅ Now safe
                    inline=True)

        # Find requesting user's rank
        user_rank = None
        for i, entry in enumerate(leaderboard_data):
            if entry['user'].id == ctx.author.id:
                user_rank = i + 1
                break

        if user_rank and user_rank > limit:
            embed.set_footer(
                text=f"Your rank: #{user_rank}")  # ✅ Fixed f-string
        elif user_rank:
            embed.set_footer(text="You're on the leaderboard! 🎉")
        else:
            embed.set_footer(
                text="Start your journey to climb the leaderboard!")

        await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Leaderboard Error",
            description=f"Failed to generate leaderboard: {str(e)}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='serverbeasts', aliases=['serverstats'])
async def server_beast_stats(ctx):
    """Show server-wide beast statistics"""
    try:
        # Get all users and beasts
        conn = ctx.bot.db._get_connection()

        # User stats
        user_cursor = conn.execute('SELECT COUNT(*) as total_users FROM users')
        total_users = user_cursor.fetchone()['total_users']

        # Beast stats
        beast_cursor = conn.execute(
            'SELECT COUNT(*) as total_beasts FROM beasts')
        total_beasts = beast_cursor.fetchone()['total_beasts']

        # Battle stats
        battle_cursor = conn.execute(
            'SELECT SUM(total_battles) as total_battles, SUM(wins) as total_wins FROM users'
        )
        battle_stats = battle_cursor.fetchone()
        total_battles = battle_stats['total_battles'] or 0
        total_wins = battle_stats['total_wins'] or 0

        # Stone stats
        stone_cursor = conn.execute(
            'SELECT SUM(spirit_stones) as total_stones, SUM(total_catches) as total_catches FROM users'
        )
        stone_stats = stone_cursor.fetchone()
        total_stones = stone_stats['total_stones'] or 0
        total_catches = stone_stats['total_catches'] or 0

        conn.close()

        # Get detailed beast data for rarity breakdown
        all_beast_data = []
        conn = ctx.bot.db._get_connection()
        user_cursor = conn.execute(...)
        user_rows = conn.execute('SELECT user_id FROM users').fetchall()

        rarity_counts = {rarity.name: 0 for rarity in BeastRarity}
        total_power = 0
        highest_level = 0

        for user_row in user_rows:
            try:
                user_beasts = await ctx.bot.db.get_user_beasts(
                    user_row['user_id'])
                for _, beast in user_beasts:
                    rarity_counts[beast.rarity.name] += 1
                    total_power += beast.power_level
                    highest_level = max(highest_level, beast.stats.level)
            except:
                continue

        conn.close()

        # Create server stats embed
        embed = discord.Embed(
            title="🏰 Server Beast Statistics",
            description=f"Overview of {ctx.guild.name}'s beast community",
            color=0x00AAFF)

        # General stats
        embed.add_field(name="👥 Community Stats",
                        value=f"**Total Players:** {total_users:,}\n"
                        f"**Total Beasts:** {total_beasts:,}\n"
                        f"**Total Catches:** {total_catches:,}",
                        inline=True)

        # Battle stats
        embed.add_field(
            name="⚔️ Battle Stats",
            value=f"**Total Battles:** {total_battles:,}\n"
            f"**Total Wins:** {total_wins:,}\n"
            f"**Average per User:** {(total_battles/total_users):.1f}"
            if total_users > 0 else "**Average per User:** 0",
            inline=True)

        # Economy stats
        embed.add_field(
            name="💰 Economy Stats",
            value=f"**Total Beast Stones:** {total_stones:,}\n"
            f"**Average per User:** {(total_stones/total_users):,.0f}"
            if total_users > 0 else "**Average per User:** 0\n"
            f"**Stones per Beast:** {(total_stones/total_beasts):,.0f}"
            if total_beasts > 0 else "**Stones per Beast:** 0",
            inline=True)

        # Rarity breakdown
        if total_beasts > 0:
            rarity_text = []
            for rarity in BeastRarity:
                count = rarity_counts.get(rarity.name, 0)
                percentage = (count /
                              total_beasts) * 100 if total_beasts > 0 else 0
                if count > 0:
                    rarity_text.append(
                        f"{rarity.emoji} {count:,} ({percentage:.1f}%)")

            embed.add_field(name="🌟 Rarity Distribution",
                            value="\n".join(rarity_text)
                            if rarity_text else "No beasts found",
                            inline=True)

            # Power stats
            embed.add_field(
                name="💪 Power Stats",
                value=f"**Total Server Power:** {total_power:,}\n"
                f"**Highest Level:** {highest_level}\n"
                f"**Average Power:** {(total_power/total_beasts):,.0f}",
                inline=True)

        # Activity stats
        embed.add_field(
            name="📊 Activity",
            value=f"**Beasts per Player:** {(total_beasts/total_users):.1f}"
            if total_users > 0 else "**Beasts per Player:** 0\n"
            f"**Catches per Player:** {(total_catches/total_users):.1f}"
            if total_users > 0 else "**Catches per Player:** 0\n"
            f"**Most Active:** Use `{ctx.bot.config.prefix}leaderboard catches`",
            inline=True)

        embed.set_footer(
            text=
            f"Data as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Use !leaderboard for rankings"
        )
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Server Stats Error",
            description=f"Failed to generate server statistics: {str(e)}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='battle', aliases=['fight'])
@require_channel("battle")
async def battle_command(ctx, opponent: Optional[discord.Member] = None):
    """Challenge another user to a beast battle"""

    # Enhanced error: No opponent provided
    if opponent is None:
        embed = discord.Embed(color=0xFF6B6B)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# ❌ ⚔️ **IMMORTAL BEAST ARENA** ⚔️ ❌\n" +
                        "## 🚫 **INVALID BATTLE REQUEST** 🚫\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(
            name="📋 **Battle Instructions**",
            value=f"```yaml\n" +
            f"Command Format: {ctx.bot.config.prefix}battle @opponent\n" +
            f"Example Usage: {ctx.bot.config.prefix}battle @username\n" +
            f"Required: Valid opponent mention\n" + f"```",
            inline=False)
        embed.add_field(
            name="💡 **How to Start a Battle**",
            value="🎯 **Step 1:** Type `!battle` followed by @ mention\n" +
            "🎯 **Step 2:** Click on someone's name to mention them\n" +
            "🎯 **Step 3:** Press Enter to send the challenge!\n\n" +
            "**Example:** `!battle @FriendName`",
            inline=False)
        embed.set_footer(
            text="⚔️ IMMORTAL BEAST ARENA • Choose your opponent wisely!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Enhanced error: Opponent is a bot
    if opponent.bot:
        embed = discord.Embed(color=0xFF8C42)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 🤖 ⚔️ **IMMORTAL BEAST ARENA** ⚔️ 🤖\n" +
                        "## ⚠️ **INVALID OPPONENT TYPE** ⚠️\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🤖 **Bot Detection**",
                        value=f"```diff\n" +
                        f"- Target: {opponent.display_name}\n" +
                        f"- Type: Discord Bot\n" +
                        f"- Status: Cannot participate in battles\n" + f"```",
                        inline=False)
        embed.add_field(name="👥 **Valid Opponents**",
                        value="✅ **Human players** in this server\n" +
                        "✅ **Active members** with beast collections\n" +
                        "✅ **Users** who have adopted beasts\n\n" +
                        "❌ **Discord bots** (like me!)\n" +
                        "❌ **Webhook users**",
                        inline=False)
        embed.add_field(
            name="🎯 **Suggestion**",
            value=f"Try challenging a human player instead!\n" +
            f"Use `{ctx.bot.config.prefix}leaderboard` to see active players.",
            inline=False)
        embed.set_footer(
            text="⚔️ IMMORTAL BEAST ARENA • Only humans can command beasts!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Enhanced error: Trying to battle yourself
    if opponent.id == ctx.author.id:
        embed = discord.Embed(color=0xFFB347)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 🪞 ⚔️ **IMMORTAL BEAST ARENA** ⚔️ 🪞\n" +
                        "## 🤔 **SELF-CHALLENGE DETECTED** 🤔\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🪞 **Mirror Match Analysis**",
                        value="```yaml\n" + "Challenger: " +
                        ctx.author.display_name + "\n" + "Opponent:   " +
                        ctx.author.display_name + "\n" +
                        "Result:     Paradox Detected\n" + "```",
                        inline=False)
        embed.add_field(
            name="🧠 **Battle Philosophy**",
            value="🌟 **Training** happens through self-reflection\n" +
            "⚔️ **Combat** requires worthy opponents\n" +
            "🏆 **Glory** comes from defeating others\n" +
            "🤝 **Growth** comes from facing challenges",
            inline=False)
        embed.add_field(
            name="🎯 **Find an Opponent**",
            value=f"📊 `{ctx.bot.config.prefix}leaderboard` - See top players\n"
            + f"👥 `{ctx.bot.config.prefix}serverstats` - View active users\n" +
            f"🔍 Look around the server for other beast masters!",
            inline=False)
        embed.set_footer(
            text=
            "⚔️ IMMORTAL BEAST ARENA • Challenge others to prove your worth!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Get user beasts
    challenger_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    opponent_beasts = await ctx.bot.db.get_user_beasts(opponent.id)

    # Enhanced error: Challenger has no beasts
    if not challenger_beasts:
        embed = discord.Embed(color=0x845EC2)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 📦 ⚔️ **IMMORTAL BEAST ARENA** ⚔️ 📦\n" +
                        "## 🚫 **NO BEASTS AVAILABLE** 🚫\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(
            name="📊 **Your Beast Collection**",
            value="```diff\n" + f"- Trainer: {ctx.author.display_name}\n" +
            "- Beasts: 0/6\n" + "- Status: No battle-ready beasts\n" + "```",
            inline=False)
        embed.add_field(
            name="🎯 **How to Get Your First Beast**",
            value=
            f"🐾 `{ctx.bot.config.prefix}adopt` - Adopt a random beast (every 2 days)\n"
            +
            f"🌟 `{ctx.bot.config.prefix}catch` - Catch wild beasts when they spawn\n"
            + f"💰 `{ctx.bot.config.prefix}stone` - Get daily beast stones\n" +
            f"👁️ `{ctx.bot.config.prefix}nextspawn` - Check when beasts spawn",
            inline=False)
        embed.add_field(name="🎮 **Beast Master Journey**",
                        value="**Step 1:** 🏠 Adopt your first companion\n" +
                        "**Step 2:** 🎯 Catch wild beasts in spawn channel\n" +
                        "**Step 3:** 💪 Train and level up your beasts\n" +
                        "**Step 4:** ⚔️ Return here to battle!",
                        inline=False)
        embed.set_footer(
            text=
            "⚔️ IMMORTAL BEAST ARENA • Every legend starts with a single beast!"
        )
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Enhanced error: Opponent has no beasts
    if not opponent_beasts:
        embed = discord.Embed(color=0xFF6B9D)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 👤 ⚔️ **IMMORTAL BEAST ARENA** ⚔️ 👤\n" +
                        "## 😔 **OPPONENT NOT READY** 😔\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🔍 **Target Analysis**",
                        value=f"```yaml\n" +
                        f"Opponent: {opponent.display_name}\n" +
                        f"Beast Count: 0\n" + f"Battle Ready: No\n" +
                        f"Status: Needs to build collection\n" + f"```",
                        inline=False)
        embed.add_field(
            name="💬 **Message for " + opponent.display_name + "**",
            value=f"Hey {opponent.mention}! 👋\n\n" +
            f"🎯 **{ctx.author.display_name} wants to battle you!**\n" +
            f"🐾 Use `{ctx.bot.config.prefix}adopt` to get your first beast\n" +
            f"⚔️ Then you can accept battle challenges!",
            inline=False)
        embed.add_field(
            name="🔄 **Alternative Opponents**",
            value=
            f"📊 `{ctx.bot.config.prefix}leaderboard` - Find active beast masters\n"
            + f"👥 Look for users with beast collections\n" +
            f"🎯 Challenge someone who's ready to fight!",
            inline=False)
        embed.set_footer(
            text=
            "⚔️ IMMORTAL BEAST ARENA • Help others start their beast journey!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Let challenger select their beast
    challenger_beast = await select_beast_for_battle(ctx, ctx.author,
                                                     challenger_beasts, "your")
    if not challenger_beast:
        embed = discord.Embed(color=0xFFA07A)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# ⏰ ⚔️ **IMMORTAL BEAST ARENA** ⚔️ ⏰\n" +
                        "## 🚫 **BATTLE CANCELLED** 🚫\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="⏱️ **Selection Timeout**",
                        value=f"```diff\n" +
                        f"- Challenger: {ctx.author.display_name}\n" +
                        f"- Action: Beast selection\n" +
                        f"- Result: No response (30s timeout)\n" +
                        f"- Status: Battle cancelled\n" + f"```",
                        inline=False)
        embed.add_field(
            name="💡 **What Happened?**",
            value="🎯 You were asked to select a beast for battle\n" +
            "⏰ The selection window timed out after 30 seconds\n" +
            "🚫 The battle was automatically cancelled\n" +
            "🔄 You can try challenging again anytime!",
            inline=False)
        embed.add_field(name="🎮 **Battle Tips**",
                        value="⚡ **Respond quickly** when selecting beasts\n" +
                        "📱 **Stay active** during the battle setup\n" +
                        "🎯 **Choose wisely** - pick your strongest beast!\n" +
                        "❤️ **Check HP** - heal damaged beasts first",
                        inline=False)
        embed.set_footer(
            text="⚔️ IMMORTAL BEAST ARENA • Quick decisions lead to victory!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Let opponent select their beast
    opponent_beast = await select_beast_for_battle(
        ctx, opponent, opponent_beasts, f"{opponent.display_name}'s")
    if not opponent_beast:
        embed = discord.Embed(color=0xDDA0DD)
        embed.add_field(name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        value="# 👤 ⚔️ **IMMORTAL BEAST ARENA** ⚔️ 👤\n" +
                        "## 😞 **OPPONENT WITHDREW** 😞\n" +
                        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                        inline=False)
        embed.add_field(name="🏃 **Battle Withdrawal**",
                        value=f"```yaml\n" +
                        f"Challenger: {ctx.author.display_name}\n" +
                        f"Opponent: {opponent.display_name}\n" +
                        f"Status: Opponent didn't select a beast\n" +
                        f"Result: Battle cancelled\n" + f"```",
                        inline=False)
        embed.add_field(
            name="🤔 **What Happened?**",
            value=f"⏰ **{opponent.display_name}** was asked to select a beast\n"
            + "📱 They didn't respond within 30 seconds\n" +
            "🚫 The battle was automatically cancelled\n" +
            "💭 They might be busy or away from keyboard",
            inline=False)
        embed.add_field(
            name="🔄 **Next Steps**",
            value=f"💬 **Message them** to arrange a better time\n" +
            f"🎯 **Try again later** when they're more active\n" +
            f"👥 **Find another opponent** who's ready to battle\n" +
            f"📊 Check `{ctx.bot.config.prefix}leaderboard` for active players",
            inline=False)
        embed.set_footer(
            text="⚔️ IMMORTAL BEAST ARENA • Patience leads to epic battles!")
        embed.timestamp = discord.utils.utcnow()
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    if challenger_beast[1].stats.hp < challenger_beast[1].stats.max_hp * 0.1:
        embed = discord.Embed(
            title="⚠️ Beast Too Injured",
            description=
            f"{challenger_beast[1].name} is too injured to battle! Heal them first.",
            color=0xFF8800)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check if opponent's beast is too injured to battle
    if opponent_beast[1].stats.hp < opponent_beast[1].stats.max_hp * 0.1:
        embed = discord.Embed(
            title="⚠️ Opponent's Beast Too Injured",
            description=
            f"{opponent_beast[1].name} is too injured to battle! They need to heal first.",
            color=0xFF8800)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Simulate the battle
    battle_result = await ctx.bot.battle_engine.simulate_battle(
        challenger_beast[1], opponent_beast[1])

    # Determine winners and update stats
    challenger_user = await ctx.bot.get_or_create_user(ctx.author.id,
                                                       str(ctx.author))
    opponent_user = await ctx.bot.get_or_create_user(opponent.id,
                                                     str(opponent))

    challenger_user.total_battles += 1
    opponent_user.total_battles += 1

    if battle_result['winner'] == challenger_beast[1].name:
        challenger_user.wins += 1
        opponent_user.losses += 1
        winner_user = ctx.author
        loser_user = opponent
    elif battle_result['winner'] == opponent_beast[1].name:
        opponent_user.wins += 1
        challenger_user.losses += 1
        winner_user = opponent
        loser_user = ctx.author
    else:
        winner_user = None
        loser_user = None

    first_win_today = False
    if winner_user:  # Only if there's a winner (not a draw)
        winner_is_challenger = (winner_user == ctx.author)
        winner_user_obj = challenger_user if winner_is_challenger else opponent_user

        # Check for first win of day
        now = datetime.datetime.now()

        if (not winner_user_obj.last_first_win
                or winner_user_obj.last_first_win.date() < now.date()):
            # First win today - activate XP boost
            winner_user_obj.last_first_win = now
            winner_user_obj.xp_boost_expires = now + datetime.timedelta(
                hours=2)
            first_win_today = True

            # Update the winner user object in the database immediately
            if winner_is_challenger:
                challenger_user = winner_user_obj
            else:
                opponent_user = winner_user_obj

    if winner_user:  # If there's a winner (not a draw)
        # Identify winner and loser beasts
        if winner_user == ctx.author:
            winner_beast_obj = challenger_beast[1]
            loser_beast_obj = opponent_beast[1]
            winner_beast_id = challenger_beast
            loser_beast_id = opponent_beast
        else:
            winner_beast_obj = opponent_beast[1]
            loser_beast_obj = challenger_beast[1]
            winner_beast_id = opponent_beast
            loser_beast_id = challenger_beast

        # Reset daily counters if day changed
        _check_and_reset_battle_day(challenger_user)
        _check_and_reset_battle_day(opponent_user)

        # Compute XP as usual
        winner_xp = calculate_enhanced_battle_xp(winner_beast_obj,
                                                 loser_beast_obj,
                                                 battle_result['turns'],
                                                 is_winner=True)
        loser_xp = calculate_enhanced_battle_xp(loser_beast_obj,
                                                winner_beast_obj,
                                                battle_result['turns'],
                                                is_winner=False)

        # Enforce hard cap = 10
        chall_cap_reached = challenger_user.battle_xp_count >= 10
        opp_cap_reached = opponent_user.battle_xp_count >= 10

        # Zero out based on who is winner/loser
        if winner_user == ctx.author and chall_cap_reached:
            winner_xp = 0
        elif winner_user == opponent and opp_cap_reached:
            winner_xp = 0

        if winner_user == ctx.author:
            # opponent is loser
            if opp_cap_reached:
                loser_xp = 0
        else:
            # challenger is loser
            if chall_cap_reached:
                loser_xp = 0

        # Apply XP and consume daily slot only if XP>0
        winner_levelups = []
        loser_levelups = []

        if winner_xp > 0:
            winner_levelups = winner_beast_obj.stats.add_exp(
                winner_xp, winner_beast_obj.rarity)
            if winner_user == ctx.author:
                challenger_user.battle_xp_count += 1
            else:
                opponent_user.battle_xp_count += 1

        if loser_xp > 0:
            loser_levelups = loser_beast_obj.stats.add_exp(
                loser_xp, loser_beast_obj.rarity)
            if winner_user == ctx.author:
                opponent_user.battle_xp_count += 1
            else:
                challenger_user.battle_xp_count += 1

        # Update database with new XP and levels
        await ctx.bot.db.update_beast(winner_beast_id[0], winner_beast_obj
                                      )  # ✅ Access tuple correctly
        await ctx.bot.db.update_beast(loser_beast_id[0], loser_beast_obj)

        # Store XP info for embed display
        xp_info = {
            'winner_xp': winner_xp,
            'loser_xp': loser_xp,
            'winner_levelups': winner_levelups,
            'loser_levelups': loser_levelups,
            'winner_beast': winner_beast_obj,
            'loser_beast': loser_beast_obj,
            'winner_user': winner_user,
            'loser_user': loser_user
        }
    else:
        # Handle draw case - both get participation XP
        draw_xp = 30

        _check_and_reset_battle_day(challenger_user)
        _check_and_reset_battle_day(opponent_user)

        challenger_award = draw_xp if challenger_user.battle_xp_count < 10 else 0
        opponent_award = draw_xp if opponent_user.battle_xp_count < 10 else 0

        challenger_levelups = []
        opponent_levelups = []

        if challenger_award > 0:
            challenger_levelups = challenger_beast[1].stats.add_exp(
                challenger_award, challenger_beast[1].rarity)
            challenger_user.battle_xp_count += 1

        if opponent_award > 0:
            opponent_levelups = opponent_beast[1].stats.add_exp(
                opponent_award, opponent_beast[1].rarity)
            opponent_user.battle_xp_count += 1

        await ctx.bot.db.update_beast(challenger_beast, challenger_beast[1])
        await ctx.bot.db.update_beast(opponent_beast, opponent_beast[1])

        xp_info = {
            'draw': True,
            'challenger_xp': challenger_award,
            'opponent_xp': opponent_award,
            'challenger_levelups': challenger_levelups,
            'opponent_levelups': opponent_levelups
        }

    await ctx.bot.db.update_user(challenger_user)
    await ctx.bot.db.update_user(opponent_user)

    # ADD THE HP PERSISTENCE FIX HERE - after user stats, before embed
    # Apply battle damage to original beasts
    challenger_beast_obj = challenger_beast[1]
    opponent_beast_obj = opponent_beast[1]

    # Update HP based on battle results
    challenger_final_hp = battle_result['final_hp'][challenger_beast_obj.name]
    opponent_final_hp = battle_result['final_hp'][opponent_beast_obj.name]

    challenger_beast_obj.stats.hp = challenger_final_hp
    opponent_beast_obj.stats.hp = opponent_final_hp

    # Save the damaged beasts to database
    await ctx.bot.db.update_beast(challenger_beast[0], challenger_beast_obj)
    await ctx.bot.db.update_beast(opponent_beast[0], opponent_beast_obj)

    if battle_result['result'] == BattleResult.WIN:
        if winner_user == ctx.author:
            color = 0x00FF7F  # Bright victory green
            title_icon = "🏆"
            result_text = "**VICTORY**"
            victory_gradient = "🟢🟡🟠🔴"
        else:
            color = 0xFF4500  # Dramatic defeat orange-red
            title_icon = "💀"
            result_text = "**DEFEAT**"
            victory_gradient = "🔴🟠🟡🟢"
    elif battle_result['result'] == BattleResult.LOSS:
        if winner_user == ctx.author:
            color = 0x00FF7F
            title_icon = "🏆"
            result_text = "**VICTORY**"
            victory_gradient = "🟢🟡🟠🔴"
        else:
            color = 0xFF4500
            title_icon = "💀"
            result_text = "**DEFEAT**"
            victory_gradient = "🔴🟠🟡🟢"
    else:
        color = 0xFFD700  # Golden draw
        title_icon = "⚖️"
        result_text = "**DRAW**"
        victory_gradient = "🟡🟠🟡🟠"

    # Create the main embed with cinematic styling
    embed = discord.Embed(color=color)

    # Epic header design
    embed.add_field(
        name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        value=f"# {title_icon} ⚔️ **IMMORTAL BEAST ARENA** ⚔️ {title_icon}\n" +
        f"## {victory_gradient} {result_text} {victory_gradient[::-1]}\n" +
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        inline=False)

    # Fighter showcase with detailed stats
    embed.add_field(
        name="🥊 **COMBATANTS**",
        value=f"### 🔵 {ctx.author.display_name}'s Champion\n" + f"```ansi\n" +
        f"\u001b[1;36m{challenger_beast_obj.name}\u001b[0m {challenger_beast_obj.rarity.emoji}\n"
        +
        f"Level: {challenger_beast_obj.stats.level} | Power: {challenger_beast_obj.power_level:,}\n"
        +
        f"ATK: {challenger_beast_obj.stats.attack} | DEF: {challenger_beast_obj.stats.defense} | SPD: {challenger_beast_obj.stats.speed}\n"
        + f"```\n" + f"### 🔴 {opponent.display_name}'s Champion\n" +
        f"```ansi\n" +
        f"\u001b[1;31m{opponent_beast_obj.name}\u001b[0m {opponent_beast_obj.rarity.emoji}\n"
        +
        f"Level: {opponent_beast_obj.stats.level} | Power: {opponent_beast_obj.power_level:,}\n"
        +
        f"ATK: {opponent_beast_obj.stats.attack} | DEF: {opponent_beast_obj.stats.defense} | SPD: {opponent_beast_obj.stats.speed}\n"
        + f"```",
        inline=False)

    # Battle analytics with enhanced visuals
    battle_intensity = "🔥🔥🔥" if battle_result[
        'turns'] > 15 else "🔥🔥" if battle_result['turns'] > 8 else "🔥"
    battle_type = "LEGENDARY EPIC" if battle_result[
        'turns'] > 20 else "EPIC CLASH" if battle_result[
            'turns'] > 12 else "QUICK STRIKE" if battle_result[
                'turns'] <= 5 else "STANDARD DUEL"

    embed.add_field(name="📊 **BATTLE ANALYTICS**",
                    value=f"```yaml\n" + f"Battle Type: {battle_type}\n" +
                    f"Intensity:  {battle_intensity}\n" +
                    f"Duration:   {battle_result['turns']} rounds\n" +
                    f"Combat ID:  #{challenger_user.total_battles:04d}\n" +
                    f"```",
                    inline=True)

    # Championship results
    if winner_user:
        winner_beast = battle_result['winner']
        winner_hp = battle_result['final_hp'][winner_beast]

        embed.add_field(
            name="👑 **CHAMPION**",
            value=f"```diff\n" + f"+ {winner_user.display_name}\n" +
            f"+ {winner_beast}\n" + f"+ {winner_hp} HP Remaining\n" +
            f"```\n" + f"🏆 **Victory Achieved!**",
            inline=True)
    else:
        embed.add_field(name="⚖️ **STALEMATE**",
                        value=f"```css\n" + f"[Both Warriors Stand]\n" +
                        f"[Honor Preserved]\n" + f"[Rematch Awaited]\n" +
                        f"```\n" + f"🤝 **Honorable Draw**",
                        inline=True)

    # Advanced health visualization
    def create_premium_health_bar(current_hp, max_hp, length=8):
        if max_hp == 0:
            return "💀" * length

        percentage = current_hp / max_hp
        filled = int(percentage * length)

        if percentage > 0.7:
            bar_char = "🟢"
        elif percentage > 0.4:
            bar_char = "🟡"
        elif percentage > 0.1:
            bar_char = "🟠"
        else:
            bar_char = "🔴"

        if current_hp <= 0:
            return "💀" * length

        empty_char = "⬛"
        return bar_char * filled + empty_char * (length - filled)

    challenger_final_hp = battle_result['final_hp'][challenger_beast_obj.name]
    opponent_final_hp = battle_result['final_hp'][opponent_beast_obj.name]

    challenger_health_bar = create_premium_health_bar(
        challenger_final_hp, challenger_beast_obj.stats.max_hp)
    opponent_health_bar = create_premium_health_bar(
        opponent_final_hp, opponent_beast_obj.stats.max_hp)

    challenger_hp_percent = int(
        (challenger_final_hp / challenger_beast_obj.stats.max_hp) *
        100) if challenger_beast_obj.stats.max_hp > 0 else 0
    opponent_hp_percent = int(
        (opponent_final_hp / opponent_beast_obj.stats.max_hp) *
        100) if opponent_beast_obj.stats.max_hp > 0 else 0

    embed.add_field(
        name="❤️ **POST-BATTLE STATUS**",
        value=f"### {challenger_beast_obj.name}\n" +
        f"{challenger_health_bar} `{challenger_hp_percent}%`\n" +
        f"`{challenger_final_hp:,}/{challenger_beast_obj.stats.max_hp:,} HP`\n\n"
        + f"### {opponent_beast_obj.name}\n" +
        f"{opponent_health_bar} `{opponent_hp_percent}%`\n" +
        f"`{opponent_final_hp:,}/{opponent_beast_obj.stats.max_hp:,} HP`",
        inline=False)
    # XP Rewards Display Section
    if 'xp_info' in locals() and not xp_info.get('draw', False):
        # Winner/Loser XP display
        embed.add_field(
            name="⚡ **EXPERIENCE REWARDS**",
            value=
            (f"### 🏆 {xp_info['winner_user'].display_name}'s {xp_info['winner_beast'].name}\n"
             + f"```\n" + f"Victory XP: {xp_info['winner_xp']}\n" +
             f"New Level: {xp_info['winner_beast'].stats.level}\n" + f"```\n" +
             f"### 📚 {xp_info['loser_user'].display_name}'s {xp_info['loser_beast'].name}\n"
             + f"```\n" + f"Participation XP: {xp_info['loser_xp']}\n" +
             f"New Level: {xp_info['loser_beast'].stats.level}\n" + f"```"),
            inline=False)

        # Level up notifications
        level_up_messages = []
        if xp_info['winner_levelups']:
            level_count = len(xp_info['winner_levelups'])
            level_up_messages.append(
                f"🎆 **{xp_info['winner_beast'].name}** gained {level_count} level(s)!"
            )
        if xp_info['loser_levelups']:
            level_count = len(xp_info['loser_levelups'])
            level_up_messages.append(
                f"📈 **{xp_info['loser_beast'].name}** gained {level_count} level(s)!"
            )
        if level_up_messages:
            embed.add_field(name="🎉 **LEVEL UP ACHIEVEMENTS**",
                            value="\n".join(level_up_messages),
                            inline=False)

    elif 'xp_info' in locals() and xp_info.get('draw', False):
        # Draw XP display
        embed.add_field(
            name="⚡ **MUTUAL EXPERIENCE**",
            value=(
                f"```\n" +
                f"{ctx.author.display_name}: +{xp_info['challenger_xp']} XP\n"
                + f"{opponent.display_name}: +{xp_info['opponent_xp']} XP\n" +
                f"```\n" +
                f"🤝 **Both warriors grow stronger from this honorable clash!**"
            ),
            inline=False)

        # Draw level ups
        draw_levelups = []
        if xp_info['challenger_levelups']:
            draw_levelups.append(
                f"📈 {challenger_beast[1].name} gained {len(xp_info['challenger_levelups'])} level(s)!"
            )
        if xp_info['opponent_levelups']:
            draw_levelups.append(
                f"📈 {opponent_beast[1].name} gained {len(xp_info['opponent_levelups'])} level(s)!"
            )
        if draw_levelups:
            embed.add_field(name="🎉 **LEVEL BREAKTHROUGHS**",
                            value="\n".join(draw_levelups),
                            inline=False)

    # Add first win boost notification if it was activated
    if 'first_win_today' in locals() and first_win_today:
        embed.add_field(name="🔥 FIRST WIN BONUS ACTIVATED!",
                        value="🎆 **50% XP boost granted for 2 hours!**\n"
                        "⚡ All XP gains are now boosted!\n"
                        "💫 Feed, Groom, Spar, and Chat XP increased!",
                        inline=False)

    # Dynamic rewards section
    if winner_user:
        embed.add_field(name="🎁 **SPOILS OF WAR**",
                        value="```diff\n" + "+ Victory Glory Earned\n" +
                        "+ Battle Experience +XP\n" +
                        "+ Win Streak Updated\n" + "```",
                        inline=True)

        embed.add_field(name="💔 **BATTLE SCARS**",
                        value="```diff\n" + "- Beast Requires Healing\n" +
                        "- Defeat Recorded\n" +
                        "- Comeback Training Needed\n" + "```",
                        inline=True)
    else:
        embed.add_field(name="🏛️ **HALL OF HONOR**",
                        value="```yaml\n" + "Status: Mutual Respect\n" +
                        "Result: Experience Gained\n" +
                        "Future: Rematch Pending\n" + "```",
                        inline=False)

    # Premium footer with arena branding
    embed.add_field(
        name="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        value=
        f"⚔️ **IMMORTAL BEAST ARENA** • Battle #{challenger_user.total_battles:04d} • {ctx.guild.name}\n"
        +
        f"💡 *Use `!heal` to restore your beast • `!beasts` to view collection*\n"
        + f"🏟️ *Next battle awaits in the arena...*",
        inline=False)

    # Set author and timestamp for premium feel
    embed.set_author(
        name=
        f"Battle Report: {ctx.author.display_name} vs {opponent.display_name}",
        icon_url=ctx.author.display_avatar.url if hasattr(
            ctx.author, 'display_avatar') else None)

    embed.timestamp = discord.utils.utcnow()

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='adopt_legend', aliases=['adoptlegend'])
@require_channel("adopt")
async def adopt_legend_beast(ctx):
    """Adopt a guaranteed legendary beast (one-time use for special role users)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    # Check if user has special role permission
    if not ctx.bot.role_manager.can_use_adopt_legend(user_role):
        embed = discord.Embed(
            title="❌ Insufficient Permissions",
            description="You need a special role to use this command!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check if already used
    if user.has_used_adopt_legend:
        embed = discord.Embed(
            title="❌ Already Used",
            description=
            "You have already used your one-time legendary adoption!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check inventory space
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=
            f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Get a guaranteed legendary beast
    template = ctx.bot.template_manager.get_random_template_by_rarity(
        BeastRarity.LEGENDARY)
    if not template:
        embed = discord.Embed(
            title="❌ No Legendary Beasts Available",
            description=
            "No legendary beasts are currently available in the template.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    beast = template.create_beast()
    beast.owner_id = ctx.author.id

    # Add beast to database
    beast_id = await ctx.bot.db.add_beast(beast)

    # Update user stats
    user.has_used_adopt_legend = True
    user.total_catches += 1
    await ctx.bot.db.update_user(user)

    # Create success embed
    embed = discord.Embed(
        title="🌟 Legendary Beast Adopted!",
        description=
        f"**{ctx.author.display_name}** adopted a guaranteed legendary beast!\n**{beast.name}** {beast.rarity.emoji}",
        color=beast.rarity.color)

    embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
    embed.add_field(name="Rarity", value="⭐⭐⭐⭐⭐ Legendary", inline=True)
    embed.add_field(name="Level", value=beast.stats.level, inline=True)
    embed.add_field(name="Power Level", value=beast.power_level, inline=True)
    embed.add_field(name="Tendency",
                    value=beast.tendency or "None",
                    inline=True)
    embed.add_field(name="Location",
                    value=beast.location or "Unknown",
                    inline=True)

    if beast.description:
        embed.add_field(name="Description",
                        value=beast.description,
                        inline=False)

    embed.add_field(name="✨ Special Privilege",
                    value="This was your one-time legendary adoption!",
                    inline=False)

    embed.set_footer(
        text=f"Beast Inventory: {len(user_beasts) + 1}/{beast_limit}")
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='adopt_mythic', aliases=['adoptmythic'])
@require_channel("adopt")
async def adopt_mythic_beast(ctx):
    """Adopt a guaranteed mythic beast (one-time use for personal role users)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    # Check if user has personal role permission
    if not ctx.bot.role_manager.can_use_adopt_mythic(user_role):
        embed = discord.Embed(
            title="❌ Insufficient Permissions",
            description="You need the personal role to use this command!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check if already used
    if user.has_used_adopt_mythic:
        embed = discord.Embed(
            title="❌ Already Used",
            description="You have already used your one-time mythic adoption!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Check inventory space
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=
            f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Get a guaranteed mythic beast
    template = ctx.bot.template_manager.get_random_template_by_rarity(
        BeastRarity.MYTHIC)
    if not template:
        embed = discord.Embed(
            title="❌ No Mythic Beasts Available",
            description=
            "No mythic beasts are currently available in the template.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    beast = template.create_beast()
    beast.owner_id = ctx.author.id

    # Add beast to database
    beast_id = await ctx.bot.db.add_beast(beast)

    # Update user stats
    user.has_used_adopt_mythic = True
    user.total_catches += 1
    await ctx.bot.db.update_user(user)

    # Create success embed with special styling for mythic
    embed = discord.Embed(
        title="🔥 MYTHIC BEAST ADOPTED! 🔥",
        description=
        f"**{ctx.author.display_name}** adopted an ULTRA-RARE mythic beast!\n**{beast.name}** {beast.rarity.emoji}",
        color=beast.rarity.color)

    embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
    embed.add_field(name="Rarity", value="⭐⭐⭐⭐⭐⭐ MYTHIC", inline=True)
    embed.add_field(name="Level", value=beast.stats.level, inline=True)
    embed.add_field(name="Power Level", value=beast.power_level, inline=True)
    embed.add_field(name="Tendency",
                    value=beast.tendency or "None",
                    inline=True)
    embed.add_field(name="Location",
                    value=beast.location or "Unknown",
                    inline=True)

    if beast.description:
        embed.add_field(name="Description",
                        value=beast.description,
                        inline=False)

    embed.add_field(
        name="🔥 ULTIMATE PRIVILEGE",
        value=
        "This was your one-time MYTHIC adoption - the rarest of all beasts!",
        inline=False)

    embed.set_footer(
        text=
        f"Beast Inventory: {len(user_beasts) + 1}/{beast_limit} | You own a MYTHIC beast!"
    )
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)

@commands.command(name='givexp')
@commands.has_permissions(administrator=True)
async def give_exp(ctx, target: str, amount: int = 0):
    """Give XP to users' active beasts (admin only):
    - `!givexp all <amount>` - Give XP to ALL users' active beasts
    - `!givexp @user <amount>` - Give XP to a specific user's active beast
    """
    if amount <= 0:
        embed = discord.Embed(
            title="❌ Invalid Amount", 
            description="Please specify a positive amount of XP to give.",
            color=0xFF0000)
        embed.add_field(name="Usage Examples",
                        value=f"`{ctx.bot.config.prefix}givexp all 2000`\n"
                              f"`{ctx.bot.config.prefix}givexp @username 1000`",
                        inline=False)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    if target.lower() == 'all':
        # Mass XP distribution to ALL users' active beasts
        embed = discord.Embed(
            title="⚠️ Mass XP Distribution",
            description=f"This will give **{amount:,} XP** to every user's **active beast**.\n"
                       "Only users with active beasts will receive XP.\n\n"
                       "React with ✅ to confirm or ❌ to cancel.",
            color=0xFF8800)

        message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        await ctx.bot.safe_add_reaction(message, "✅")
        await ctx.bot.safe_add_reaction(message, "❌")

        def check(reaction, user):
            return (user == ctx.author and str(reaction.emoji) in ["✅", "❌"] 
                    and reaction.message.id == message.id)

        try:
            reaction, user = await ctx.bot.wait_for('reaction_add', timeout=30.0, check=check)

            if str(reaction.emoji) == "✅":
                embed = discord.Embed(
                    title="🔄 Distributing XP...",
                    description=f"Giving {amount:,} XP to all active beasts. This may take a few minutes.",
                    color=0xFFAA00)
                await ctx.bot.safe_edit_message(message, embed=embed)

                # Fetch all registered users
                conn = ctx.bot.db._get_connection()
                user_cursor = conn.execute('SELECT user_id FROM users')
                all_users = user_cursor.fetchall()
                conn.close()

                users_updated = 0
                users_skipped_no_active = 0
                users_skipped_no_beasts = 0
                total_level_ups = 0

                for user_row in all_users:
                    try:
                        user_id = user_row['user_id']
                        user_data = await ctx.bot.get_or_create_user(user_id, str(user_id))

                        # Check if user has an active beast
                        if not user_data.active_beast_id:
                            users_skipped_no_active += 1
                            continue

                        # Get user's beasts
                        user_beasts = await ctx.bot.db.get_user_beasts(user_id)
                        if not user_beasts:
                            users_skipped_no_beasts += 1
                            continue

                        # Find the active beast
                        active_beast = None
                        active_beast_id = None
                        for beast_id, beast in user_beasts:
                            if beast_id == user_data.active_beast_id:
                                active_beast = beast
                                active_beast_id = beast_id
                                break

                        # Give XP to active beast only
                        if active_beast:
                            old_level = active_beast.stats.level
                            level_ups = active_beast.stats.add_exp(amount, active_beast.rarity)
                            await ctx.bot.db.update_beast(active_beast_id, active_beast)

                            users_updated += 1
                            if level_ups:
                                total_level_ups += len(level_ups)
                        else:
                            users_skipped_no_active += 1

                        # Progress update every 25 users
                        if users_updated % 25 == 0:
                            print(f"📊 Processed {users_updated} users with active beasts")

                    except Exception as e:
                        print(f"❌ Error processing user {user_row['user_id']}: {e}")
                        continue

                # Success message
                embed = discord.Embed(
                    title="✅ Mass XP Distribution Complete!",
                    description=f"Successfully distributed **{amount:,} XP** to active beasts!",
                    color=0x00FF00)

                embed.add_field(name="📊 Distribution Stats",
                                value=f"**Active Beasts Updated:** {users_updated:,}\n"
                                      f"**Total Level-Ups:** {total_level_ups:,}\n"
                                      f"**Total XP Distributed:** {amount * users_updated:,}",
                                inline=True)

                embed.add_field(name="⏭️ Users Skipped",
                                value=f"**No Active Beast:** {users_skipped_no_active:,}\n"
                                      f"**No Beasts:** {users_skipped_no_beasts:,}\n"
                                      f"**Total Processed:** {len(all_users):,}",
                                inline=True)

                embed.add_field(name="🎁 Gift Details",
                                value=f"**XP Per Active Beast:** {amount:,}\n"
                                      f"**Reason:** Admin Gift\n"
                                      f"**Target:** Active Beasts Only",
                                inline=False)

                embed.set_footer(text=f"Mass XP distribution by {ctx.author.display_name}")

            else:
                embed = discord.Embed(
                    title="❌ Mass Distribution Cancelled",
                    description="XP distribution has been cancelled.",
                    color=0x808080)

        except asyncio.TimeoutError:
            embed = discord.Embed(
                title="⏰ Distribution Timeout",
                description="Distribution confirmation timed out.",
                color=0x808080)

        await ctx.bot.safe_edit_message(message, embed=embed)

    else:
        # Single user XP gift (active beast only)
        if not ctx.message.mentions:
            embed = discord.Embed(
                title="❌ No User Mentioned",
                description="Please mention a user to give XP to, or use 'all' for everyone.",
                color=0xFF0000)
            embed.add_field(name="Usage Examples",
                            value=f"`{ctx.bot.config.prefix}givexp @username 1000`\n"
                                  f"`{ctx.bot.config.prefix}givexp all 2000`",
                            inline=False)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        target_member = ctx.message.mentions[0]
        user_id = target_member.id

        # Get user data
        user_data = await ctx.bot.get_or_create_user(user_id, str(target_member))

        # Check if user has an active beast
        if not user_data.active_beast_id:
            embed = discord.Embed(
                title="❌ No Active Beast",
                description=f"**{target_member.display_name}** doesn't have an active beast set.\n"
                           f"They need to use `{ctx.bot.config.prefix}active <beast_id>` first.",
                color=0xFF0000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Get user's beasts
        user_beasts = await ctx.bot.db.get_user_beasts(user_id)
        if not user_beasts:
            embed = discord.Embed(
                title="❌ No Beasts Found",
                description=f"**{target_member.display_name}** doesn't have any beasts.",
                color=0xFF0000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Find the active beast
        active_beast = None
        active_beast_id = None
        for beast_id, beast in user_beasts:
            if beast_id == user_data.active_beast_id:
                active_beast = beast
                active_beast_id = beast_id
                break

        if not active_beast:
            embed = discord.Embed(
                title="❌ Active Beast Not Found",
                description=f"**{target_member.display_name}**'s active beast couldn't be found.\n"
                           f"Their active beast ID might be invalid.",
                color=0xFF0000)
            await ctx.bot.safe_send_message(ctx.channel, embed=embed)
            return

        # Apply XP to active beast only
        old_level = active_beast.stats.level
        level_ups = active_beast.stats.add_exp(amount, active_beast.rarity)
        await ctx.bot.db.update_beast(active_beast_id, active_beast)

        # Success embed
        embed = discord.Embed(
            title="🎁 XP Gift Delivered!",
            description=f"**{target_member.display_name}**'s active beast received **{amount:,} XP**!",
            color=active_beast.rarity.color)

        embed.add_field(name="🐉 Active Beast",
                        value=f"**{active_beast.name}** {active_beast.rarity.emoji}\n"
                              f"Level {old_level} → {active_beast.stats.level}",
                        inline=True)

        embed.add_field(name="⚡ XP Details",
                        value=f"**XP Received:** {amount:,}\n"
                              f"**Current XP:** {active_beast.stats.exp:,}\n"
                              f"**Power Level:** {active_beast.power_level:,}",
                        inline=True)

        if level_ups:
            levels_gained = len(level_ups)
            embed.add_field(name="🎉 Level-Up Achievement!",
                            value=f"**{active_beast.name}** gained **{levels_gained} level(s)**!\n"
                                  "🌟 The XP gift triggered level-ups!",
                            inline=False)

            # Show stat gains from level-ups
            if level_ups:
                last_gains = level_ups[-1][2]  # Get stat gains from last level-up
                stat_text = []
                for stat, gain in last_gains.items():
                    if not stat.startswith('bonus_'):
                        stat_text.append(f"{stat.title()}: +{gain}")

                if stat_text:
                    embed.add_field(name="📈 Recent Stat Gains",
                                    value="\n".join(stat_text[:4]),  # Show first 4 stats
                                    inline=True)
        else:
            embed.add_field(name="📊 No Level-Up",
                            value=f"**{active_beast.name}** received XP but didn't level up yet.\n"
                                  f"Keep growing stronger! 💪",
                            inline=False)

        embed.set_footer(text=f"XP gift from {ctx.author.display_name} • Active beast only")

        await ctx.bot.safe_send_message(ctx.channel, embed=embed)



@commands.command(name='adoption_status', aliases=['adoptstatus'])
async def adoption_status(ctx):
    """Check your special adoption status"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)

    embed = discord.Embed(
        title=f"🎯 {ctx.author.display_name}'s Adoption Status",
        description="Check your special adoption privileges",
        color=0x00AAFF)

    # User role info
    role_names = {
        UserRole.NORMAL: "Normal User",
        UserRole.SPECIAL: "Special Role User",
        UserRole.PERSONAL: "Personal Role User"
    }
    embed.add_field(name="👤 Your Role",
                    value=role_names[user_role],
                    inline=True)
    embed.add_field(
        name="📦 Beast Limit",
        value=f"{ctx.bot.role_manager.get_beast_limit(user_role)} beasts",
        inline=True)

    # Regular adoption status
    if user.last_adopt:
        cooldown_hours = ctx.bot.config.adopt_cooldown_hours
        next_adopt = user.last_adopt + datetime.timedelta(hours=cooldown_hours)
        if datetime.datetime.now() < next_adopt:
            embed.add_field(name="⏰ Next Regular Adopt",
                            value=f"<t:{int(next_adopt.timestamp())}:R>",
                            inline=False)
        else:
            embed.add_field(name="✅ Regular Adopt",
                            value="Available now!",
                            inline=False)
    else:
        embed.add_field(name="✅ Regular Adopt",
                        value="Available now!",
                        inline=False)

    # Special adoptions
    legend_available = ctx.bot.role_manager.can_use_adopt_legend(user_role)
    mythic_available = ctx.bot.role_manager.can_use_adopt_mythic(user_role)

    # Legend adoption status
    if legend_available:
        legend_status = "❌ Already Used" if user.has_used_adopt_legend else "✅ Available"
        embed.add_field(name="🌟 Legendary Adoption",
                        value=legend_status,
                        inline=True)
    else:
        embed.add_field(name="🌟 Legendary Adoption",
                        value="🔒 Requires Special Role",
                        inline=True)

    # Mythic adoption status
    if mythic_available:
        mythic_status = "❌ Already Used" if user.has_used_adopt_mythic else "✅ Available"
        embed.add_field(name="🔥 Mythic Adoption",
                        value=mythic_status,
                        inline=True)
    else:
        embed.add_field(name="🔥 Mythic Adoption",
                        value="🔒 Requires Personal Role",
                        inline=True)

    # Command info
    available_commands = []
    if not user.last_adopt or datetime.datetime.now(
    ) >= user.last_adopt + datetime.timedelta(
            hours=ctx.bot.config.adopt_cooldown_hours):
        available_commands.append(
            f"`{ctx.bot.config.prefix}adopt` - Regular adoption")

    if legend_available and not user.has_used_adopt_legend:
        available_commands.append(
            f"`{ctx.bot.config.prefix}adopt_legend` - Guaranteed legendary")

    if mythic_available and not user.has_used_adopt_mythic:
        available_commands.append(
            f"`{ctx.bot.config.prefix}adopt_mythic` - Guaranteed mythic")

    if available_commands:
        embed.add_field(name="🎯 Available Commands",
                        value="\n".join(available_commands),
                        inline=False)
    else:
        embed.add_field(name="⏸️ No Commands Available",
                        value="All adoptions are on cooldown or used",
                        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# Replace the existing setchannel and removechannel commands with these:


@commands.command(name='migrate_xp')
@commands.has_permissions(administrator=True)
async def migrate_xp_curve(ctx):
    """Migrate all beasts to new XP curve (admin only, one-time use)"""
    embed = discord.Embed(
        title="⚠️ XP Curve Migration",
        description=
        "This will recalculate ALL beast levels using the new XP curve.\n"
        "**This is irreversible!**\n\n"
        "React with ✅ to confirm or ❌ to cancel.",
        color=0xFF8800)

    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)
    await ctx.bot.safe_add_reaction(message, "✅")
    await ctx.bot.safe_add_reaction(message, "❌")

    def check(reaction, user):
        return (user == ctx.author and str(reaction.emoji) in ["✅", "❌"]
                and reaction.message.id == message.id)

    try:
        reaction, user = await ctx.bot.wait_for('reaction_add',
                                                timeout=30.0,
                                                check=check)

        if str(reaction.emoji) == "✅":
            embed = discord.Embed(
                title="🔄 Migration in Progress...",
                description=
                "Migrating all beasts to new XP curve. This may take a few minutes.",
                color=0xFFAA00)
            await ctx.bot.safe_edit_message(message, embed=embed)

            success = await migrate_all_beasts_to_new_xp_curve(ctx.bot)

            if success:
                embed = discord.Embed(
                    title="✅ Migration Complete!",
                    description=
                    "All beasts have been successfully migrated to the new XP curve.\n"
                    "Players will notice their beasts may have different levels now.",
                    color=0x00FF00)
            else:
                embed = discord.Embed(
                    title="❌ Migration Failed",
                    description=
                    "Migration encountered errors. Check console logs for details.",
                    color=0xFF0000)
        else:
            embed = discord.Embed(
                title="❌ Migration Cancelled",
                description="XP curve migration has been cancelled.",
                color=0x808080)

    except asyncio.TimeoutError:
        embed = discord.Embed(
            title="⏰ Migration Timeout",
            description="Migration confirmation timed out. No changes made.",
            color=0x808080)

    await ctx.bot.safe_edit_message(message, embed=embed)

@commands.command(name='migrate_stats_enhanced')
@commands.has_permissions(administrator=True)
async def migrate_beast_stats_enhanced(ctx):
    """Fix beast stats while preserving sacrifice/spar boosts (admin only)"""
    embed = discord.Embed(
        title="⚠️ Enhanced Beast Stat Migration",
        description="This will recalculate ALL beast stats while **preserving**:\n"
                    "✅ Sacrifice stat boosts\n"
                    "✅ Spar training gains\n" 
                    "✅ All extra stat bonuses\n\n"
                    "**This fixes weak high-level beasts without losing boosts!**\n\n"
                    "React with ✅ to confirm or ❌ to cancel.",
        color=0xFF8800)

    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)
    await ctx.bot.safe_add_reaction(message, "✅")
    await ctx.bot.safe_add_reaction(message, "❌")

    def check(reaction, user):
        return (user == ctx.author and str(reaction.emoji) in ["✅", "❌"] 
                and reaction.message.id == message.id)

    try:
        reaction, user = await ctx.bot.wait_for('reaction_add', timeout=30.0, check=check)

        if str(reaction.emoji) == "✅":
            embed = discord.Embed(
                title="🔄 Enhanced Migration in Progress...",
                description="Recalculating stats while preserving all boosts...",
                color=0xFFAA00)
            await ctx.bot.safe_edit_message(message, embed=embed)

            success = await fix_all_beast_stats_preserve_boosts(ctx.bot)

            if success:
                embed = discord.Embed(
                    title="✅ Enhanced Migration Complete!",
                    description="All beast stats fixed **with boosts preserved**!\n"
                               "🎯 Sacrifice boosts: **Preserved**\n"
                               "💪 Spar training gains: **Preserved**\n"
                               "⚡ High-level stats: **Fixed**",
                    color=0x00FF00)
            else:
                embed = discord.Embed(
                    title="❌ Migration Failed",
                    description="Enhanced migration encountered errors. Check logs.",
                    color=0xFF0000)
        else:
            embed = discord.Embed(
                title="❌ Migration Cancelled", 
                description="Enhanced migration cancelled.",
                color=0x808080)

    except asyncio.TimeoutError:
        embed = discord.Embed(
            title="⏰ Migration Timeout",
            description="Migration confirmation timed out.",
            color=0x808080)

    await ctx.bot.safe_edit_message(message, embed=embed)

@commands.command(name='setchannel')
@commands.has_permissions(administrator=True)
async def set_spawn_channel(ctx):
    """Set current channel as THE spawn channel"""
    # Update the bot's spawn channel (this would need a way to persist)
    ctx.bot.spawn_channel_id = ctx.channel.id
    embed = discord.Embed(
        title="✅ Spawn Channel Set",
        description=f"{ctx.channel.mention} is now THE beast spawn channel!",
        color=0x00FF00)
    embed.add_field(
        name="⚠️ Note",
        value=
        "This change is temporary. Update your environment variables for permanent change.",
        inline=False)
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='antispam')
@commands.has_permissions(administrator=True)
async def configure_antispam(ctx,
                             setting: Optional[str] = None,
                             value: Optional[int] = None):
    """Configure anti-spam settings (admin only)"""

    valid_settings = {
        'minlength': 'min_message_length',
        'similarity': 'max_similar_messages',
        'frequency': 'message_frequency_limit',
        'cooldown': 'xp_cooldown_seconds'
    }

    if setting is None:
        # Show current settings
        embed = discord.Embed(title="🛡️ Anti-Spam Configuration",
                              description="Current anti-spam settings",
                              color=0x00AAFF)

        embed.add_field(
            name="📏 Message Length",
            value=f"Minimum: {ctx.bot.config.min_message_length} characters",
            inline=True)

        embed.add_field(
            name="🔄 Similarity Check",
            value=
            f"Max similar: {ctx.bot.config.max_similar_messages} messages",
            inline=True)

        embed.add_field(
            name="⚡ Frequency Limit",
            value=
            f"Max: {ctx.bot.config.message_frequency_limit} messages/2min",
            inline=True)

        embed.add_field(name="⏰ XP Cooldown",
                        value=f"{ctx.bot.config.xp_cooldown_seconds} seconds",
                        inline=True)

        embed.add_field(
            name="🔧 Usage",
            value=f"`{ctx.bot.config.prefix}antispam <setting> <value>`\n"
            f"Settings: {', '.join(valid_settings.keys())}",
            inline=False)

        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    if setting.lower() not in valid_settings:
        embed = discord.Embed(
            title="❌ Invalid Setting",
            description=f"Valid settings: {', '.join(valid_settings.keys())}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    if value is None:
        embed = discord.Embed(
            title="❌ Missing Value",
            description=f"Please provide a value for {setting}",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Update the setting
    config_attr = valid_settings[setting.lower()]
    old_value = getattr(ctx.bot.config, config_attr)
    setattr(ctx.bot.config, config_attr, value)

    embed = discord.Embed(
        title="✅ Anti-Spam Setting Updated",
        description=
        f"**{setting.title()}** changed from `{old_value}` to `{value}`",
        color=0x00FF00)

    embed.add_field(
        name="⚠️ Note",
        value=
        "This change is temporary. Update environment variables for permanent change.",
        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='removechannel')
@commands.has_permissions(administrator=True)
async def remove_spawn_channel(ctx):
    """Remove current channel as spawn channel"""
    if ctx.channel.id == ctx.bot.spawn_channel_id:
        ctx.bot.spawn_channel_id = 0  # Disable spawning
        embed = discord.Embed(
            title="✅ Spawn Channel Removed",
            description=
            f"{ctx.channel.mention} is no longer the spawn channel.",
            color=0x00FF00)
    else:
        embed = discord.Embed(
            title="❌ Not the Spawn Channel",
            description=
            f"{ctx.channel.mention} is not the current spawn channel.",
            color=0xFF0000)
    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='catch')
@require_channel("spawn")
async def catch_beast(ctx):
    """Catch a spawned beast (3 attempts per user per beast)"""
    if not ctx.bot.current_spawned_beast:
        embed = discord.Embed(
            title="❌ No Beast Here",
            description="There's no beast to catch in this channel!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    beast = ctx.bot.current_spawned_beast
    user_id = ctx.author.id

    # Check catch attempts
    current_attempts = ctx.bot.catch_attempts.get(user_id, 0)
    if current_attempts >= ctx.bot.max_catch_attempts:
        embed = discord.Embed(
            title="❌ No More Attempts",
            description=
            f"You've already used all {ctx.bot.max_catch_attempts} attempts to catch this beast!",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=
            f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\n"
            f"Use `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    # Increment attempt count
    ctx.bot.catch_attempts[user_id] = current_attempts + 1
    remaining_attempts = ctx.bot.max_catch_attempts - ctx.bot.catch_attempts[
        user_id]

    # Calculate catch rate
    catch_rate = beast.rarity.catch_rate
    success = random.randint(1, 100) <= catch_rate

    if success:
        beast.owner_id = ctx.author.id
        beast_id = await ctx.bot.db.add_beast(beast)

        user.total_catches += 1
        await ctx.bot.db.update_user(user)

        # Remove spawned beast
        ctx.bot.current_spawned_beast = None
        ctx.bot.catch_attempts.clear()

        embed = discord.Embed(
            title="🎉 Beast Caught!",
            description=
            f"**{ctx.author.display_name}** successfully caught **{beast.name}**!\n{beast.rarity.emoji}",
            color=beast.rarity.color)
        embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
        embed.add_field(name="Level", value=beast.stats.level, inline=True)
        embed.add_field(name="Power Level",
                        value=beast.power_level,
                        inline=True)
        embed.add_field(
            name="Attempt",
            value=f"{current_attempts + 1}/{ctx.bot.max_catch_attempts}",
            inline=True)
        embed.add_field(name="Catch Rate", value=f"{catch_rate}%", inline=True)
        embed.add_field(name="Total Beasts",
                        value=f"{len(user_beasts) + 1}/{beast_limit}",
                        inline=True)
    else:
        embed = discord.Embed(
            title="💥 Beast Escaped!",
            description=f"**{beast.name}** broke free and escaped!",
            color=0xFF4444)
        embed.add_field(
            name="Attempt",
            value=f"{current_attempts + 1}/{ctx.bot.max_catch_attempts}",
            inline=True)
        embed.add_field(name="Remaining Attempts",
                        value=remaining_attempts,
                        inline=True)
        embed.add_field(name="Catch Rate", value=f"{catch_rate}%", inline=True)

        if remaining_attempts > 0:
            embed.add_field(
                name="Keep Trying!",
                value=f"You have {remaining_attempts} attempts left!",
                inline=False)
        else:
            embed.add_field(
                name="No More Attempts",
                value="You've used all your attempts for this beast!",
                inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# Flask Web Server (to keep Render happy)

app = Flask(__name__)


@app.route('/')
def home():
    return "🐉 Immortal Beasts Discord Bot is running!"


@app.route('/health')
def health():
    return "OK"


@app.route('/status')
def status():
    return {
        "status": "running",
        "bot": "Immortal Beasts Bot",
        "message": "Discord bot is active"
    }


def run_flask():
    """Run Flask in a separate thread"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)


def run_bot_with_flask():
    """Run both Flask and Discord bot"""
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Run Discord bot in main thread
    main()


# NEW BACKUP ADMIN COMMANDS - ADD THESE BEFORE def main():


@commands.command(name='backupstatus')
@commands.has_permissions(administrator=True)
async def backup_status(ctx):
    """Check backup storage usage and status"""
    storage_info = ctx.bot.db.get_storage_usage()

    embed = discord.Embed(title="📊 Backup System Status", color=0x00AAFF)

    embed.add_field(name="📁 Total Backups",
                    value=storage_info['total_files'],
                    inline=True)
    embed.add_field(name="💾 Storage Used",
                    value=f"{storage_info['total_size_mb']:.1f} MB",
                    inline=True)
    embed.add_field(name="📊 Max Storage",
                    value=f"{ctx.bot.config.backup_max_size_mb} MB",
                    inline=True)

    embed.add_field(name="⚙️ Backup Enabled",
                    value="✅ Yes" if ctx.bot.config.backup_enabled else "❌ No",
                    inline=True)
    embed.add_field(name="⏰ Interval",
                    value=f"{ctx.bot.config.backup_interval_hours}h",
                    inline=True)
    embed.add_field(name="🗃️ Retention",
                    value=f"{ctx.bot.config.backup_retention_count} files",
                    inline=True)

    if storage_info['oldest_backup']:
        oldest_time = datetime.datetime.fromtimestamp(
            storage_info['oldest_backup'].stat().st_mtime)
        embed.add_field(name="📅 Oldest Backup",
                        value=oldest_time.strftime("%Y-%m-%d %H:%M"),
                        inline=True)

    if storage_info['newest_backup']:
        newest_time = datetime.datetime.fromtimestamp(
            storage_info['newest_backup'].stat().st_mtime)
        embed.add_field(name="📅 Newest Backup",
                        value=newest_time.strftime("%Y-%m-%d %H:%M"),
                        inline=True)

    # Storage warning
    usage_percent = (storage_info['total_size_mb'] /
                     ctx.bot.config.backup_max_size_mb) * 100
    if usage_percent > 80:
        embed.add_field(name="⚠️ Warning",
                        value=f"Storage usage at {usage_percent:.1f}%",
                        inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='backup')
@commands.has_permissions(administrator=True)
async def manual_backup(ctx):
    """Create a manual backup"""
    embed = discord.Embed(
        title="🔄 Creating Backup...",
        description="Please wait while the backup is created.",
        color=0xFFAA00)
    message = await ctx.bot.safe_send_message(ctx.channel, embed=embed)

    try:
        backup_file = await ctx.bot.db.backup_database(
            keep_count=ctx.bot.config.backup_retention_count)
        if backup_file:
            embed = discord.Embed(
                title="✅ Backup Created",
                description=f"Database backup saved successfully!",
                color=0x00FF00)
            embed.add_field(name="📁 File",
                            value=f"`{Path(backup_file).name}`",
                            inline=True)

            # Show updated storage info
            storage_info = ctx.bot.db.get_storage_usage()
            embed.add_field(name="💾 Total Storage",
                            value=f"{storage_info['total_size_mb']:.1f} MB",
                            inline=True)
            embed.add_field(name="📊 Total Backups",
                            value=storage_info['total_files'],
                            inline=True)
        else:
            embed = discord.Embed(
                title="❌ Backup Failed",
                description="Failed to create backup. Check logs for details.",
                color=0xFF0000)
    except Exception as e:
        embed = discord.Embed(title="❌ Backup Error",
                              description=f"An error occurred: {str(e)}",
                              color=0xFF0000)

    await ctx.bot.safe_edit_message(message, embed=embed)


# Add this command to help debug channel configuration:


@commands.command(name='channels')
@commands.has_permissions(administrator=True)
async def show_channel_config(ctx):
    """Show current channel configuration (admin only)"""
    embed = discord.Embed(title="📋 Channel Configuration",
                          description="Current bot channel settings",
                          color=0x00AAFF)

    # Battle Channels
    battle_channels = []
    for channel_id in ctx.bot.config.battle_channel_ids:
        channel = ctx.bot.get_channel(channel_id)
        if channel:
            battle_channels.append(f"#{channel.name} ({channel_id})")
        else:
            battle_channels.append(f"Unknown ({channel_id})")

    embed.add_field(name="⚔️ Battle Channels",
                    value="\n".join(battle_channels)
                    if battle_channels else "None configured",
                    inline=False)

    # Adopt Channel
    adopt_channel = ctx.bot.get_channel(ctx.bot.config.adopt_channel_id)
    embed.add_field(name="🐾 Adopt Channel",
                    value=f"#{adopt_channel.name}" if adopt_channel else
                    f"Unknown ({ctx.bot.config.adopt_channel_id})",
                    inline=True)

    # Spawn Channel
    spawn_channel = ctx.bot.get_channel(ctx.bot.spawn_channel_id)
    embed.add_field(name="🌟 Spawn Channel",
                    value=f"#{spawn_channel.name}" if spawn_channel else
                    f"Unknown ({ctx.bot.spawn_channel_id})",
                    inline=True)

    # XP Channels
    xp_channels = []
    for channel_id in ctx.bot.config.xp_chat_channel_ids:
        channel = ctx.bot.get_channel(channel_id)
        if channel:
            xp_channels.append(f"#{channel.name}")
        else:
            xp_channels.append(f"Unknown ({channel_id})")

    embed.add_field(name="💫 XP Channels",
                    value="\n".join(xp_channels[:5]) +
                    (f"\n... and {len(xp_channels)-5} more" if len(xp_channels)
                     > 5 else "") if xp_channels else "None configured",
                    inline=False)

    embed.add_field(name="📍 Current Channel",
                    value=f"#{ctx.channel.name} ({ctx.channel.id})",
                    inline=False)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


@commands.command(name='cleanbackups')
@commands.has_permissions(administrator=True)
async def clean_backups(ctx, keep_count: Optional[int] = None):
    """Clean old backups manually"""
    if keep_count is None:
        keep_count = ctx.bot.config.backup_retention_count

    # ✅ FIXED: Ensure keep_count is a valid integer
    if keep_count is None or keep_count < 1:
        embed = discord.Embed(title="❌ Invalid Count",
                              description="Keep count must be at least 1.",
                              color=0xFF0000)
        await ctx.bot.safe_send_message(ctx.channel, embed=embed)
        return

    try:
        storage_before = ctx.bot.db.get_storage_usage()
        await ctx.bot.db._cleanup_old_backups(Path("backups"), keep_count)
        storage_after = ctx.bot.db.get_storage_usage()

        files_removed = storage_before['total_files'] - storage_after[
            'total_files']
        space_freed = storage_before['total_size_mb'] - storage_after[
            'total_size_mb']

        embed = discord.Embed(title="🧹 Backup Cleanup Complete",
                              color=0x00FF00)
        embed.add_field(name="🗑️ Files Removed",
                        value=files_removed,
                        inline=True)
        embed.add_field(name="💾 Space Freed",
                        value=f"{space_freed:.1f} MB",
                        inline=True)
        embed.add_field(name="📁 Files Remaining",
                        value=storage_after['total_files'],
                        inline=True)

    except Exception as e:
        embed = discord.Embed(title="❌ Cleanup Failed",
                              description=f"Error during cleanup: {str(e)}",
                              color=0xFF0000)

    await ctx.bot.safe_send_message(ctx.channel, embed=embed)


# In your main() function, add these two lines with the other bot.add_command() calls:


def main():
    """Main entry point"""
    try:
        config = BotConfig.load_from_file()
    except FileNotFoundError as e:
        print(f"Configuration error: {e}")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    bot = ImmortalBeastsBot(config)

    # Add all commands
    bot.add_command(catch_beast)
    bot.add_command(daily_stone_reward)
    bot.add_command(adopt_beast)
    bot.add_command(show_beasts)
    bot.add_command(help_command)
    bot.add_command(beast_info)
    bot.add_command(set_active_beast)
    bot.add_command(show_balance)
    bot.add_command(battle_command)
    bot.add_command(adopt_legend_beast)
    bot.add_command(adopt_mythic_beast)
    bot.add_command(adoption_status)
    bot.add_command(force_spawn)
    bot.add_command(spawn_info)
    bot.add_command(next_spawn_time)
    bot.add_command(backup_status)
    bot.add_command(manual_backup)
    bot.add_command(clean_backups)
    bot.add_command(set_spawn_channel)
    bot.add_command(remove_spawn_channel)
    bot.add_command(show_channel_config)
    bot.add_command(sacrifice_beast)
    bot.add_command(release_beast)
    bot.add_command(fix_xp_cooldown)
    bot.add_command(check_cooldown)
    bot.add_command(ping_command)
    bot.add_command(heal_beast)
    bot.add_command(heal_all_beasts)
    bot.add_command(manual_cloud_backup)
    bot.add_command(restore_from_cloud)
    bot.add_command(check_user_beasts)
    bot.add_command(user_stats)
    bot.add_command(leaderboard)
    bot.add_command(server_beast_stats)
    bot.add_command(xp_config)
    bot.add_command(xp_status)
    bot.add_command(remove_xp_channel)
    bot.add_command(configure_antispam)
    bot.add_command(feed_beast)
    bot.add_command(groom_beast)
    bot.add_command(spar_training)
    bot.add_command(daily_status)
    bot.add_command(sacrifice_status)
    bot.add_command(migrate_xp_curve)
    bot.add_command(migrate_beast_stats_enhanced)
    bot.add_command(give_exp)

    # RETRY LOGIC INSIDE THE FUNCTION WHERE VARIABLES ARE AVAILABLE
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Starting bot (attempt {attempt + 1}/{max_retries})...")
            bot.run(config.token)
            break  # Success, exit retry loop
        except discord.LoginFailure:
            print("Invalid bot token. Please check your configuration.")
            break  # Don't retry for invalid token
        except Exception as e:
            print(f"Error running bot (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 60 * (attempt + 1)  # 60, 120, 180 seconds
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded. Bot failed to start.")


if __name__ == "__main__":
    if os.getenv('PORT'):  # Running on Render
        run_bot_with_flask()
    else:  # Running locally
        main()
