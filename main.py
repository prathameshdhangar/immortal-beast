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
from typing import Optional, List, Dict, Tuple, Union, Any
from contextlib import asynccontextmanager
import sqlite3

import discord
from discord.ext import commands, tasks
import aiofiles
import yaml
from pydantic import BaseModel, Field, field_validator
import threading
from flask import Flask
import gzip
import base64
import aiohttp

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
    spawn_interval_min: int = Field(
        default=15,
        ge=5,
        le=120,
        description="Minimum spawn interval in minutes")
    spawn_interval_max: int = Field(
        default=45,
        ge=5,
        le=120,
        description="Maximum spawn interval in minutes")
    log_level: str = Field(default="INFO", description="Logging level")

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

    # XP System - UPDATED FOR MULTIPLE CHANNELS
    xp_chat_channel_ids: List[int] = Field(
        default=[],
        description="List of channel IDs where beasts gain XP from chatting")
    xp_per_message: int = Field(default=5, description="XP gained per message")
    xp_cooldown_seconds: int = Field(
        default=60, description="Cooldown between XP gains from same user")

    # Starting resources
    starting_beast_stones: int = Field(
        default=1000, ge=0, description="Starting beast stones for new users")

    # Adopt cooldowns (in hours)
    adopt_cooldown_hours: int = Field(
        default=48,
        description="Adopt command cooldown in hours (2 days = 48)")

    @field_validator('spawn_interval_max')
    @classmethod
    def validate_spawn_intervals(cls, v, info):
        if info.data.get(
                'spawn_interval_min') and v <= info.data['spawn_interval_min']:
            raise ValueError(
                'spawn_interval_max must be greater than spawn_interval_min')
        return v

    @classmethod
    def load_from_file(cls, config_path: str = "config.yaml") -> 'BotConfig':
        """Load configuration from YAML file or environment variables"""

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
                battle_channel_ids=[1397783271961792604,1397783317163806730],
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
        if os.getenv('PORT'):  # Production
            backup_retention = 3  # Keep only 3 backups in production
            backup_interval = 12  # Backup twice daily
        else:  # Development
            backup_retention = 10  # Keep 10 backups locally
            backup_interval = 6  # Backup every 6 hours

        return cls(
            token=token,
            prefix=os.getenv('BOT_PREFIX', '!'),
            database_path='/tmp/immortal_beasts.db',  # Use /tmp for Render
            special_role_ids=special_role_ids,
            personal_role_id=int(
                os.getenv('PERSONAL_ROLE_ID', '1393176170601775175')),
            xp_chat_channel_ids=xp_channel_ids,
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            backup_interval_hours=int(
                os.getenv('BACKUP_INTERVAL_HOURS', str(backup_interval))),
            backup_retention_count=int(
                os.getenv('BACKUP_RETENTION_COUNT', str(backup_retention))),
            backup_max_size_mb=int(os.getenv('BACKUP_MAX_SIZE_MB', '100')),
            backup_enabled=os.getenv('BACKUP_ENABLED',
                                     'true').lower() == 'true',
            spawn_interval_min=int(os.getenv('SPAWN_INTERVAL_MIN', '15')),
            spawn_interval_max=int(os.getenv('SPAWN_INTERVAL_MAX', '45')),
            xp_per_message=int(os.getenv('XP_PER_MESSAGE', '5')),
            xp_cooldown_seconds=int(os.getenv('XP_COOLDOWN_SECONDS', '60')),
            starting_beast_stones=int(
                os.getenv('STARTING_BEAST_STONES', '1000')),
            adopt_cooldown_hours=int(os.getenv('ADOPT_COOLDOWN_HOURS', '48')),
            # ADD THESE MISSING LINES:
            battle_channel_ids=battle_channel_ids,
            adopt_channel_id=int(os.getenv('ADOPT_CHANNEL_ID', '1397783378618748948')),
            spawn_channel_id=int(os.getenv('SPAWN_CHANNEL_ID', '1397783188394475520')))


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
        """Get XP required for next level based on rarity"""
        base_requirements = {
            BeastRarity.COMMON: 100,
            BeastRarity.UNCOMMON: 150,
            BeastRarity.RARE: 200,
            BeastRarity.EPIC: 300,
            BeastRarity.LEGENDARY: 500,
            BeastRarity.MYTHIC: 800
        }
        base_req = base_requirements[rarity]
        return int(base_req * (self.level**1.2))

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

    def heal(self, amount: int = None) -> int:
        """Heal the beast, return amount healed"""
        if amount is None:
            amount = self.max_hp
        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - old_hp


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

    def create_beast(self) -> 'Beast':
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
class Beast:
    """Main beast class"""
    name: str
    rarity: BeastRarity
    tendency: str
    location: str
    stats: BeastStats
    description: str = ""
    caught_at: datetime.datetime = None
    owner_id: int = None
    unique_id: str = None

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
            'caught_at': self.caught_at.isoformat(),
            'owner_id': self.owner_id,
            'unique_id': self.unique_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Beast':
        """Create beast from dictionary"""
        stats = BeastStats(**data['stats'])
        return cls(name=data['name'],
                   rarity=BeastRarity(data['rarity']),
                   tendency=data['tendency'],
                   location=data['location'],
                   stats=stats,
                   description=data.get('description', ''),
                   caught_at=datetime.datetime.fromisoformat(
                       data['caught_at']),
                   owner_id=data.get('owner_id'),
                   unique_id=data.get('unique_id'))

    @property
    def power_level(self) -> int:
        """Calculate overall power level"""
        return (self.stats.hp + self.stats.attack + self.stats.defense +
                self.stats.speed) * self.stats.level


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
    created_at: datetime.datetime = None
    has_used_adopt_legend: bool = False
    has_used_adopt_mythic: bool = False

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
    async def get_user_beasts(self,
                              user_id: int,
                              limit: int = None) -> List[Tuple[int, Beast]]:
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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize database schema"""
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

                CREATE INDEX IF NOT EXISTS idx_beasts_owner ON beasts(owner_id);
                CREATE INDEX IF NOT EXISTS idx_battle_history_users ON battle_history(user1_id, user2_id);
                CREATE INDEX IF NOT EXISTS idx_users_active_beast ON users(active_beast_id);
            """)
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
            cursor = conn.execute('SELECT * FROM users WHERE user_id = ?',
                                  (user_id, ))
            row = cursor.fetchone()
            if row:
                return User(
                    user_id=row['user_id'],
                    username=row['username'],
                    spirit_stones=row['spirit_stones'],
                    last_daily=datetime.datetime.fromisoformat(
                        row['last_daily']) if row['last_daily'] else None,
                    last_adopt=datetime.datetime.fromisoformat(
                        row['last_adopt']) if row['last_adopt'] else None,
                    last_xp_gain=datetime.datetime.fromisoformat(
                        row['last_xp_gain']) if row['last_xp_gain'] else None,
                    active_beast_id=row['active_beast_id'],
                    total_catches=row['total_catches'],
                    total_battles=row['total_battles'],
                    wins=row['wins'],
                    losses=row['losses'],
                    has_used_adopt_legend=bool(row['has_used_adopt_legend']),
                    has_used_adopt_mythic=bool(row['has_used_adopt_mythic']),
                    created_at=datetime.datetime.fromisoformat(
                        row['created_at']))
            return None
        finally:
            conn.close()

    async def create_user(self, user: User) -> bool:
        """Create new user"""
        try:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO users 
                    (user_id, username, spirit_stones, total_catches, total_battles, wins, losses, 
                     has_used_adopt_legend, has_used_adopt_mythic, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (user.user_id, user.username, user.spirit_stones,
                      user.total_catches, user.total_battles, user.wins,
                      user.losses, user.has_used_adopt_legend,
                      user.has_used_adopt_mythic, user.created_at.isoformat()))
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
                conn.execute(
                    """
                    UPDATE users SET 
                    username=?, spirit_stones=?, last_daily=?, last_adopt=?, last_xp_gain=?,
                    active_beast_id=?, total_catches=?, total_battles=?, wins=?, losses=?,
                    has_used_adopt_legend=?, has_used_adopt_mythic=?
                    WHERE user_id=?
                """, (user.username, user.spirit_stones,
                      user.last_daily.isoformat() if user.last_daily else None,
                      user.last_adopt.isoformat() if user.last_adopt else None,
                      user.last_xp_gain.isoformat()
                      if user.last_xp_gain else None, user.active_beast_id,
                      user.total_catches, user.total_battles, user.wins,
                      user.losses, user.has_used_adopt_legend,
                      user.has_used_adopt_mythic, user.user_id))
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
            cursor = conn.execute(
                """
                INSERT INTO beasts (owner_id, beast_data, caught_at)
                VALUES (?, ?, ?)
            """, (beast.owner_id, json.dumps(
                    beast.to_dict()), beast.caught_at.isoformat()))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    async def get_user_beasts(self,
                              user_id: int,
                              limit: int = None) -> List[Tuple[int, Beast]]:
        """Get user's beasts"""
        query = "SELECT id, beast_data FROM beasts WHERE owner_id = ? ORDER BY caught_at DESC"
        params = [user_id]
        if limit:
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
            rarity_weights: Dict[BeastRarity, float] = None) -> BeastTemplate:
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

        if not available_templates:
            available_templates = [
                t for t in self.templates.values()
                if t.rarity.value <= max_rarity.value
            ]
            return random.choice(available_templates)

        return random.choices(available_templates, weights=weights)[0]

    def get_template_by_name(self, name: str) -> Optional[BeastTemplate]:
        """Get specific template by name"""
        return self.templates.get(name)


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
        beasts: List[Tuple[int, Beast]],
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

    message = await ctx.send(embed=embed)

    for i in range(len(options)):
        await message.add_reaction(number_emojis[i])

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


# Main Bot Class


class ImmortalBeastsBot(commands.Bot):
    """Main bot class"""

    def __init__(self, config: BotConfig):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.prefix,
                         intents=intents,
                         help_command=None)

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

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler('bot.log'),
                      logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    async def setup_hook(self):
        """Enhanced setup with automatic restoration"""
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

        await self.db.initialize()
        self.logger.info("Database initialized")

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

        self.spawn_task.start()
        self.logger.info("Background tasks started")

    async def on_ready(self):
        """Called when bot is ready"""
        self.logger.info(f'{self.user} has connected to Discord!')
        activity = discord.Game(
            name=f"Immortal Beasts | {self.config.prefix}help")
        await self.change_presence(activity=activity)

    async def on_message(self, message):
        """Handle messages for XP gain"""
        if message.author.bot:
            await self.process_commands(message)
            return

        # Check if message is in any XP channel (updated for multiple channels)
        if (message.channel.id in self.config.xp_chat_channel_ids
                and hasattr(message, 'content') and len(message.content) > 3):
            await self.handle_xp_gain(message)

        await self.process_commands(message)

    async def handle_xp_gain(self, message):
        """Handle XP gain from chatting"""
        try:
            user = await self.get_or_create_user(message.author.id,
                                                 str(message.author))

            if not user.can_gain_xp(self.config.xp_cooldown_seconds):
                return

            if not user.active_beast_id:
                return

            user_beasts = await self.db.get_user_beasts(user.user_id)
            active_beast = None
            active_beast_id = None

            for beast_id, beast in user_beasts:
                if beast_id == user.active_beast_id:
                    active_beast = beast
                    active_beast_id = beast_id
                    break

            if not active_beast:
                return

            old_level = active_beast.stats.level
            level_ups = active_beast.stats.add_exp(self.config.xp_per_message,
                                                   active_beast.rarity)

            user.last_xp_gain = datetime.datetime.now()
            await self.db.update_user(user)
            await self.db.update_beast(active_beast_id, active_beast)

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

                    gain_text = []
                    for stat, gain in stat_gains.items():
                        if not stat.startswith('bonus_'):
                            gain_text.append(f"{stat.title()}: +{gain}")

                    embed.add_field(name="Stat Gains",
                                    value="\n".join(gain_text),
                                    inline=False)

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

                    embed.set_footer(
                        text=
                        f"XP gained from chatting: +{self.config.xp_per_message}"
                    )
                    await message.channel.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error in handle_xp_gain: {e}")

    async def on_command_error(self, ctx, error):
        """Global error handler"""
        if isinstance(error, commands.CommandNotFound):
            return

        if isinstance(error, commands.MissingPermissions):
            embed = discord.Embed(
                title="❌ Missing Permissions",
                description="You don't have permission to use this command.",
                color=0xFF0000)
            await ctx.send(embed=embed)

        self.logger.error(f"Command error in {ctx.command}: {error}")
        embed = discord.Embed(
            title="❌ Error",
            description="An error occurred while processing your command.",
            color=0xFF0000)
        await ctx.send(embed=embed)

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
            backup_file = await self.backup_manager.create_backup_with_cloud_storage(
            )
            if backup_file:
                self.logger.info(f"Enhanced backup completed: {backup_file}")
            else:
                self.logger.error("Enhanced backup failed")

        except Exception as e:
            self.logger.error(f"Enhanced backup task failed: {e}")

    # Replace your existing spawn_task with this improved version

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
            if not hasattr(self, '_next_spawn_time'):
                wait_minutes = random.randint(self.config.spawn_interval_min,
                                              self.config.spawn_interval_max)
                self._next_spawn_time = datetime.datetime.now(
                ) + datetime.timedelta(minutes=wait_minutes)
                self.logger.info(
                    f"Next beast spawn scheduled in {wait_minutes} minutes")
                return

            if datetime.datetime.now() < self._next_spawn_time:
                return  # Not time yet

            # Time to spawn!
            channel = self.get_channel(self.spawn_channel_id)
            if channel:
                await self.spawn_beast(channel)
                self.logger.info(
                    f"Beast spawned in channel: {channel.name} ({self.spawn_channel_id})"
                )

                # Schedule next spawn
                wait_minutes = random.randint(self.config.spawn_interval_min,
                                              self.config.spawn_interval_max)
                self._next_spawn_time = datetime.datetime.now(
                ) + datetime.timedelta(minutes=wait_minutes)
                self.logger.info(
                    f"Next beast spawn scheduled in {wait_minutes} minutes")
            else:
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

        async def wrapper(ctx, *args, **kwargs):
            config = ctx.bot.config

            if channel_type == "battle":
                if ctx.channel.id not in config.battle_channel_ids:
                    embed = discord.Embed(
                        title="❌ Wrong Channel",
                        description=
                        "This command can only be used in designated battle channels!",
                        color=0xFF0000)
                    await ctx.send(embed=embed)
                    return
            elif channel_type == "adopt":
                if ctx.channel.id != config.adopt_channel_id:
                    embed = discord.Embed(
                        title="❌ Wrong Channel",
                        description=
                        "This command can only be used in the adoption channel!",
                        color=0xFF0000)
                    await ctx.send(embed=embed)
                    return
            elif channel_type == "spawn":
                if ctx.channel.id != config.spawn_channel_id:
                    embed = discord.Embed(
                        title="❌ Wrong Channel",
                        description=
                        "This command can only be used in the beast spawn channel!",
                        color=0xFF0000)
                    await ctx.send(embed=embed)
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
            await ctx.send(embed=embed)
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
    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
        return

    # Check if there's already a beast spawned
    if ctx.bot.current_spawned_beast:
        embed = discord.Embed(
            title="❌ Beast Already Spawned",
            description=
            f"There's already a {ctx.bot.current_spawned_beast.name} waiting to be caught!",
            color=0xFF0000)
        await ctx.send(embed=embed)
        return

    try:
        await ctx.bot.spawn_beast(ctx.channel)
        embed = discord.Embed(
            title="✅ Beast Force Spawned",
            description="A wild beast has been manually spawned!",
            color=0x00FF00)
        await ctx.send(embed=embed)
    except Exception as e:
        embed = discord.Embed(title="❌ Spawn Failed",
                              description=f"Failed to spawn beast: {str(e)}",
                              color=0xFF0000)
        await ctx.send(embed=embed)


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

    await ctx.send(embed=embed)


@commands.command(name='nextspawn')
async def next_spawn_time(ctx):
    """Check when the next beast will spawn"""
    if not hasattr(ctx.bot, '_next_spawn_time'):
        embed = discord.Embed(
            title="⏰ Next Beast Spawn",
            description=
            "Spawn timing not initialized yet. Please wait a moment.",
            color=0xFFAA00)
        await ctx.send(embed=embed)
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

        embed.add_field(
            name="🎲 Spawn Range",
            value=
            f"{ctx.bot.config.spawn_interval_min}-{ctx.bot.config.spawn_interval_max} min",
            inline=True)

    # NEW: Show if there's currently a beast to catch
    if ctx.bot.current_spawned_beast:
        beast = ctx.bot.current_spawned_beast
        embed.add_field(
            name="🐉 Current Beast Available",
            value=f"{beast.name} {beast.rarity.emoji} - Go catch it!",
            inline=False)

    await ctx.send(embed=embed)


@commands.command(name='cloudbackup')
@commands.has_permissions(administrator=True)
async def manual_cloud_backup(ctx):
    """Create a manual backup with cloud storage"""
    embed = discord.Embed(
        title="☁️ Creating Cloud Backup...",
        description="Creating backup and uploading to cloud storage.",
        color=0xFFAA00)
    message = await ctx.send(embed=embed)

    try:
        backup_file = await ctx.bot.backup_manager.create_backup_with_cloud_storage(
        )
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

    await message.edit(embed=embed)


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
    message = await ctx.send(embed=embed)
    await message.add_reaction("✅")
    await message.add_reaction("❌")

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
            await message.edit(embed=embed)

            success = await ctx.bot.backup_manager.restore_from_cloud()
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

    await message.edit(embed=embed)


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
        await ctx.send(embed=embed)
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
            await ctx.send(embed=embed)
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

    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
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
    await ctx.send(embed=embed)


@commands.command(name='help')
async def help_command(ctx):
    """Show help information"""
    embed = discord.Embed(
        title="🐉 Immortal Beasts Bot - Help",
        description="Collect, train, and battle mythical beasts!",
        color=0x00AAFF)

    embed.add_field(
        name="📦 Collection Commands",
        value=
        f"`{ctx.bot.config.prefix}adopt` - Adopt a random beast (2 day cooldown)\n"
        f"`{ctx.bot.config.prefix}catch` - Catch a wild beast\n"
        f"`{ctx.bot.config.prefix}beasts` - View your beast collection\n"
        f"`{ctx.bot.config.prefix}beast <id>` - View detailed beast info",
        inline=False)

    embed.add_field(
        name="💰 Economy Commands",
        value=f"`{ctx.bot.config.prefix}stone` - Claim daily beast stones\n"
        f"`{ctx.bot.config.prefix}balance` - Check your beast stones",
        inline=False)

    embed.add_field(
        name="⚔️ Battle Commands",
        value=f"`{ctx.bot.config.prefix}battle @user` - Challenge another user\n"
        f"`{ctx.bot.config.prefix}active <beast_id>` - Set active beast for XP gain",
        inline=False)

    embed.add_field(
        name="🛠️ Admin Commands",
        value=f"`{ctx.bot.config.prefix}setchannel` - Set spawn channel\n"
        f"`{ctx.bot.config.prefix}removechannel` - Remove spawn channel",
        inline=False)

    embed.add_field(name="📊 Beast Rarities",
                    value="⭐ Common | ⭐⭐ Uncommon | ⭐⭐⭐ Rare\n"
                    "⭐⭐⭐⭐ Epic | ⭐⭐⭐⭐⭐ Legendary | ⭐⭐⭐⭐⭐⭐ Mythic",
                    inline=False)

    embed.set_footer(text="Beasts gain XP from chatting when set as active!")
    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
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
    await ctx.send(embed=embed)


@commands.command(name='active')
async def set_active_beast(ctx, beast_id: int):
    """Set a beast as your active beast for XP gain"""
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
        await ctx.send(embed=embed)
        return

    user.active_beast_id = beast_id
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🟢 Active Beast Set!",
        description=f"**{target_beast.name}** is now your active beast!\n"
        f"It will gain XP when you chat in XP channels",
        color=target_beast.rarity.color)
    embed.add_field(
        name="Beast",
        value=f"#{beast_id} {target_beast.name} {target_beast.rarity.emoji}",
        inline=True)
    embed.add_field(name="Level", value=target_beast.stats.level, inline=True)
    embed.add_field(name="XP per Message",
                    value=f"+{ctx.bot.config.xp_per_message}",
                    inline=True)

    await ctx.send(embed=embed)


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

    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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
    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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

    await ctx.send(embed=embed)


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
            await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to retrieve user beast data: {str(e)}",
            color=0xFF0000)
        await ctx.send(embed=embed)


@commands.command(name='userstats')
async def user_stats(ctx, user: discord.Member = None):
    """Check public stats for a user"""
    if user is None:
        user = ctx.author

    try:
        # Get user data
        user_data = await ctx.bot.get_or_create_user(user.id, str(user))
        user_beasts = await ctx.bot.db.get_user_beasts(user.id)
        user_role = ctx.bot.role_manager.get_user_role(user)
        beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

        embed = discord.Embed(title=f"📊 {user.display_name}'s Public Stats",
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
        await ctx.send(embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Error",
            description=f"Failed to retrieve user stats: {str(e)}",
            color=0xFF0000)
        await ctx.send(embed=embed)


@commands.command(name='leaderboard', aliases=['lb', 'top'])
async def leaderboard(ctx, category: str = "beasts", limit: int = 10):
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
        await ctx.send(embed=embed)
        return

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
            await ctx.send(embed=embed)
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
            await ctx.send(embed=embed)
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

        # Create leaderboard embed
        embed = discord.Embed(
            title=title,
            description=f"Top {min(limit, len(leaderboard_data))} players",
            color=0xFFD700)

        # Add leaderboard entries
        medals = ["🥇", "🥈", "🥉"]

        for i, entry in enumerate(leaderboard_data[:limit]):
            rank = i + 1
            medal = medals[i] if i < 3 else f"{rank}."

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
                    f"**{value:,}**{value_suffix}\n*{win_rate:.1f}% win rate*",
                    inline=True)
            else:
                embed.add_field(name=f"{medal} {entry['user'].display_name}",
                                value=f"**{value:,}**{value_suffix}",
                                inline=True)

        # Find requesting user's rank
        user_rank = None
        for i, entry in enumerate(leaderboard_data):
            if entry['user'].id == ctx.author.id:
                user_rank = i + 1
                break

        if user_rank and user_rank > limit:
            embed.set_footer(text=f"Your rank: #{user_rank}")
        elif user_rank:
            embed.set_footer(text=f"You're on the leaderboard! 🎉")
        else:
            embed.set_footer(
                text="Start your journey to climb the leaderboard!")

        await ctx.send(embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Leaderboard Error",
            description=f"Failed to generate leaderboard: {str(e)}",
            color=0xFF0000)
        await ctx.send(embed=embed)


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
        user_cursor = conn = ctx.bot.db._get_connection()
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
        await ctx.send(embed=embed)

    except Exception as e:
        embed = discord.Embed(
            title="❌ Server Stats Error",
            description=f"Failed to generate server statistics: {str(e)}",
            color=0xFF0000)
        await ctx.send(embed=embed)


@commands.command(name='battle', aliases=['fight'])
@require_channel("battle")
async def battle_command(ctx, opponent: discord.Member = None):
    """Challenge another user to a beast battle"""
    if opponent is None:
        embed = discord.Embed(
            title="❌ Invalid Usage",
            description=f"Usage: `{ctx.bot.config.prefix}battle @user`",
            color=0xFF0000)
        await ctx.send(embed=embed)
        return

    if opponent.bot:
        embed = discord.Embed(title="❌ Invalid Opponent",
                              description="You can't battle a bot!",
                              color=0xFF0000)
        await ctx.send(embed=embed)
        return

    if opponent.id == ctx.author.id:
        embed = discord.Embed(title="❌ Invalid Opponent",
                              description="You can't battle yourself!",
                              color=0xFF0000)
        await ctx.send(embed=embed)
        return

    # Get user beasts
    challenger_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    opponent_beasts = await ctx.bot.db.get_user_beasts(opponent.id)

    if not challenger_beasts:
        embed = discord.Embed(
            title="❌ No Beasts",
            description="You don't have any beasts to battle with!",
            color=0xFF0000)
        await ctx.send(embed=embed)
        return

    if not opponent_beasts:
        embed = discord.Embed(
            title="❌ Opponent Has No Beasts",
            description=
            f"{opponent.display_name} doesn't have any beasts to battle with!",
            color=0xFF0000)
        await ctx.send(embed=embed)
        return

    # Let challenger select their beast
    challenger_beast = await select_beast_for_battle(ctx, ctx.author,
                                                     challenger_beasts, "your")
    if not challenger_beast:
        embed = discord.Embed(title="❌ Battle Cancelled",
                              description="No beast selected for battle.",
                              color=0xFF0000)
        await ctx.send(embed=embed)
        return

    # Let opponent select their beast
    opponent_beast = await select_beast_for_battle(
        ctx, opponent, opponent_beasts, f"{opponent.display_name}'s")
    if not opponent_beast:
        embed = discord.Embed(
            title="❌ Battle Cancelled",
            description=f"{opponent.display_name} didn't select a beast.",
            color=0xFF0000)
        await ctx.send(embed=embed)
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

    await ctx.bot.db.update_user(challenger_user)
    await ctx.bot.db.update_user(opponent_user)

    # Create battle result embed
    if battle_result['result'] == BattleResult.WIN:
        color = 0x00FF00 if winner_user == ctx.author else 0xFF0000
        title = "🏆 Victory!" if winner_user == ctx.author else "💀 Defeat!"
    elif battle_result['result'] == BattleResult.LOSS:
        color = 0xFF0000 if winner_user == ctx.author else 0x00FF00
        title = "💀 Defeat!" if winner_user == ctx.author else "🏆 Victory!"
    else:
        color = 0xFFFF00
        title = "🤝 Draw!"

    embed = discord.Embed(title=f"⚔️ Battle Result: {title}", color=color)

    embed.add_field(
        name="🥊 Fighters",
        value=
        f"**{ctx.author.display_name}**: {challenger_beast[1].name} {challenger_beast[1].rarity.emoji}\n"
        f"**{opponent.display_name}**: {opponent_beast[1].name} {opponent_beast[1].rarity.emoji}",
        inline=False)

    if winner_user:
        embed.add_field(name="🏆 Winner",
                        value=winner_user.display_name,
                        inline=True)
    else:
        embed.add_field(name="🤝 Result", value="Draw", inline=True)

    embed.add_field(name="🔢 Turns", value=battle_result['turns'], inline=True)

    # Final HP
    embed.add_field(
        name="❤️ Final HP",
        value=
        f"**{challenger_beast[1].name}**: {battle_result['final_hp'][challenger_beast[1].name]}\n"
        f"**{opponent_beast[1].name}**: {battle_result['final_hp'][opponent_beast[1].name]}",
        inline=True)

    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
        return

    # Check if already used
    if user.has_used_adopt_legend:
        embed = discord.Embed(
            title="❌ Already Used",
            description=
            "You have already used your one-time legendary adoption!",
            color=0xFF0000)
        await ctx.send(embed=embed)
        return

    # Check inventory space
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=
            f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000)
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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
    await ctx.send(embed=embed)


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
        await ctx.send(embed=embed)
        return

    # Check if already used
    if user.has_used_adopt_mythic:
        embed = discord.Embed(
            title="❌ Already Used",
            description="You have already used your one-time mythic adoption!",
            color=0xFF0000)
        await ctx.send(embed=embed)
        return

    # Check inventory space
    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=
            f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000)
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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
    await ctx.send(embed=embed)


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

    await ctx.send(embed=embed)


# Replace the existing setchannel and removechannel commands with these:

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
    embed.add_field(name="⚠️ Note", 
                    value="This change is temporary. Update your environment variables for permanent change.",
                    inline=False)
    await ctx.send(embed=embed)

@commands.command(name='removechannel')
@commands.has_permissions(administrator=True)
async def remove_spawn_channel(ctx):
    """Remove current channel as spawn channel"""
    if ctx.channel.id == ctx.bot.spawn_channel_id:
        ctx.bot.spawn_channel_id = 0  # Disable spawning
        embed = discord.Embed(
            title="✅ Spawn Channel Removed",
            description=f"{ctx.channel.mention} is no longer the spawn channel.",
            color=0x00FF00)
    else:
        embed = discord.Embed(
            title="❌ Not the Spawn Channel",
            description=f"{ctx.channel.mention} is not the current spawn channel.",
            color=0xFF0000)
    await ctx.send(embed=embed)


@commands.command(name='catch')
@require_channel("spawn")
async def catch_beast(ctx):
    """Catch a spawned beast (3 attempts per user per beast)"""
    if not ctx.bot.current_spawned_beast:
        embed = discord.Embed(
            title="❌ No Beast Here",
            description="There's no beast to catch in this channel!",
            color=0xFF0000)
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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
        await ctx.send(embed=embed)
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

    await ctx.send(embed=embed)


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

    await ctx.send(embed=embed)


@commands.command(name='backup')
@commands.has_permissions(administrator=True)
async def manual_backup(ctx):
    """Create a manual backup"""
    embed = discord.Embed(
        title="🔄 Creating Backup...",
        description="Please wait while the backup is created.",
        color=0xFFAA00)
    message = await ctx.send(embed=embed)

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

    await message.edit(embed=embed)

# Add this command to help debug channel configuration:

@commands.command(name='channels')
@commands.has_permissions(administrator=True)
async def show_channel_config(ctx):
    """Show current channel configuration (admin only)"""
    embed = discord.Embed(
        title="📋 Channel Configuration",
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

    embed.add_field(
        name="⚔️ Battle Channels",
        value="\n".join(battle_channels) if battle_channels else "None configured",
        inline=False)

    # Adopt Channel
    adopt_channel = ctx.bot.get_channel(ctx.bot.config.adopt_channel_id)
    embed.add_field(
        name="🐾 Adopt Channel",
        value=f"#{adopt_channel.name}" if adopt_channel else f"Unknown ({ctx.bot.config.adopt_channel_id})",
        inline=True)

    # Spawn Channel
    spawn_channel = ctx.bot.get_channel(ctx.bot.spawn_channel_id)
    embed.add_field(
        name="🌟 Spawn Channel", 
        value=f"#{spawn_channel.name}" if spawn_channel else f"Unknown ({ctx.bot.spawn_channel_id})",
        inline=True)

    # XP Channels
    xp_channels = []
    for channel_id in ctx.bot.config.xp_chat_channel_ids:
        channel = ctx.bot.get_channel(channel_id)
        if channel:
            xp_channels.append(f"#{channel.name}")
        else:
            xp_channels.append(f"Unknown ({channel_id})")

    embed.add_field(
        name="💫 XP Channels",
        value="\n".join(xp_channels[:5]) + (f"\n... and {len(xp_channels)-5} more" if len(xp_channels) > 5 else "") if xp_channels else "None configured",
        inline=False)

    embed.add_field(
        name="📍 Current Channel",
        value=f"#{ctx.channel.name} ({ctx.channel.id})",
        inline=False)

    await ctx.send(embed=embed)


@commands.command(name='cleanbackups')
@commands.has_permissions(administrator=True)
async def clean_backups(ctx, keep_count: int = None):
    """Clean old backups manually"""
    if keep_count is None:
        keep_count = ctx.bot.config.backup_retention_count

    if keep_count < 1:
        embed = discord.Embed(title="❌ Invalid Count",
                              description="Keep count must be at least 1.",
                              color=0xFF0000)
        await ctx.send(embed=embed)
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

    await ctx.send(embed=embed)


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

    # Add all commands to the bot
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
    bot.add_command(set_spawn_channel)    # Uses the fixed version
    bot.add_command(remove_spawn_channel) # Uses the fixed version
    bot.add_command(show_channel_config)

    # ADD THESE NEW HEAL COMMANDS:
    bot.add_command(heal_beast)  # !heal command
    bot.add_command(heal_all_beasts)  # !healall command

    bot.add_command(manual_cloud_backup)  #mmmmm
    bot.add_command(restore_from_cloud)
    bot.add_command(check_user_beasts)  # !checkbeasts
    bot.add_command(user_stats)  # !userstats
    bot.add_command(leaderboard)  # !leaderboard
    bot.add_command(server_beast_stats)  # !serverbeasts

    try:
        bot.run(config.token)
    except discord.LoginFailure:
        print("Invalid bot token. Please check your configuration.")
    except Exception as e:
        print(f"Error running bot: {e}")


if __name__ == "__main__":
    if os.getenv('PORT'):  # Running on Render
        run_bot_with_flask()
    else:  # Running locally
        main()
