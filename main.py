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


# ============================================================================
# Configuration Management
# ============================================================================

class BotConfig(BaseModel):
    """Bot configuration with validation"""
    token: str = Field(..., description="Discord bot token")
    prefix: str = Field(default="!", description="Command prefix")
    database_path: str = Field(default="data/immortal_beasts.db", description="Database file path")
    backup_interval_hours: int = Field(default=6, ge=1, le=24, description="Backup interval in hours")
    spawn_interval_min: int = Field(default=15, ge=5, le=120, description="Minimum spawn interval in minutes")
    spawn_interval_max: int = Field(default=45, ge=5, le=120, description="Maximum spawn interval in minutes")
    log_level: str = Field(default="INFO", description="Logging level")

    # Inventory limits
    normal_user_beast_limit: int = Field(default=6, description="Beast limit for normal users")
    special_role_beast_limit: int = Field(default=8, description="Beast limit for special role users")
    personal_role_beast_limit: int = Field(default=10, description="Beast limit for personal role users")

    # Role IDs - UPDATED FOR MULTIPLE ROLES
    special_role_ids: List[int] = Field(default=[], description="List of special role IDs for 8 beast limit and adopt legend")
    personal_role_id: int = Field(default=0, description="Personal role ID for 10 beast limit and adopt mythic")

    # XP System - UPDATED FOR MULTIPLE CHANNELS
    xp_chat_channel_ids: List[int] = Field(default=[], description="List of channel IDs where beasts gain XP from chatting")
    xp_per_message: int = Field(default=5, description="XP gained per message")
    xp_cooldown_seconds: int = Field(default=60, description="Cooldown between XP gains from same user")

    # Starting resources
    starting_beast_stones: int = Field(default=1000, ge=0, description="Starting beast stones for new users")

    # Adopt cooldowns (in hours)
    adopt_cooldown_hours: int = Field(default=48, description="Adopt command cooldown in hours (2 days = 48)")

    @field_validator('spawn_interval_max')
    @classmethod
    def validate_spawn_intervals(cls, v, info):
        if info.data.get('spawn_interval_min') and v <= info.data['spawn_interval_min']:
            raise ValueError('spawn_interval_max must be greater than spawn_interval_min')
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
                xp_chat_channel_ids=[1393424880787259482, 1393626191935705198, 1394289930515124325, 1393163125850640414]
            )
            with open(config_path, 'w') as f:
                yaml.dump(default_config.model_dump(), f, default_flow_style=False)
            raise FileNotFoundError(f"Config file created at {config_path}. Please fill in your bot token, role IDs, and XP channel IDs.")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def load_from_environment(cls) -> 'BotConfig':
        """Load configuration from environment variables (for production)"""
        token = os.getenv('DISCORD_BOT_TOKEN')
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable is required")

        # Parse role IDs from environment (comma-separated)
        special_role_ids = []
        special_roles_env = os.getenv('SPECIAL_ROLE_IDS', '1393927051685400790,1393094845479780426')
        if special_roles_env:
            special_role_ids = [int(role_id.strip()) for role_id in special_roles_env.split(',') if role_id.strip()]

        # Parse XP channel IDs from environment (comma-separated)
        xp_channel_ids = []
        xp_channels_env = os.getenv('XP_CHANNEL_IDS', '1393424880787259482,1393626191935705198,1394289930515124325,1393163125850640414')
        if xp_channels_env:
            xp_channel_ids = [int(channel_id.strip()) for channel_id in xp_channels_env.split(',') if channel_id.strip()]

        return cls(
            token=token,
            prefix=os.getenv('BOT_PREFIX', '!'),
            database_path='/tmp/immortal_beasts.db',  # Use /tmp for Render
            special_role_ids=special_role_ids,
            personal_role_id=int(os.getenv('PERSONAL_ROLE_ID', '1393176170601775175')),
            xp_chat_channel_ids=xp_channel_ids,
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            backup_interval_hours=int(os.getenv('BACKUP_INTERVAL_HOURS', '6')),
            spawn_interval_min=int(os.getenv('SPAWN_INTERVAL_MIN', '15')),
            spawn_interval_max=int(os.getenv('SPAWN_INTERVAL_MAX', '45')),
            xp_per_message=int(os.getenv('XP_PER_MESSAGE', '5')),
            xp_cooldown_seconds=int(os.getenv('XP_COOLDOWN_SECONDS', '60')),
            starting_beast_stones=int(os.getenv('STARTING_BEAST_STONES', '1000')),
            adopt_cooldown_hours=int(os.getenv('ADOPT_COOLDOWN_HOURS', '48'))
        )


# ============================================================================
# Enums and Constants
# ============================================================================

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
        rates = {1: 95, 2: 80, 3: 65, 4: 40, 5: 25, 6: 10}
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


# ============================================================================
# Data Models
# ============================================================================

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
        return int(base_req * (self.level ** 1.2))

    def get_stat_gains(self, rarity: BeastRarity) -> Dict[str, Tuple[int, int]]:
        """Get stat gain ranges based on rarity"""
        stat_ranges = {
            BeastRarity.COMMON: {'hp': (8, 15), 'attack': (2, 5), 'defense': (1, 3), 'speed': (1, 3)},
            BeastRarity.UNCOMMON: {'hp': (10, 18), 'attack': (3, 6), 'defense': (2, 4), 'speed': (1, 4)},
            BeastRarity.RARE: {'hp': (12, 22), 'attack': (4, 8), 'defense': (2, 5), 'speed': (2, 5)},
            BeastRarity.EPIC: {'hp': (15, 28), 'attack': (5, 10), 'defense': (3, 7), 'speed': (3, 7)},
            BeastRarity.LEGENDARY: {'hp': (20, 35), 'attack': (7, 14), 'defense': (4, 9), 'speed': (4, 9)},
            BeastRarity.MYTHIC: {'hp': (25, 45), 'attack': (10, 18), 'defense': (6, 12), 'speed': (6, 12)}
        }
        return stat_ranges[rarity]

    def get_bonus_stat_ranges(self, rarity: BeastRarity) -> Dict[str, Tuple[int, int]]:
        """Get bonus stat ranges for every 5 levels"""
        bonus_ranges = {
            BeastRarity.COMMON: {'hp': (5, 10), 'attack': (1, 3), 'defense': (1, 2), 'speed': (1, 2)},
            BeastRarity.UNCOMMON: {'hp': (8, 15), 'attack': (2, 4), 'defense': (1, 3), 'speed': (1, 3)},
            BeastRarity.RARE: {'hp': (12, 20), 'attack': (3, 6), 'defense': (2, 4), 'speed': (2, 4)},
            BeastRarity.EPIC: {'hp': (15, 25), 'attack': (4, 8), 'defense': (3, 6), 'speed': (3, 6)},
            BeastRarity.LEGENDARY: {'hp': (20, 35), 'attack': (6, 12), 'defense': (4, 8), 'speed': (4, 8)},
            BeastRarity.MYTHIC: {'hp': (30, 50), 'attack': (8, 16), 'defense': (6, 12), 'speed': (6, 12)}
        }
        return bonus_ranges[rarity]

    def level_up(self, rarity: BeastRarity) -> Tuple[bool, bool, Dict[str, int]]:
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

        stat_gains = {'hp': hp_gain, 'attack': attack_gain, 'defense': defense_gain, 'speed': speed_gain}

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

    def add_exp(self, amount: int, rarity: BeastRarity) -> List[Tuple[bool, bool, Dict[str, int]]]:
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

        stats = BeastStats(
            hp=base_hp,
            max_hp=base_hp,
            attack=base_attack,
            defense=self.rarity.value * 5 + random.randint(1, 10),
            speed=random.randint(10, 50)
        )

        return Beast(
            name=self.name,
            rarity=self.rarity,
            tendency=self.tendency,
            location=self.location,
            stats=stats,
            description=self.description
        )


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
        return cls(
            name=data['name'],
            rarity=BeastRarity(data['rarity']),
            tendency=data['tendency'],
            location=data['location'],
            stats=stats,
            description=data.get('description', ''),
            caught_at=datetime.datetime.fromisoformat(data['caught_at']),
            owner_id=data.get('owner_id'),
            unique_id=data.get('unique_id')
        )

    @property
    def power_level(self) -> int:
        """Calculate overall power level"""
        return (self.stats.hp + self.stats.attack + self.stats.defense + self.stats.speed) * self.stats.level


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


# ============================================================================
# Database Layer
# ============================================================================

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
    async def get_user_beasts(self, user_id: int, limit: int = None) -> List[Tuple[int, Beast]]:
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
            cursor = conn.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                return User(
                    user_id=row['user_id'],
                    username=row['username'],
                    spirit_stones=row['spirit_stones'],
                    last_daily=datetime.datetime.fromisoformat(row['last_daily']) if row['last_daily'] else None,
                    last_adopt=datetime.datetime.fromisoformat(row['last_adopt']) if row['last_adopt'] else None,
                    last_xp_gain=datetime.datetime.fromisoformat(row['last_xp_gain']) if row['last_xp_gain'] else None,
                    active_beast_id=row['active_beast_id'],
                    total_catches=row['total_catches'],
                    total_battles=row['total_battles'],
                    wins=row['wins'],
                    losses=row['losses'],
                    has_used_adopt_legend=bool(row['has_used_adopt_legend']),
                    has_used_adopt_mythic=bool(row['has_used_adopt_mythic']),
                    created_at=datetime.datetime.fromisoformat(row['created_at'])
                )
            return None
        finally:
            conn.close()

    async def create_user(self, user: User) -> bool:
        """Create new user"""
        try:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO users 
                    (user_id, username, spirit_stones, total_catches, total_battles, wins, losses, 
                     has_used_adopt_legend, has_used_adopt_mythic, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id, user.username, user.spirit_stones,
                    user.total_catches, user.total_battles, user.wins, user.losses,
                    user.has_used_adopt_legend, user.has_used_adopt_mythic,
                    user.created_at.isoformat()
                ))
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
                conn.execute("""
                    UPDATE users SET 
                    username=?, spirit_stones=?, last_daily=?, last_adopt=?, last_xp_gain=?,
                    active_beast_id=?, total_catches=?, total_battles=?, wins=?, losses=?,
                    has_used_adopt_legend=?, has_used_adopt_mythic=?
                    WHERE user_id=?
                """, (
                    user.username, user.spirit_stones,
                    user.last_daily.isoformat() if user.last_daily else None,
                    user.last_adopt.isoformat() if user.last_adopt else None,
                    user.last_xp_gain.isoformat() if user.last_xp_gain else None,
                    user.active_beast_id, user.total_catches, user.total_battles, user.wins, user.losses,
                    user.has_used_adopt_legend, user.has_used_adopt_mythic,
                    user.user_id
                ))
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
            cursor = conn.execute("""
                INSERT INTO beasts (owner_id, beast_data, caught_at)
                VALUES (?, ?, ?)
            """, (beast.owner_id, json.dumps(beast.to_dict()), beast.caught_at.isoformat()))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    async def get_user_beasts(self, user_id: int, limit: int = None) -> List[Tuple[int, Beast]]:
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
                conn.execute("DELETE FROM beasts WHERE id = ?", (beast_id,))
                conn.commit()
                return True
            finally:
                conn.close()
        except Exception as e:
            logging.error(f"Failed to delete beast {beast_id}: {e}")
            return False

    async def backup_database(self, backup_dir: str = "backups") -> Optional[str]:
        """Create database backup"""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"backup_{timestamp}.db"
        try:
            shutil.copy2(self.db_path, backup_file)
            logging.info(f"Database backup created: {backup_file}")
            return str(backup_file)
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return None


# ============================================================================
# Battle System and Template Manager
# ============================================================================

class BattleEngine:
    """Handles beast battles"""

    @staticmethod
    def calculate_damage(attacker: Beast, defender: Beast) -> int:
        """Calculate damage dealt"""
        base_damage = attacker.stats.attack
        defense_reduction = defender.stats.defense * 0.5
        damage_variance = random.randint(-10, 10)
        final_damage = max(1, int(base_damage - defense_reduction + damage_variance))
        return final_damage

    @staticmethod
    def determine_turn_order(beast1: Beast, beast2: Beast) -> Tuple[Beast, Beast]:
        """Determine which beast goes first based on speed"""
        if beast1.stats.speed > beast2.stats.speed:
            return beast1, beast2
        elif beast2.stats.speed > beast1.stats.speed:
            return beast2, beast1
        else:
            return random.choice([(beast1, beast2), (beast2, beast1)])

    async def simulate_battle(self, beast1: Beast, beast2: Beast) -> Dict[str, Any]:
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
                        base_attack_range=tuple(template_data['base_attack_range']),
                        description=template_data.get('description', '')
                    )
                except Exception as e:
                    logging.error(f"Failed to load beast template '{name}': {e}")
        except Exception as e:
            logging.error(f"Failed to load beast templates from {self.data_file}: {e}")
            self._create_default_templates()

    def _create_default_templates(self):
        """Create default beast templates"""
        default_data = {
            "Flood Dragonling": {
                "rarity": 1, "tendency": "n/a", "location": "Fire",
                "base_hp_range": [80, 120], "base_attack_range": [15, 25],
                "description": "A young dragon with control over floods"
            },
            "Shenghuang": {
                "rarity": 2, "tendency": "n/a", "location": "Heaven Mountain",
                "base_hp_range": [120, 180], "base_attack_range": [25, 35],
                "description": "A celestial phoenix of divine nature"
            },
            "Shadow Monster": {
                "rarity": 3, "tendency": "n/a", "location": "Tai Di Event",
                "base_hp_range": [180, 250], "base_attack_range": [35, 50],
                "description": "A creature born from pure shadow"
            },
            "Azure Dragon": {
                "rarity": 4, "tendency": "Primordial Land +12%", "location": "Heaven Space Palace",
                "base_hp_range": [300, 450], "base_attack_range": [60, 90],
                "description": "A mighty dragon of the eastern skies"
            },
            "Qing Greenbull": {
                "rarity": 5, "tendency": "Skyshine Continent +15%", "location": "Mystic Forest",
                "base_hp_range": [500, 700], "base_attack_range": [100, 140],
                "description": "A legendary bull with emerald horns"
            },
            "Yinglong": {
                "rarity": 6, "tendency": "LeiZu, Xianle, or Taiyuan +18%", "location": "Mythical Realm",
                "base_hp_range": [800, 1200], "base_attack_range": [150, 200],
                "description": "A legendary winged dragon of immense power"
            }
        }

        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_data, f, default_flow_style=False)
            logging.info(f"Created default beast templates in {self.data_file}")
            # Load the templates we just created
            self._load_templates()
        except Exception as e:
            logging.error(f"Failed to create default templates: {e}")

    def get_random_template_by_rarity(self, rarity: BeastRarity) -> Optional[BeastTemplate]:
        """Get random template of specific rarity"""
        rarity_templates = [template for template in self.templates.values() if template.rarity == rarity]
        return random.choice(rarity_templates) if rarity_templates else None

    def get_random_template_up_to_rarity(self, max_rarity: BeastRarity, rarity_weights: Dict[BeastRarity, float] = None) -> BeastTemplate:
        """Get random template up to specified rarity with weights"""
        if not self.templates:
            raise ValueError("No beast templates available")

        if rarity_weights is None:
            rarity_weights = {
                BeastRarity.COMMON: 35, BeastRarity.UNCOMMON: 30, BeastRarity.RARE: 20,
                BeastRarity.EPIC: 12, BeastRarity.LEGENDARY: 3
            }

        available_templates = []
        weights = []
        for template in self.templates.values():
            if template.rarity.value <= max_rarity.value and template.rarity in rarity_weights:
                available_templates.append(template)
                weights.append(rarity_weights[template.rarity])

        if not available_templates:
            available_templates = [t for t in self.templates.values() if t.rarity.value <= max_rarity.value]
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
        elif any(role_id in user_role_ids for role_id in self.config.special_role_ids):
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


async def select_beast_for_battle(ctx, user: discord.Member, beasts: List[Tuple[int, Beast]], pronoun: str = "your") -> Optional[Tuple[int, Beast]]:
    """Helper function to let a user select a beast for battle"""
    if not beasts:
        return None

    embed = discord.Embed(
        title=f"Select {pronoun.title()} Beast for Battle",
        description=f"{user.mention}, choose a beast by reacting with the corresponding number:",
        color=0x00AAFF
    )

    options = beasts[:10]
    number_emojis = ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '5️⃣', '6️⃣', '7️⃣', '8️⃣', '9️⃣', '🔟']

    for i, (beast_id, beast) in enumerate(options):
        embed.add_field(
            name=f"{number_emojis[i]} #{beast_id} {beast.name} {beast.rarity.emoji}",
            value=f"Level {beast.stats.level} | HP: {beast.stats.hp}/{beast.stats.max_hp} | Power: {beast.power_level}",
            inline=False
        )

    message = await ctx.send(embed=embed)

    for i in range(len(options)):
        await message.add_reaction(number_emojis[i])

    def check(reaction, react_user):
        return (react_user == user and 
                str(reaction.emoji) in number_emojis[:len(options)] and 
                reaction.message.id == message.id)

    try:
        reaction, _ = await ctx.bot.wait_for('reaction_add', timeout=30.0, check=check)
        selected_index = number_emojis.index(str(reaction.emoji))
        selected_beast = options[selected_index]
        await message.delete()
        return selected_beast
    except asyncio.TimeoutError:
        await message.delete()
        return None


# ============================================================================
# Main Bot Class
# ============================================================================

class ImmortalBeastsBot(commands.Bot):
    """Main bot class"""

    def __init__(self, config: BotConfig):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=config.prefix, intents=intents, help_command=None)

        self.config = config
        self.db: DatabaseInterface = SQLiteDatabase(config.database_path)
        self.template_manager = BeastTemplateManager()
        self.role_manager = UserRoleManager(config)
        self.battle_engine = BattleEngine()
        self.spawn_channels: Dict[int, Beast] = {}

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def setup_hook(self):
        """Setup hook called when bot starts"""
        await self.db.initialize()
        self.logger.info("Database initialized")
        self.backup_task.start()
        self.spawn_task.start()
        self.logger.info("Background tasks started")

    async def on_ready(self):
        """Called when bot is ready"""
        self.logger.info(f'{self.user} has connected to Discord!')
        activity = discord.Game(name=f"Immortal Beasts | {self.config.prefix}help")
        await self.change_presence(activity=activity)

    async def on_message(self, message):
        """Handle messages for XP gain"""
        if message.author.bot:
            await self.process_commands(message)
            return

        # Check if message is in any XP channel (updated for multiple channels)
        if (message.channel.id in self.config.xp_chat_channel_ids and 
            hasattr(message, 'content') and len(message.content) > 3):
            await self.handle_xp_gain(message)

        await self.process_commands(message)

    async def handle_xp_gain(self, message):
        """Handle XP gain from chatting"""
        try:
            user = await self.get_or_create_user(message.author.id, str(message.author))

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
            level_ups = active_beast.stats.add_exp(self.config.xp_per_message, active_beast.rarity)

            user.last_xp_gain = datetime.datetime.now()
            await self.db.update_user(user)
            await self.db.update_beast(active_beast_id, active_beast)

            if level_ups:
                for leveled_up, bonus_level, stat_gains in level_ups:
                    embed = discord.Embed(
                        title="🎉 Level Up!",
                        description=f"{message.author.mention}'s **{active_beast.name}** leveled up!",
                        color=active_beast.rarity.color
                    )
                    embed.add_field(name="New Level", value=f"Level {active_beast.stats.level}", inline=True)
                    embed.add_field(name="Power Level", value=active_beast.power_level, inline=True)

                    gain_text = []
                    for stat, gain in stat_gains.items():
                        if not stat.startswith('bonus_'):
                            gain_text.append(f"{stat.title()}: +{gain}")

                    embed.add_field(name="Stat Gains", value="\n".join(gain_text), inline=False)

                    if bonus_level:
                        bonus_text = []
                        for stat, gain in stat_gains.items():
                            if stat.startswith('bonus_'):
                                clean_stat = stat.replace('bonus_', '')
                                bonus_text.append(f"{clean_stat.title()}: +{gain}")
                        embed.add_field(name="🌟 Bonus Stats (Level 5 Multiple)!", value="\n".join(bonus_text), inline=False)

                    embed.set_footer(text=f"XP gained from chatting: +{self.config.xp_per_message}")
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
                color=0xFF0000
            )
            await ctx.send(embed=embed)

        self.logger.error(f"Command error in {ctx.command}: {error}")
        embed = discord.Embed(
            title="❌ Error",
            description="An error occurred while processing your command.",
            color=0xFF0000
        )
        await ctx.send(embed=embed)

    async def get_or_create_user(self, user_id: int, username: str) -> User:
        """Get user or create if doesn't exist"""
        user = await self.db.get_user(user_id)
        if not user:
            user = User(
                user_id=user_id,
                username=username,
                spirit_stones=self.config.starting_beast_stones
            )
            await self.db.create_user(user)
        return user

    @tasks.loop(hours=6)
    async def backup_task(self):
        """Regular database backup"""
        try:
            await self.db.backup_database()
        except Exception as e:
            self.logger.error(f"Backup task failed: {e}")

    @tasks.loop(minutes=30)
    async def spawn_task(self):
        """Regular beast spawning"""
        try:
            if not self.spawn_channels:
                return

            wait_time = random.randint(
                self.config.spawn_interval_min * 60,
                self.config.spawn_interval_max * 60
            )
            await asyncio.sleep(wait_time)

            channel_id = random.choice(list(self.spawn_channels.keys()))
            channel = self.get_channel(channel_id)

            if channel:
                await self.spawn_beast(channel)
        except Exception as e:
            self.logger.error(f"Spawn task failed: {e}")

    async def spawn_beast(self, channel: discord.TextChannel):
        """Spawn a beast in the given channel"""
        try:
            rarity_weights = {
                BeastRarity.COMMON: 50, BeastRarity.UNCOMMON: 25, BeastRarity.RARE: 15,
                BeastRarity.EPIC: 7, BeastRarity.LEGENDARY: 2.5, BeastRarity.MYTHIC: 0.5
            }

            template = self.template_manager.get_random_template_up_to_rarity(BeastRarity.MYTHIC, rarity_weights)
            beast = template.create_beast()

            self.spawn_channels[channel.id] = beast

            embed = discord.Embed(
                title="🌟 A Wild Beast Appeared! 🌟",
                description=f"**{beast.name}** has appeared!\n{beast.rarity.emoji}\n\nQuick! Use `{self.config.prefix}catch` to capture it!",
                color=beast.rarity.color
            )

            embed.add_field(name="Rarity", value=beast.rarity.emoji, inline=True)
            embed.add_field(name="Level", value=beast.stats.level, inline=True)
            embed.add_field(name="Power", value=beast.power_level, inline=True)
            embed.add_field(name="Tendency", value=beast.tendency or "None", inline=False)
            embed.add_field(name="Location", value=beast.location or "Unknown", inline=False)

            if beast.description:
                embed.add_field(name="Description", value=beast.description, inline=False)

            await channel.send(embed=embed)

            # Beast disappears after 5 minutes
            await asyncio.sleep(300)
            if channel.id in self.spawn_channels:
                del self.spawn_channels[channel.id]
                embed = discord.Embed(
                    title="💨 Beast Fled",
                    description=f"The {beast.name} has disappeared...",
                    color=0x808080
                )
                await channel.send(embed=embed)
        except Exception as e:
            self.logger.error(f"Error spawning beast: {e}")


# ============================================================================
# Commands
# ============================================================================

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
                description=f"You can claim your next daily beast stones in {hours}h {minutes}m",
                color=0xFF8000
            )
            await ctx.send(embed=embed)
            return

    daily_reward = 100
    user.spirit_stones += daily_reward
    user.last_daily = now
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🔮 Daily Beast Stones Claimed!",
        description=f"**{ctx.author.display_name}** received **{daily_reward} Beast Stones**!",
        color=0x9932CC
    )

    embed.add_field(name="💎 Reward", value=f"{daily_reward} beast stones", inline=True)
    embed.add_field(name="💰 Total Beast Stones", value=f"{user.spirit_stones:,} stones", inline=True)
    embed.add_field(name="⏰ Next Claim", value=f"<t:{int((now + datetime.timedelta(hours=24)).timestamp())}:R>", inline=False)

    embed.set_footer(text="Come back tomorrow for another 100 beast stones!")
    await ctx.send(embed=embed)


@commands.command(name='adopt')
async def adopt_beast(ctx):
    """Adopt a random beast (available every 2 days)"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    now = datetime.datetime.now()
    if user.last_adopt:
        cooldown_hours = ctx.bot.config.adopt_cooldown_hours
        time_since_last = now - user.last_adopt
        if time_since_last.total_seconds() < cooldown_hours * 3600:
            remaining_time = datetime.timedelta(hours=cooldown_hours) - time_since_last
            hours = int(remaining_time.total_seconds() // 3600)
            minutes = int((remaining_time.total_seconds() % 3600) // 60)

            embed = discord.Embed(
                title="⏰ Adopt Cooldown",
                description=f"You can adopt another beast in {hours}h {minutes}m",
                color=0xFF8000
            )
            await ctx.send(embed=embed)
            return

    template = ctx.bot.template_manager.get_random_template_up_to_rarity(BeastRarity.LEGENDARY)
    beast = template.create_beast()
    beast.owner_id = ctx.author.id

    beast_id = await ctx.bot.db.add_beast(beast)

    user.last_adopt = now
    user.total_catches += 1
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🎉 Beast Adopted!",
        description=f"**{ctx.author.display_name}** adopted **{beast.name}**!\n{beast.rarity.emoji}",
        color=beast.rarity.color
    )
    embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
    embed.add_field(name="Level", value=beast.stats.level, inline=True)
    embed.add_field(name="Power Level", value=beast.power_level, inline=True)
    embed.add_field(name="Tendency", value=beast.tendency or "None", inline=False)
    embed.add_field(name="Next Adopt", value=f"<t:{int((now + datetime.timedelta(hours=ctx.bot.config.adopt_cooldown_hours)).timestamp())}:R>", inline=False)

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
            description=f"You don't have any beasts yet!\nUse `{ctx.bot.config.prefix}adopt` or catch wild beasts to start your collection.",
            color=0x808080
        )
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
        description=f"**{len(user_beasts)}/{beast_limit}** beasts | Page {page}/{total_pages}",
        color=0x00AAFF
    )

    for beast_id, beast in page_beasts:
        active_indicator = "🟢 " if beast_id == user.active_beast_id else ""
        embed.add_field(
            name=f"{active_indicator}#{beast_id} {beast.name} {beast.rarity.emoji}",
            value=f"Level {beast.stats.level} | HP: {beast.stats.hp}/{beast.stats.max_hp}\n"
                  f"Power: {beast.power_level} | Location: {beast.location}",
            inline=False
        )

    embed.set_footer(text=f"💰 {user.spirit_stones:,} Beast Stones | Use {ctx.bot.config.prefix}beast <id> for details")
    await ctx.send(embed=embed)


@commands.command(name='help')
async def help_command(ctx):
    """Show help information"""
    embed = discord.Embed(
        title="🐉 Immortal Beasts Bot - Help",
        description="Collect, train, and battle mythical beasts!",
        color=0x00AAFF
    )

    embed.add_field(
        name="📦 Collection Commands",
        value=f"`{ctx.bot.config.prefix}adopt` - Adopt a random beast (2 day cooldown)\n"
              f"`{ctx.bot.config.prefix}catch` - Catch a wild beast\n"
              f"`{ctx.bot.config.prefix}beasts` - View your beast collection\n"
              f"`{ctx.bot.config.prefix}beast <id>` - View detailed beast info",
        inline=False
    )

    embed.add_field(
        name="💰 Economy Commands",
        value=f"`{ctx.bot.config.prefix}stone` - Claim daily beast stones\n"
              f"`{ctx.bot.config.prefix}balance` - Check your beast stones",
        inline=False
    )

    embed.add_field(
        name="⚔️ Battle Commands",
        value=f"`{ctx.bot.config.prefix}battle @user` - Challenge another user\n"
              f"`{ctx.bot.config.prefix}active <beast_id>` - Set active beast for XP gain",
        inline=False
    )

    embed.add_field(
        name="🛠️ Admin Commands",
        value=f"`{ctx.bot.config.prefix}setchannel` - Set spawn channel\n"
              f"`{ctx.bot.config.prefix}removechannel` - Remove spawn channel",
        inline=False
    )

    embed.add_field(
        name="📊 Beast Rarities",
        value="⭐ Common | ⭐⭐ Uncommon | ⭐⭐⭐ Rare\n"
              "⭐⭐⭐⭐ Epic | ⭐⭐⭐⭐⭐ Legendary | ⭐⭐⭐⭐⭐⭐ Mythic",
        inline=False
    )

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
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    embed = discord.Embed(
        title=f"🐉 {target_beast.name} {target_beast.rarity.emoji}",
        description=target_beast.description or "A mysterious beast",
        color=target_beast.rarity.color
    )

    # Status indicator
    status = "🟢 Active" if beast_id == user.active_beast_id else "⚪ Inactive"
    embed.add_field(name="Status", value=status, inline=True)
    embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
    embed.add_field(name="Rarity", value=target_beast.rarity.name.title(), inline=True)

    # Stats
    embed.add_field(name="Level", value=target_beast.stats.level, inline=True)
    embed.add_field(name="Experience", value=f"{target_beast.stats.exp}/{target_beast.stats.get_level_up_requirements(target_beast.rarity)}", inline=True)
    embed.add_field(name="Power Level", value=target_beast.power_level, inline=True)

    # Detailed stats
    embed.add_field(
        name="📊 Combat Stats",
        value=f"**HP:** {target_beast.stats.hp}/{target_beast.stats.max_hp}\n"
              f"**Attack:** {target_beast.stats.attack}\n"
              f"**Defense:** {target_beast.stats.defense}\n"
              f"**Speed:** {target_beast.stats.speed}",
        inline=True
    )

    # Location and tendency
    embed.add_field(
        name="🌍 Origin",
        value=f"**Location:** {target_beast.location}\n"
              f"**Tendency:** {target_beast.tendency}",
        inline=True
    )

    # Caught date
    caught_date = target_beast.caught_at.strftime("%Y-%m-%d %H:%M")
    embed.add_field(name="📅 Caught", value=caught_date, inline=True)

    embed.set_footer(text=f"Use {ctx.bot.config.prefix}active {beast_id} to set as active beast")
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
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    user.active_beast_id = beast_id
    await ctx.bot.db.update_user(user)

    embed = discord.Embed(
        title="🟢 Active Beast Set!",
        description=f"**{target_beast.name}** is now your active beast!\n"
                   f"It will gain XP when you chat in XP channels",
        color=target_beast.rarity.color
    )
    embed.add_field(name="Beast", value=f"#{beast_id} {target_beast.name} {target_beast.rarity.emoji}", inline=True)
    embed.add_field(name="Level", value=target_beast.stats.level, inline=True)
    embed.add_field(name="XP per Message", value=f"+{ctx.bot.config.xp_per_message}", inline=True)

    await ctx.send(embed=embed)


@commands.command(name='balance', aliases=['stones'])
async def show_balance(ctx):
    """Show your beast stone balance"""
    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))

    embed = discord.Embed(
        title="💰 Beast Stone Balance",
        description=f"**{ctx.author.display_name}** has **{user.spirit_stones:,}** Beast Stones",
        color=0x9932CC
    )

    # Next daily claim time
    if user.last_daily:
        next_daily = user.last_daily + datetime.timedelta(hours=24)
        if datetime.datetime.now() < next_daily:
            embed.add_field(name="⏰ Next Daily", value=f"<t:{int(next_daily.timestamp())}:R>", inline=True)
        else:
            embed.add_field(name="✅ Daily Available", value="Use !stone to claim", inline=True)
    else:
        embed.add_field(name="✅ Daily Available", value="Use !stone to claim", inline=True)

    embed.add_field(name="📊 Stats", value=f"Total Catches: {user.total_catches}\nBattles: {user.total_battles}\nWin Rate: {user.win_rate:.1f}%", inline=True)

    await ctx.send(embed=embed)


@commands.command(name='battle', aliases=['fight'])
async def battle_command(ctx, opponent: discord.Member = None):
    """Challenge another user to a beast battle"""
    if opponent is None:
        embed = discord.Embed(
            title="❌ Invalid Usage",
            description=f"Usage: `{ctx.bot.config.prefix}battle @user`",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    if opponent.bot:
        embed = discord.Embed(
            title="❌ Invalid Opponent",
            description="You can't battle a bot!",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    if opponent.id == ctx.author.id:
        embed = discord.Embed(
            title="❌ Invalid Opponent",
            description="You can't battle yourself!",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    # Get user beasts
    challenger_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    opponent_beasts = await ctx.bot.db.get_user_beasts(opponent.id)

    if not challenger_beasts:
        embed = discord.Embed(
            title="❌ No Beasts",
            description="You don't have any beasts to battle with!",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    if not opponent_beasts:
        embed = discord.Embed(
            title="❌ Opponent Has No Beasts",
            description=f"{opponent.display_name} doesn't have any beasts to battle with!",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    # Let challenger select their beast
    challenger_beast = await select_beast_for_battle(ctx, ctx.author, challenger_beasts, "your")
    if not challenger_beast:
        embed = discord.Embed(
            title="❌ Battle Cancelled",
            description="No beast selected for battle.",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    # Let opponent select their beast
    opponent_beast = await select_beast_for_battle(ctx, opponent, opponent_beasts, f"{opponent.display_name}'s")
    if not opponent_beast:
        embed = discord.Embed(
            title="❌ Battle Cancelled",
            description=f"{opponent.display_name} didn't select a beast.",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    # Simulate the battle
    battle_result = await ctx.bot.battle_engine.simulate_battle(challenger_beast[1], opponent_beast[1])

    # Determine winners and update stats
    challenger_user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    opponent_user = await ctx.bot.get_or_create_user(opponent.id, str(opponent))

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

    embed = discord.Embed(
        title=f"⚔️ Battle Result: {title}",
        color=color
    )

    embed.add_field(
        name="🥊 Fighters",
        value=f"**{ctx.author.display_name}**: {challenger_beast[1].name} {challenger_beast[1].rarity.emoji}\n"
              f"**{opponent.display_name}**: {opponent_beast[1].name} {opponent_beast[1].rarity.emoji}",
        inline=False
    )

    if winner_user:
        embed.add_field(name="🏆 Winner", value=winner_user.display_name, inline=True)
    else:
        embed.add_field(name="🤝 Result", value="Draw", inline=True)

    embed.add_field(name="🔢 Turns", value=battle_result['turns'], inline=True)

    # Final HP
    embed.add_field(
        name="❤️ Final HP",
        value=f"**{challenger_beast[1].name}**: {battle_result['final_hp'][challenger_beast[1].name]}\n"
              f"**{opponent_beast[1].name}**: {battle_result['final_hp'][opponent_beast[1].name]}",
        inline=True
    )

    await ctx.send(embed=embed)


@commands.command(name='setchannel')
@commands.has_permissions(administrator=True)
async def set_spawn_channel(ctx):
    """Set current channel as a spawn channel"""
    ctx.bot.spawn_channels[ctx.channel.id] = None
    embed = discord.Embed(
        title="✅ Spawn Channel Set",
        description=f"{ctx.channel.mention} is now a beast spawn channel!",
        color=0x00FF00
    )
    await ctx.send(embed=embed)

@commands.command(name='removechannel')
@commands.has_permissions(administrator=True)
async def remove_spawn_channel(ctx):
    """Remove current channel from spawn channels"""
    if ctx.channel.id in ctx.bot.spawn_channels:
        del ctx.bot.spawn_channels[ctx.channel.id]
        embed = discord.Embed(
            title="✅ Spawn Channel Removed",
            description=f"{ctx.channel.mention} is no longer a spawn channel.",
            color=0x00FF00
        )
    else:
        embed = discord.Embed(
            title="❌ Not a Spawn Channel",
            description=f"{ctx.channel.mention} was not a spawn channel.",
            color=0xFF0000
        )
    await ctx.send(embed=embed)

@commands.command(name='catch')
async def catch_beast(ctx):
    """Catch a spawned beast"""
    if ctx.channel.id not in ctx.bot.spawn_channels:
        embed = discord.Embed(
            title="❌ No Beast Here",
            description="There's no beast to catch in this channel!",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    beast = ctx.bot.spawn_channels[ctx.channel.id]
    if not beast:
        embed = discord.Embed(
            title="❌ No Beast Here", 
            description="There's no beast to catch in this channel!",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    user = await ctx.bot.get_or_create_user(ctx.author.id, str(ctx.author))
    user_role = ctx.bot.role_manager.get_user_role(ctx.author)
    beast_limit = ctx.bot.role_manager.get_beast_limit(user_role)

    user_beasts = await ctx.bot.db.get_user_beasts(ctx.author.id)
    if len(user_beasts) >= beast_limit:
        embed = discord.Embed(
            title="❌ Beast Inventory Full",
            description=f"Your inventory is full ({len(user_beasts)}/{beast_limit} beasts)!\nUse `{ctx.bot.config.prefix}sacrifice <beast_id>` to make room.",
            color=0xFF0000
        )
        await ctx.send(embed=embed)
        return

    # Calculate catch rate
    catch_rate = beast.rarity.catch_rate
    success = random.randint(1, 100) <= catch_rate

    if success:
        beast.owner_id = ctx.author.id
        beast_id = await ctx.bot.db.add_beast(beast)

        user.total_catches += 1
        await ctx.bot.db.update_user(user)

        # Remove from spawn channels
        del ctx.bot.spawn_channels[ctx.channel.id]

        embed = discord.Embed(
            title="🎉 Beast Caught!",
            description=f"**{ctx.author.display_name}** successfully caught **{beast.name}**!\n{beast.rarity.emoji}",
            color=beast.rarity.color
        )
        embed.add_field(name="Beast ID", value=f"#{beast_id}", inline=True)
        embed.add_field(name="Level", value=beast.stats.level, inline=True)
        embed.add_field(name="Power Level", value=beast.power_level, inline=True)
        embed.add_field(name="Catch Rate", value=f"{catch_rate}%", inline=True)
        embed.add_field(name="Total Beasts", value=f"{len(user_beasts) + 1}/{beast_limit}", inline=True)
    else:
        embed = discord.Embed(
            title="💥 Beast Escaped!",
            description=f"**{beast.name}** broke free and escaped!",
            color=0xFF4444
        )
        embed.add_field(name="Catch Rate", value=f"{catch_rate}%", inline=True)
        embed.add_field(name="Better Luck Next Time!", value="Keep trying!", inline=True)

    await ctx.send(embed=embed)


# ============================================================================
# Flask Web Server (to keep Render happy)
# ============================================================================

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
    bot.add_command(set_spawn_channel)
    bot.add_command(remove_spawn_channel)
    bot.add_command(catch_beast)
    bot.add_command(daily_stone_reward)
    bot.add_command(adopt_beast)
    bot.add_command(show_beasts)
    bot.add_command(help_command)
    bot.add_command(beast_info)
    bot.add_command(set_active_beast)
    bot.add_command(show_balance)
    bot.add_command(battle_command)

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