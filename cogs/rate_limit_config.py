import discord
from discord.ext import commands
import logging
import asyncio
from utils.guild_rate_manager import GuildRateLimitManager, GuildRateLimits


class RateLimitConfig(commands.Cog):
    """Configuration commands for rate limiting system"""

    def __init__(self, bot, guild_manager: GuildRateLimitManager):  # Fixed __init__
        self.bot = bot
        self.guild_manager = guild_manager

    @commands.group(name='ratelimit', aliases=['rl'])
    @commands.has_permissions(administrator=True)
    async def ratelimit_group(self, ctx):
        """Rate limit configuration commands"""
        if ctx.invoked_subcommand is None:
            embed = discord.Embed(title="Rate Limit Configuration",
                                  description="Available commands:",
                                  color=0x3498db)
            embed.add_field(name="View Settings",
                            value="`rl show` - Show current rate limits",
                            inline=False)
            embed.add_field(
                name="Update Settings",
                value=("`rl set <setting> <value>` - Update a rate limit\n"
                       "`rl reset` - Reset to default values"),
                inline=False)
            embed.add_field(
                name="Available Settings",
                value=
                ("• `xp_cooldown` - XP gain cooldown (seconds)\n"
                 "• `daily_stones_cooldown` - Daily stones cooldown (seconds)\n"
                 "• `beast_adoption_cooldown` - Beast adoption cooldown (seconds)\n"
                 "• `max_catch_attempts` - Max beast catch attempts\n"
                 "• `api_requests_per_minute` - API requests per minute\n"
                 "• `battle_timeout` - Battle selection timeout (seconds)"),
                inline=False)
            await ctx.send(embed=embed)

    @ratelimit_group.command(name='show')
    async def show_ratelimits(self, ctx):
        """Show current rate limit settings for this guild"""
        limits = self.guild_manager.get_guild_limits(ctx.guild.id)
        embed = discord.Embed(title=f"Rate Limits for {ctx.guild.name}",
                              color=0x2ecc71)
        embed.add_field(
            name="User Cooldowns",
            value=
            (f"**XP Cooldown:** {limits.xp_cooldown}s\n"
             f"**Daily Stones:** {limits.daily_stones_cooldown}s ({limits.daily_stones_cooldown//3600}h)\n"
             f"**Beast Adoption:** {limits.beast_adoption_cooldown}s ({limits.beast_adoption_cooldown//3600}h)"
             ),
            inline=False)
        embed.add_field(
            name="System Limits",
            value=(f"**Max Catch Attempts:** {limits.max_catch_attempts}\n"
                   f"**API Requests/Min:** {limits.api_requests_per_minute}\n"
                   f"**Battle Timeout:** {limits.battle_timeout}s"),
            inline=False)
        await ctx.send(embed=embed)

    @ratelimit_group.command(name='set')
    async def set_ratelimit(self, ctx, setting: str, value: int):
        """Set a rate limit value"""
        valid_settings = [
            'xp_cooldown', 'daily_stones_cooldown', 'beast_adoption_cooldown',
            'max_catch_attempts', 'api_requests_per_minute', 'battle_timeout'
        ]

        if setting not in valid_settings:
            await ctx.send(
                f"❌ Invalid setting. Valid options: {', '.join(valid_settings)}"
            )
            return

        # Validation
        if value < 0:
            await ctx.send("❌ Value must be non-negative")
            return

        if setting == 'api_requests_per_minute' and (value < 1 or value > 300):
            await ctx.send(
                "❌ API requests per minute must be between 1 and 300")
            return

        if setting in ['xp_cooldown', 'battle_timeout'] and value > 3600:
            await ctx.send(f"❌ {setting} cannot exceed 1 hour (3600 seconds)")
            return

        # Update setting
        self.guild_manager.update_guild_limits(ctx.guild.id, **{setting: value})

        embed = discord.Embed(
            title="✅ Rate Limit Updated",
            description=f"**{setting}** set to **{value}** for {ctx.guild.name}",
            color=0x2ecc71)
        await ctx.send(embed=embed)

        # Log the change
        logging.info(
            f"Rate limit updated in {ctx.guild.name} ({ctx.guild.id}): {setting} = {value} by {ctx.author}"
        )

    @ratelimit_group.command(name='reset')
    async def reset_ratelimits(self, ctx):
        """Reset rate limits to default values"""
        # Remove custom settings for this guild
        if ctx.guild.id in self.guild_manager.guild_limits:
            del self.guild_manager.guild_limits[ctx.guild.id]
        if ctx.guild.id in self.guild_manager.guild_rate_limiters:
            del self.guild_manager.guild_rate_limiters[ctx.guild.id]

        self.guild_manager.save_config()

        embed = discord.Embed(
            title="✅ Rate Limits Reset",
            description=f"All rate limits for {ctx.guild.name} have been reset to default values",
            color=0x2ecc71)
        await ctx.send(embed=embed)

    @ratelimit_group.command(name='template')
    async def create_template(self, ctx, template_name: str):
        """Create rate limit templates for different server sizes"""
        templates = {
            'small': GuildRateLimits(
                xp_cooldown=45,
                api_requests_per_minute=30,
                max_catch_attempts=5
            ),
            'medium': GuildRateLimits(
                xp_cooldown=60,
                api_requests_per_minute=60,
                max_catch_attempts=3
            ),
            'large': GuildRateLimits(
                xp_cooldown=90,
                api_requests_per_minute=100,
                max_catch_attempts=2
            ),
            'mega': GuildRateLimits(
                xp_cooldown=120,
                api_requests_per_minute=150,
                max_catch_attempts=1
            )
        }

        if template_name.lower() not in templates:
            await ctx.send(
                f"❌ Invalid template. Available: {', '.join(templates.keys())}"
            )
            return

        template = templates[template_name.lower()]

        # Apply template settings
        for setting, value in template.to_dict().items():
            self.guild_manager.update_guild_limits(ctx.guild.id, **{setting: value})

        embed = discord.Embed(
            title="✅ Template Applied",
            description=f"Applied **{template_name}** server template to {ctx.guild.name}",
            color=0x2ecc71)
        await ctx.send(embed=embed)

    # Enhanced safe API call method using the new system (moved inside class)
    async def safe_api_call(self, guild_id: int, api_func, *args, **kwargs):
        """Enhanced safe API call with per-guild rate limiting"""
        rate_limiter = await self.guild_manager.get_api_rate_limiter(guild_id)
        max_retries = 5
        base_delay = 2

        for attempt in range(max_retries):
            # Wait for rate limit slot
            wait_time = await rate_limiter.wait_if_needed()
            if wait_time:
                logging.warning(
                    f"Rate limited for {wait_time:.2f}s in guild {guild_id}")

            try:
                result = await api_func(*args, **kwargs)
                return result

            except discord.HTTPException as e:
                if e.status == 429:  # Rate limited
                    # Fixed retry_after access
                    retry_after = getattr(e, 'retry_after', base_delay * (2**attempt))
                    logging.warning(
                        f"Hit 429 rate limit, waiting {retry_after}s (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(retry_after)
                else:
                    raise e

        raise Exception(f"Max retries ({max_retries}) exceeded for API call")
