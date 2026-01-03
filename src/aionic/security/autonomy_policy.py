"""
Autonomy Policy - Risk-aware execution permission system.

Defines risk tiers and permission rules for agent actions,
ensuring safe and controlled execution.
"""

from enum import Enum
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


class RiskTier(Enum):
    """Risk tiers for actions and tools."""
    NONE = "none"           # No risk
    LOW = "low"             # Minimal risk
    MEDIUM = "medium"       # Moderate risk
    HIGH = "high"           # Significant risk
    CRITICAL = "critical"   # Critical risk - requires approval


@dataclass
class PermissionRule:
    """
    Permission rule for agent-tool combinations.
    
    Attributes:
        agent_role: Minimum agent role required
        min_competence: Minimum competence score required
        risk_tier: Maximum risk tier allowed
        requires_approval: Whether explicit approval needed
    """
    
    agent_role: str  # Minimum role
    min_competence: float  # Minimum competence (0.0-1.0)
    risk_tier: RiskTier
    requires_approval: bool = False


class AutonomyPolicy:
    """
    Policy engine for agent autonomy and permission management.
    
    Determines what actions agents can perform based on:
    - Agent role and competence
    - Action/tool risk tier
    - Historical performance
    - Policy rules
    """
    
    def __init__(self):
        """Initialize autonomy policy with default rules."""
        # Default permission matrix: role -> max risk tier allowed
        self._role_permissions: Dict[str, RiskTier] = {
            "junior": RiskTier.LOW,
            "associate": RiskTier.MEDIUM,
            "senior": RiskTier.HIGH,
            "expert": RiskTier.CRITICAL,
            "architect": RiskTier.CRITICAL,
        }
        
        # Competence thresholds for risk tiers
        self._competence_thresholds: Dict[RiskTier, float] = {
            RiskTier.NONE: 0.0,
            RiskTier.LOW: 0.3,
            RiskTier.MEDIUM: 0.5,
            RiskTier.HIGH: 0.7,
            RiskTier.CRITICAL: 0.9,
        }
        
        # Explicit approval requirements
        self._requires_approval: Set[RiskTier] = {RiskTier.CRITICAL}
        
        # Blacklisted agents (cannot execute high-risk actions)
        self._blacklisted_agents: Set[str] = set()
        
        # Custom rules for specific agent-tool combinations
        self._custom_rules: Dict[str, PermissionRule] = {}  # "agent_id:tool_name" -> rule
    
    def can_execute_tool(
        self,
        agent_id: str,
        tool_name: str,
        tool_risk_tier: str,
        agent_role: str,
        agent_competence: float,
    ) -> bool:
        """
        Check if agent can execute a tool.
        
        Args:
            agent_id: Agent identifier
            tool_name: Tool name
            tool_risk_tier: Tool's risk tier (string)
            agent_role: Agent's current role
            agent_competence: Agent's competence score
            
        Returns:
            True if execution is permitted, False otherwise
        """
        # Check blacklist
        if agent_id in self._blacklisted_agents:
            return False
        
        # Convert string risk tier to enum
        try:
            risk_tier = RiskTier(tool_risk_tier.lower())
        except ValueError:
            # Unknown risk tier - default to HIGH for safety
            risk_tier = RiskTier.HIGH
        
        # Check custom rule first
        rule_key = f"{agent_id}:{tool_name}"
        if rule_key in self._custom_rules:
            rule = self._custom_rules[rule_key]
            if agent_role < rule.agent_role or agent_competence < rule.min_competence:
                return False
            if risk_tier.value > rule.risk_tier.value:
                return False
            return True
        
        # Check role-based permissions
        max_allowed_risk = self._role_permissions.get(agent_role.lower(), RiskTier.LOW)
        
        # Compare risk tiers (enum values are ordered)
        risk_values = {
            RiskTier.NONE: 0,
            RiskTier.LOW: 1,
            RiskTier.MEDIUM: 2,
            RiskTier.HIGH: 3,
            RiskTier.CRITICAL: 4,
        }
        
        if risk_values[risk_tier] > risk_values[max_allowed_risk]:
            return False
        
        # Check competence threshold
        required_competence = self._competence_thresholds.get(risk_tier, 1.0)
        if agent_competence < required_competence:
            return False
        
        return True
    
    def requires_approval(
        self,
        agent_id: str,
        tool_name: str,
        tool_risk_tier: str,
    ) -> bool:
        """
        Check if execution requires explicit approval.
        
        Args:
            agent_id: Agent identifier
            tool_name: Tool name
            tool_risk_tier: Tool's risk tier
            
        Returns:
            True if approval required, False otherwise
        """
        try:
            risk_tier = RiskTier(tool_risk_tier.lower())
            return risk_tier in self._requires_approval
        except ValueError:
            return True  # Err on side of caution
    
    def add_custom_rule(
        self,
        agent_id: str,
        tool_name: str,
        rule: PermissionRule,
    ) -> None:
        """
        Add custom permission rule for specific agent-tool combination.
        
        Args:
            agent_id: Agent identifier
            tool_name: Tool name
            rule: Permission rule
        """
        rule_key = f"{agent_id}:{tool_name}"
        self._custom_rules[rule_key] = rule
    
    def blacklist_agent(self, agent_id: str) -> None:
        """Add agent to blacklist."""
        self._blacklisted_agents.add(agent_id)
    
    def whitelist_agent(self, agent_id: str) -> None:
        """Remove agent from blacklist."""
        self._blacklisted_agents.discard(agent_id)
    
    def update_role_permission(
        self,
        role: str,
        max_risk_tier: RiskTier,
    ) -> None:
        """
        Update maximum risk tier for a role.
        
        Args:
            role: Agent role
            max_risk_tier: Maximum allowed risk tier
        """
        self._role_permissions[role.lower()] = max_risk_tier
    
    def get_policy_summary(self) -> Dict:
        """Get summary of current policy configuration."""
        return {
            "role_permissions": {
                role: tier.value for role, tier in self._role_permissions.items()
            },
            "competence_thresholds": {
                tier.value: threshold
                for tier, threshold in self._competence_thresholds.items()
            },
            "requires_approval": [tier.value for tier in self._requires_approval],
            "blacklisted_agents": list(self._blacklisted_agents),
            "custom_rules_count": len(self._custom_rules),
        }

