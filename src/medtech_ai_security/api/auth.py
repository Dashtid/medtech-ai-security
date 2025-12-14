"""
JWT Authentication and Role-Based Access Control for MedTech AI Security API.

This module provides:
- JWT token generation and validation
- Password hashing with bcrypt
- Role-based access control (RBAC)
- API key support for automated tools
- Rate limiting integration

Security considerations:
- Tokens expire after configurable duration
- Passwords are hashed with bcrypt (cost factor 12)
- Secret keys should be provided via environment variables
- Supports both Bearer tokens and API keys
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Annotated, Any

import bcrypt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Secret key for JWT signing - MUST be set in production
SECRET_KEY = os.getenv("MEDSEC_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("MEDSEC_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("MEDSEC_REFRESH_TOKEN_DAYS", "7"))

# OAuth2 scheme for Bearer tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# HTTP Bearer scheme (alternative to OAuth2)
http_bearer = HTTPBearer(auto_error=False)


# =============================================================================
# Enums and Models
# =============================================================================


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"  # Full access to all endpoints
    ANALYST = "analyst"  # Read/write access to analysis endpoints
    VIEWER = "viewer"  # Read-only access
    API_SERVICE = "api_service"  # Automated service access


class TokenType(str, Enum):
    """Types of authentication tokens."""

    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class Token(BaseModel):
    """OAuth2 token response."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int = Field(description="Token expiration in seconds")


class TokenData(BaseModel):
    """Decoded token payload."""

    sub: str  # Subject (username or service name)
    role: UserRole
    token_type: TokenType
    exp: datetime
    iat: datetime
    jti: str | None = None  # JWT ID for token revocation


class User(BaseModel):
    """User model for authentication."""

    username: str
    email: EmailStr | None = None
    full_name: str | None = None
    role: UserRole = UserRole.VIEWER
    disabled: bool = False
    hashed_password: str | None = None


class UserCreate(BaseModel):
    """Request model for user creation."""

    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None
    role: UserRole = UserRole.VIEWER


class UserResponse(BaseModel):
    """Response model for user data (excludes password)."""

    username: str
    email: EmailStr | None
    full_name: str | None
    role: UserRole
    disabled: bool


class APIKeyCreate(BaseModel):
    """Request model for API key creation."""

    name: str = Field(min_length=3, max_length=100)
    role: UserRole = UserRole.API_SERVICE
    expires_days: int = Field(default=365, ge=1, le=3650)


class APIKeyResponse(BaseModel):
    """Response model for API key creation."""

    key: str = Field(description="API key (shown only once)")
    name: str
    role: UserRole
    expires_at: datetime
    key_id: str


# =============================================================================
# In-Memory Storage (Replace with database in production)
# =============================================================================


# Helper function for hashing at module load time
def _hash_password(password: str) -> str:
    """Hash password using bcrypt (for module initialization)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


# Demo users for testing - in production, use a database
DEMO_USERS: dict[str, User] = {
    "admin": User(
        username="admin",
        email="admin@medtech-security.example",
        full_name="Admin User",
        role=UserRole.ADMIN,
        disabled=False,
        hashed_password=_hash_password("admin123secure"),
    ),
    "analyst": User(
        username="analyst",
        email="analyst@medtech-security.example",
        full_name="Security Analyst",
        role=UserRole.ANALYST,
        disabled=False,
        hashed_password=_hash_password("analyst123secure"),
    ),
    "viewer": User(
        username="viewer",
        email="viewer@medtech-security.example",
        full_name="Read-Only User",
        role=UserRole.VIEWER,
        disabled=False,
        hashed_password=_hash_password("viewer123secure"),
    ),
}

# API keys storage - in production, use a database
API_KEYS: dict[str, dict[str, Any]] = {}

# Revoked tokens (for logout) - in production, use Redis or database
REVOKED_TOKENS: set[str] = set()


# =============================================================================
# Password Utilities
# =============================================================================


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=12),
    ).decode("utf-8")


# =============================================================================
# Token Utilities
# =============================================================================


def create_access_token(
    subject: str,
    role: UserRole,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token."""
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    to_encode = {
        "sub": subject,
        "role": role.value,
        "token_type": TokenType.ACCESS.value,
        "exp": expire,
        "iat": now,
        "jti": secrets.token_urlsafe(16),
    }

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(subject: str, role: UserRole) -> str:
    """Create a JWT refresh token."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode = {
        "sub": subject,
        "role": role.value,
        "token_type": TokenType.REFRESH.value,
        "exp": expire,
        "iat": now,
        "jti": secrets.token_urlsafe(16),
    }

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData | None:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Check if token is revoked
        jti = payload.get("jti")
        if jti and jti in REVOKED_TOKENS:
            logger.warning(f"Attempted use of revoked token: {jti[:8]}...")
            return None

        return TokenData(
            sub=payload["sub"],
            role=UserRole(payload["role"]),
            token_type=TokenType(payload["token_type"]),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=jti,
        )
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        return None


def revoke_token(token: str) -> bool:
    """Revoke a token by adding its JTI to the revoked set."""
    token_data = decode_token(token)
    if token_data and token_data.jti:
        REVOKED_TOKENS.add(token_data.jti)
        logger.info(f"Token revoked: {token_data.jti[:8]}...")
        return True
    return False


# =============================================================================
# API Key Utilities
# =============================================================================


def create_api_key(name: str, role: UserRole, expires_days: int = 365) -> APIKeyResponse:
    """Create a new API key."""
    key = f"medsec_{secrets.token_urlsafe(32)}"
    key_id = secrets.token_urlsafe(8)
    expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)

    # Store hashed key
    API_KEYS[key_id] = {
        "name": name,
        "role": role,
        "key_hash": get_password_hash(key),
        "expires_at": expires_at,
        "created_at": datetime.now(timezone.utc),
    }

    logger.info(f"API key created: {name} (ID: {key_id})")

    return APIKeyResponse(
        key=key,
        name=name,
        role=role,
        expires_at=expires_at,
        key_id=key_id,
    )


def validate_api_key(api_key: str) -> TokenData | None:
    """Validate an API key and return token data."""
    for key_id, key_data in API_KEYS.items():
        if verify_password(api_key, key_data["key_hash"]):
            # Check expiration
            if datetime.now(timezone.utc) > key_data["expires_at"]:
                logger.warning(f"Expired API key used: {key_id}")
                return None

            return TokenData(
                sub=f"api:{key_data['name']}",
                role=key_data["role"],
                token_type=TokenType.API_KEY,
                exp=key_data["expires_at"],
                iat=key_data["created_at"],
                jti=key_id,
            )
    return None


# =============================================================================
# User Management
# =============================================================================


def get_user(username: str) -> User | None:
    """Get a user by username."""
    return DEMO_USERS.get(username)


def authenticate_user(username: str, password: str) -> User | None:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not user.hashed_password:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    if user.disabled:
        return None
    return user


def create_user(user_data: UserCreate) -> User:
    """Create a new user."""
    if user_data.username in DEMO_USERS:
        raise ValueError(f"User {user_data.username} already exists")

    user = User(
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        disabled=False,
        hashed_password=get_password_hash(user_data.password),
    )

    DEMO_USERS[user_data.username] = user
    logger.info(f"User created: {user_data.username} (role: {user_data.role})")

    return user


# =============================================================================
# FastAPI Dependencies
# =============================================================================


async def get_token_from_header(
    oauth2_token: Annotated[str | None, Depends(oauth2_scheme)] = None,
    bearer_credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(http_bearer)
    ] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> TokenData:
    """
    Extract and validate authentication from request headers.

    Supports:
    - OAuth2 Bearer token (Authorization: Bearer <token>)
    - HTTP Bearer (Authorization: Bearer <token>)
    - API Key (X-API-Key: <key>)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Try API key first
    if api_key:
        token_data = validate_api_key(api_key)
        if token_data:
            return token_data

    # Try Bearer token
    token = oauth2_token or (bearer_credentials.credentials if bearer_credentials else None)

    if not token:
        raise credentials_exception

    token_data = decode_token(token)
    if not token_data:
        raise credentials_exception

    # Check token type
    if token_data.token_type not in (TokenType.ACCESS, TokenType.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    return token_data


async def get_current_user(
    token_data: Annotated[TokenData, Depends(get_token_from_header)],
) -> User:
    """Get the current authenticated user from token."""
    # For API keys, create a virtual user
    if token_data.token_type == TokenType.API_KEY:
        return User(
            username=token_data.sub,
            role=token_data.role,
            disabled=False,
        )

    user = get_user(token_data.sub)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return user


# =============================================================================
# Role-Based Access Control (RBAC)
# =============================================================================


class RoleChecker:
    """Dependency for checking user roles."""

    def __init__(self, allowed_roles: list[UserRole]) -> None:
        self.allowed_roles = allowed_roles

    async def __call__(
        self,
        user: User = Depends(get_current_user),
    ) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.role.value}' not authorized for this operation",
            )
        return user


# Predefined role checkers
require_admin = RoleChecker([UserRole.ADMIN])
require_analyst = RoleChecker([UserRole.ADMIN, UserRole.ANALYST])
require_viewer = RoleChecker([UserRole.ADMIN, UserRole.ANALYST, UserRole.VIEWER])
require_any_authenticated = RoleChecker(list(UserRole))


# =============================================================================
# Optional Authentication (for public endpoints with enhanced access)
# =============================================================================


async def get_optional_user(
    oauth2_token: Annotated[str | None, Depends(oauth2_scheme)] = None,
    bearer_credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(http_bearer)
    ] = None,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
) -> User | None:
    """
    Get current user if authenticated, otherwise return None.

    Useful for endpoints that work for anonymous users but provide
    enhanced functionality for authenticated users.
    """
    try:
        token_data = await get_token_from_header(oauth2_token, bearer_credentials, api_key)
        return await get_current_user(token_data)
    except HTTPException:
        return None
